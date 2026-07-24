"""
evaluate.py
-----------
Comprehensive evaluation suite for the Bayesian Network.


Metrics
-------
- Top-k Accuracy (k=1, 3, 5)
- Per-class Precision, Recall, F1
- Confusion Matrix
- K-fold Cross-Validation
- Calibration analysis (how well do predicted probabilities match reality)
"""


import numpy as np
import pandas as pd
from collections import defaultdict


from src.network_structure import COLUMN_MAP, TOPOLOGICAL_ORDER, DISPOSITION_LABELS
from src.cpt_builder import CPTBuilder
from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler



def build_evidence(row: pd.Series, exclude_var: str = "final_disposition") -> dict:
    """
    Build an evidence dict from a DataFrame row, excluding the query variable.


    Parameters
    ----------
    row         : pd.Series — one case's data
    exclude_var : str — the variable to predict (must not appear in evidence)


    Returns
    -------
    dict {node_name: value}
    """
    evidence = {}
    for node_name, col_name in COLUMN_MAP.items():
        if node_name == exclude_var:
            continue
        if col_name in row.index and pd.notna(row[col_name]):
            evidence[node_name] = row[col_name]
    return evidence



def top_k_accuracy(
    df_test: pd.DataFrame,
    sampler: RejectionSampler,
    k: int = 3,
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    verbose: bool = True,
) -> float:
    """
    Compute Top-k accuracy on a test set.


    Parameters
    ----------
    df_test   : pd.DataFrame
    sampler   : RejectionSampler
    k         : int — top-k to consider correct
    n_samples : int — rejection samples per case
    query_var : str — variable to predict
    verbose   : bool — print per-case results


    Returns
    -------
    float — accuracy in [0, 1]
    """
    target_col = COLUMN_MAP.get(query_var, query_var)
    n_correct = 0
    n_total = 0


    for i, (_, row) in enumerate(df_test.iterrows()):
        if target_col not in row.index or pd.isna(row[target_col]):
            continue


        true_val = row[target_col]
        evidence = build_evidence(row, exclude_var=query_var)


        top_preds = sampler.top_k_predictions(
            query_var=query_var,
            evidence=evidence,
            k=k,
            n_samples=n_samples,
        )
        pred_values = [v for v, _ in top_preds]


        correct = true_val in pred_values
        if correct:
            n_correct += 1
        n_total += 1


        if verbose:
            label = DISPOSITION_LABELS.get(int(true_val), str(true_val)) if true_val else "?"
            print(f"  Case {i+1:>3}: true={int(true_val)} ({label[:30]})  "
                  f"top-{k}={[int(p) for p in pred_values]}  "
                  f"{'✓' if correct else '✗'}")


    accuracy = n_correct / n_total if n_total > 0 else 0.0
    print(f"\n  Top-{k} Accuracy: {n_correct}/{n_total} = {accuracy*100:.1f}%")
    return accuracy



def distribution_summary(
    sampler,
    evidence: dict,
    n_samples: int = 5000,
    query_var: str = "final_disposition",
):
    """Print the full predicted probability distribution for a single case."""
    dist = sampler.query(query_var, evidence, n_samples=n_samples)
    print(f"\n  Predicted distribution for {query_var}:")
    for val, prob in sorted(dist.items(), key=lambda x: -x[1]):
        label = DISPOSITION_LABELS.get(int(val), str(val)) if val else str(val)
        bar = "█" * int(prob * 40)
        print(f"    {int(val):>2}: {label[:35]:<35} {prob*100:5.1f}%  {bar}")



# ---------------------------------------------------------------------------
# Multi-k accuracy
# ---------------------------------------------------------------------------


def multi_k_accuracy(
    df_test: pd.DataFrame,
    sampler,
    ks: list[int] = (1, 3, 5),
    n_samples: int = 1000,
    query_var: str = "final_disposition",
) -> dict[int, float]:
    """
    Compute Top-k accuracy for multiple values of k simultaneously.


    Returns
    -------
    dict — {k: accuracy}
    """
    target_col = COLUMN_MAP.get(query_var, query_var)
    correct = {k: 0 for k in ks}
    n_total = 0


    for _, row in df_test.iterrows():
        if target_col not in row.index or pd.isna(row[target_col]):
            continue


        true_val = row[target_col]
        evidence = build_evidence(row, exclude_var=query_var)
        dist = sampler.query(query_var, evidence, n_samples=n_samples)
        sorted_preds = sorted(dist.items(), key=lambda x: -x[1])


        for k in ks:
            top_vals = [v for v, _ in sorted_preds[:k]]
            if true_val in top_vals:
                correct[k] += 1
        n_total += 1


    return {k: c / max(1, n_total) for k, c in correct.items()}



# ---------------------------------------------------------------------------
# Per-class precision / recall / F1
# ---------------------------------------------------------------------------


def classification_report(
    df_test: pd.DataFrame,
    sampler,
    n_samples: int = 1000,
    query_var: str = "final_disposition",
) -> dict:
    """
    Compute per-class precision, recall, and F1 for Top-1 predictions.


    Returns
    -------
    dict with keys: "per_class" (dict of class → metrics), "macro_f1", "accuracy"
    """
    target_col = COLUMN_MAP.get(query_var, query_var)


    true_labels = []
    pred_labels = []


    for _, row in df_test.iterrows():
        if target_col not in row.index or pd.isna(row[target_col]):
            continue


        true_val = row[target_col]
        evidence = build_evidence(row, exclude_var=query_var)
        preds = sampler.top_k_predictions(query_var, evidence, k=1, n_samples=n_samples)


        true_labels.append(true_val)
        pred_labels.append(preds[0][0] if preds else None)


    # Compute per-class metrics
    all_classes = sorted(set(true_labels) | set(pred_labels) - {None})
    per_class = {}


    for cls in all_classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)


        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


        label = DISPOSITION_LABELS.get(int(cls), str(cls)) if cls else str(cls)
        per_class[cls] = {
            "label": label,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for t in true_labels if t == cls),
        }


    accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / max(1, len(true_labels))
    f1_scores = [m["f1"] for m in per_class.values() if m["support"] > 0]
    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))


    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 3),
        "accuracy": round(accuracy, 3),
        "n_test": len(true_labels),
    }



def print_classification_report(report: dict):
    """Pretty-print the classification report."""
    print(f"\n── Classification Report (n={report['n_test']}) ──────────────────")
    print(f"  {'Class':<40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*70}")


    for cls in sorted(report["per_class"].keys()):
        m = report["per_class"][cls]
        print(f"  {m['label'][:40]:<40} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")


    print(f"  {'─'*70}")
    print(f"  {'Macro F1':<40} {'':>6} {'':>6} {report['macro_f1']:>6.3f}")
    print(f"  {'Top-1 Accuracy':<40} {'':>6} {'':>6} {report['accuracy']:>6.3f}")



# ---------------------------------------------------------------------------
# K-Fold Cross-Validation
# ---------------------------------------------------------------------------


def k_fold_cross_validation(
    df: pd.DataFrame,
    n_folds: int = 5,
    k: int = 3,
    n_samples: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
    use_likelihood_weighting: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run k-fold cross-validation.


    Parameters
    ----------
    df       : full dataset
    n_folds  : number of folds
    k        : top-k for accuracy computation
    n_samples: rejection/LW samples per case
    alpha    : Laplace smoothing
    seed     : random seed
    use_likelihood_weighting : use LW instead of rejection sampling
    verbose  : print per-fold results


    Returns
    -------
    dict with "fold_accuracies", "mean", "std"
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    fold_size = len(df) // n_folds


    fold_accuracies = []


    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(df)
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])


        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]


        builder = CPTBuilder(alpha=alpha)
        builder.fit(df_train)


        if use_likelihood_weighting:
            sampler = LikelihoodWeightingSampler(builder, random_state=seed + fold)
        else:
            sampler = RejectionSampler(builder, random_state=seed + fold)


        acc = top_k_accuracy(
            df_test, sampler, k=k, n_samples=n_samples, verbose=False
        )
        fold_accuracies.append(acc)


        if verbose:
            print(f"  Fold {fold+1}/{n_folds}: Top-{k} = {acc*100:.1f}% "
                  f"(train={len(df_train)}, test={len(df_test)})")


    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)


    if verbose:
        print(f"\n  {n_folds}-Fold CV: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")


    return {
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "mean": round(mean_acc, 4),
        "std": round(std_acc, 4),
    }



# ---------------------------------------------------------------------------
# Compare inference methods
# ---------------------------------------------------------------------------


def compare_inference_methods(
    df_test: pd.DataFrame,
    builder: CPTBuilder,
    k: int = 3,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compare Rejection Sampling vs Likelihood Weighting on the same test set.


    Returns
    -------
    dict with accuracy and timing for each method
    """
    import time


    results = {}
    for name, SamplerClass in [
        ("Rejection Sampling", RejectionSampler),
        ("Likelihood Weighting", LikelihoodWeightingSampler),
        ("Gibbs Sampling (MCMC)", GibbsSampler),
    ]:
        sampler = SamplerClass(builder, random_state=seed)
        t0 = time.perf_counter()
        acc = top_k_accuracy(df_test, sampler, k=k, n_samples=n_samples, verbose=False)
        elapsed = time.perf_counter() - t0


        results[name] = {"accuracy": round(acc, 4), "time_seconds": round(elapsed, 2)}
        print(f"  {name:<25} Top-{k}: {acc*100:.1f}%  ({elapsed:.1f}s)")


    return results



# ---------------------------------------------------------------------------
# Calibration Analysis
# ---------------------------------------------------------------------------


def calibration_analysis(
    df_test: pd.DataFrame,
    sampler,
    n_samples: int = 1000,
    n_bins: int = 10,
    query_var: str = "final_disposition",
) -> dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.


    A well-calibrated model should have P(correct | confidence = c) ≈ c.
    ECE measures the weighted average absolute gap between predicted
    confidence and actual accuracy across confidence bins.


    Parameters
    ----------
    df_test   : test DataFrame
    sampler   : inference engine
    n_samples : samples per case
    n_bins    : number of calibration bins
    query_var : target variable


    Returns
    -------
    dict with "ece", "bin_edges", "bin_accuracies", "bin_confidences", "bin_counts"
    """
    target_col = COLUMN_MAP.get(query_var, query_var)


    confidences = []
    correct = []


    for _, row in df_test.iterrows():
        if target_col not in row.index or pd.isna(row[target_col]):
            continue


        true_val = row[target_col]
        evidence = build_evidence(row, exclude_var=query_var)
        dist = sampler.query(query_var, evidence, n_samples=n_samples)


        if not dist:
            continue


        # Top-1 prediction and its confidence
        top_val, top_prob = max(dist.items(), key=lambda x: x[1])
        confidences.append(top_prob)
        correct.append(1.0 if top_val == true_val else 0.0)


    if not confidences:
        return {"ece": 0.0, "bin_edges": [], "bin_accuracies": [],
                "bin_confidences": [], "bin_counts": []}


    confidences = np.array(confidences)
    correct = np.array(correct)


    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    ece = 0.0


    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 \
            else (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts.append(int(count))


        if count > 0:
            avg_acc = correct[mask].mean()
            avg_conf = confidences[mask].mean()
            bin_accuracies.append(float(avg_acc))
            bin_confidences.append(float(avg_conf))
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((lo + hi) / 2)


    ece /= len(confidences)


    return {
        "ece": round(ece, 4),
        "bin_edges": [round(e, 2) for e in bin_edges.tolist()],
        "bin_accuracies": [round(a, 4) for a in bin_accuracies],
        "bin_confidences": [round(c, 4) for c in bin_confidences],
        "bin_counts": bin_counts,
        "overall_accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
    }



def print_calibration_report(cal: dict):
    """Pretty-print the calibration analysis."""
    print(f"\n── Calibration Analysis ─────────────────────────────────")
    print(f"  Expected Calibration Error (ECE): {cal['ece']:.4f}")
    print(f"  Mean Confidence:  {cal.get('mean_confidence', 0):.4f}")
    print(f"  Overall Accuracy: {cal.get('overall_accuracy', 0):.4f}")


    if cal["bin_counts"]:
        print(f"\n  {'Bin':>10} {'Confidence':>12} {'Accuracy':>10} {'Count':>7} {'Gap':>7}")
        print(f"  {'─' * 50}")
        edges = cal["bin_edges"]
        for i, (conf, acc, cnt) in enumerate(
            zip(cal["bin_confidences"], cal["bin_accuracies"], cal["bin_counts"])
        ):
            if cnt > 0:
                gap = abs(acc - conf)
                lo, hi = edges[i], edges[i + 1]
                print(f"  {lo:.1f}-{hi:.1f}   {conf:>10.3f}   {acc:>8.3f}   {cnt:>5}   {gap:>5.3f}")