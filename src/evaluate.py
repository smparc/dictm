"""
evaluate.py
-----------
Evaluation suite for the Bayesian Network.

Design note
-----------
Every metric here derives from one object: a matrix `P` of shape
(n_test_cases, n_classes) holding the posterior each engine assigns to each
case, computed once by `predict_distributions`. Previously top-k accuracy, the
classification report and the calibration analysis each re-ran inference over
the whole test set, so a single evaluation performed four full inference passes.
With a real test set (~1,800 cases rather than 30) that difference is the
difference between seconds and minutes.

Metrics
-------
- Top-k accuracy (k = 1, 3, 5)
- Log-loss and Brier score — *proper scoring rules*. Unlike top-k accuracy they
  reward calibrated probabilities and cannot be gamed by always predicting the
  most common classes, which is exactly the failure mode this project needs to
  be able to detect.
- Macro-averaged one-vs-rest AUC
- Per-class precision / recall / F1
- Expected Calibration Error, with reliability-diagram bins
- Bootstrap 95% confidence intervals on all of the above
- K-fold cross-validation
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src.cpt_builder import CPTBuilder, _sort_key
from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler
from src.network_structure import (
    COLUMN_MAP,
    DISPOSITION_LABELS,
    FEATURE_SETS,
    BINARY_LABELS,
    to_binary_disposition,
)

log = logging.getLogger(__name__)

EPSILON = 1e-12


def build_evidence(
    row: pd.Series,
    exclude_var: str = "final_disposition",
    evidence_vars=None,
) -> dict:
    """
    Build an evidence dict from a DataFrame row, excluding the query variable.

    Parameters
    ----------
    row           : pd.Series — one case's data
    exclude_var   : str — the variable to predict (must not appear in evidence)
    evidence_vars : iterable of node names, or a key of FEATURE_SETS
                    ("explanatory" / "ex_ante"). When None, every mapped column
                    is used.

                    This is the single switch that selects an evidence policy.
                    The `explanatory` set includes variables recorded *from* the
                    decision (decision_type, split_vote, unconstitutional,
                    precedent_alteration), so a model evaluated with it is
                    describing, not forecasting. The `ex_ante` set is restricted
                    to what is knowable before the Court rules.

    Returns
    -------
    dict {node_name: value}
    """
    if isinstance(evidence_vars, str):
        evidence_vars = FEATURE_SETS[evidence_vars]
    allowed = set(evidence_vars) if evidence_vars is not None else None

    evidence = {}
    for node_name, col_name in COLUMN_MAP.items():
        if node_name == exclude_var:
            continue
        if allowed is not None and node_name not in allowed:
            continue
        if col_name in row.index and pd.notna(row[col_name]):
            evidence[node_name] = row[col_name]
    return evidence


# ---------------------------------------------------------------------------
# One inference pass -> a posterior matrix
# ---------------------------------------------------------------------------


def predict_distributions(
    df_test: pd.DataFrame,
    engine,
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    evidence_vars=None,
    classes: list | None = None,
    binary: bool = False,
):
    """
    Run inference once per test case and assemble the posterior matrix.

    Parameters
    ----------
    df_test       : test set
    engine        : anything exposing `query(query_var, evidence, n_samples)` —
                    the samplers, the exact engine, or any baseline
    binary        : collapse onto the affirm/reverse task, dropping cases whose
                    disposition belongs to neither class

    Returns
    -------
    (y_true, P, classes)
        y_true  : ndarray (n,) of true class values
        P       : ndarray (n, n_classes), rows summing to 1
        classes : list of class values, matching P's columns
    """
    target_col = COLUMN_MAP.get(query_var, query_var)

    raw_true = []
    raw_dists = []

    for _, row in df_test.iterrows():
        if target_col not in row.index or pd.isna(row[target_col]):
            continue

        true_val = row[target_col]
        if binary:
            mapped = to_binary_disposition(true_val)
            if mapped is None:
                continue
            true_val = mapped

        evidence = build_evidence(row, exclude_var=query_var, evidence_vars=evidence_vars)
        dist = engine.query(query_var, evidence, n_samples=n_samples)
        if not dist:
            continue

        if binary:
            collapsed = defaultdict(float)
            for value, prob in dist.items():
                mapped = to_binary_disposition(value)
                if mapped is not None:
                    collapsed[mapped] += prob
            total = sum(collapsed.values())
            if total <= 0:
                collapsed = {0: 0.5, 1: 0.5}
                total = 1.0
            dist = {k: v / total for k, v in collapsed.items()}

        raw_true.append(true_val)
        raw_dists.append(dist)

    if classes is None:
        observed = set(raw_true)
        for dist in raw_dists:
            observed.update(dist)
        classes = sorted(observed, key=_sort_key)

    index = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(raw_dists), len(classes)), dtype=float)
    for i, dist in enumerate(raw_dists):
        for value, prob in dist.items():
            if value in index:
                P[i, index[value]] = prob

    totals = P.sum(axis=1, keepdims=True)
    P = np.where(totals > 0, P / np.where(totals > 0, totals, 1.0), 1.0 / len(classes))

    return np.array(raw_true), P, classes


# ---------------------------------------------------------------------------
# Metrics computed from (y_true, P)
# ---------------------------------------------------------------------------


def top_k_from_probs(y_true: np.ndarray, P: np.ndarray, classes: list, k: int) -> float:
    """Fraction of cases whose true class is among the k highest-probability classes."""
    if len(y_true) == 0:
        return 0.0
    k = min(k, P.shape[1])
    top = np.argsort(-P, axis=1)[:, :k]
    class_array = np.array(classes, dtype=object)
    hits = [y_true[i] in set(class_array[top[i]]) for i in range(len(y_true))]
    return float(np.mean(hits))


def log_loss_from_probs(y_true: np.ndarray, P: np.ndarray, classes: list) -> float:
    """
    Mean negative log-likelihood of the true class.

    A proper scoring rule: minimised only by reporting one's true beliefs. Lower
    is better; the marginal baseline's log-loss is the entropy of the class
    distribution, which is the number any real model must beat.
    """
    if len(y_true) == 0:
        return float("nan")
    index = {c: i for i, c in enumerate(classes)}
    picked = np.array([P[i, index[y]] if y in index else 0.0 for i, y in enumerate(y_true)])
    return float(-np.mean(np.log(np.clip(picked, EPSILON, 1.0))))


def brier_from_probs(y_true: np.ndarray, P: np.ndarray, classes: list) -> float:
    """Multi-class Brier score: mean squared error against the one-hot truth."""
    if len(y_true) == 0:
        return float("nan")
    index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(P)
    for i, y in enumerate(y_true):
        if y in index:
            onehot[i, index[y]] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def macro_auc_from_probs(y_true: np.ndarray, P: np.ndarray, classes: list) -> float:
    """
    Macro-averaged one-vs-rest ROC AUC.

    Returns NaN when it is undefined (fewer than two classes present, or
    scikit-learn unavailable).
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")

    present = [c for c in classes if np.any(y_true == c)]
    if len(present) < 2:
        return float("nan")

    index = {c: i for i, c in enumerate(classes)}
    scores = []
    for c in present:
        binary_truth = (y_true == c).astype(int)
        if binary_truth.min() == binary_truth.max():
            continue
        try:
            scores.append(roc_auc_score(binary_truth, P[:, index[c]]))
        except ValueError:
            continue
    return float(np.mean(scores)) if scores else float("nan")


def ece_from_probs(y_true: np.ndarray, P: np.ndarray, classes: list, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error plus reliability-diagram bins.

    A well-calibrated model satisfies P(correct | confidence = c) ~= c. ECE is
    the count-weighted mean gap between confidence and accuracy across bins.
    """
    if len(y_true) == 0:
        return {"ece": 0.0, "bin_edges": [], "bin_accuracies": [],
                "bin_confidences": [], "bin_counts": []}

    class_array = np.array(classes, dtype=object)
    predicted = class_array[np.argmax(P, axis=1)]
    confidences = P.max(axis=1)
    correct = np.array([1.0 if predicted[i] == y_true[i] else 0.0 for i in range(len(y_true))])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies, bin_confidences, bin_counts = [], [], []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        count = int(mask.sum())
        bin_counts.append(count)
        if count > 0:
            avg_acc = float(correct[mask].mean())
            avg_conf = float(confidences[mask].mean())
            bin_accuracies.append(avg_acc)
            bin_confidences.append(avg_conf)
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((lo + hi) / 2))

    return {
        "ece": round(ece / len(confidences), 4),
        "bin_edges": [round(float(e), 2) for e in bin_edges],
        "bin_accuracies": [round(a, 4) for a in bin_accuracies],
        "bin_confidences": [round(c, 4) for c in bin_confidences],
        "bin_counts": bin_counts,
        "overall_accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    P: np.ndarray,
    classes: list,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a metric computed from (y_true, P).

    Resampling operates on the cached posterior matrix, so this costs no
    additional inference. An accuracy reported without an interval invites the
    reader to over-read a difference that a test set of this size cannot
    support, which is precisely how this project's original 73.3% came to be
    compared against the wrong baseline.
    """
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = metric_fn(y_true[idx], P[idx], classes)
        if not np.isnan(value):
            stats.append(value)

    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (round(lo, 4), round(hi, 4))


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    df_test: pd.DataFrame,
    engine,
    name: str = "model",
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    evidence_vars=None,
    ks=(1, 3, 5),
    binary: bool = False,
    n_boot: int = 500,
    seed: int = 42,
    classes: list | None = None,
) -> dict:
    """
    Run one inference pass and report every metric, each with a bootstrap CI.

    Returns
    -------
    dict suitable for `print_results_table`.
    """
    y_true, P, classes = predict_distributions(
        df_test, engine,
        n_samples=n_samples, query_var=query_var,
        evidence_vars=evidence_vars, classes=classes, binary=binary,
    )

    result = {
        "name": name,
        "n_test": int(len(y_true)),
        "n_classes": len(classes),
        "classes": classes,
        "binary": binary,
    }

    for k in ks:
        if k > len(classes):
            continue
        point = top_k_from_probs(y_true, P, classes, k)
        ci = bootstrap_ci(
            lambda yt, pp, cl, _k=k: top_k_from_probs(yt, pp, cl, _k),
            y_true, P, classes, n_boot=n_boot, seed=seed,
        )
        result[f"top_{k}"] = round(point, 4)
        result[f"top_{k}_ci"] = ci

    result["log_loss"] = round(log_loss_from_probs(y_true, P, classes), 4)
    result["log_loss_ci"] = bootstrap_ci(log_loss_from_probs, y_true, P, classes,
                                         n_boot=n_boot, seed=seed)
    result["brier"] = round(brier_from_probs(y_true, P, classes), 4)
    result["brier_ci"] = bootstrap_ci(brier_from_probs, y_true, P, classes,
                                      n_boot=n_boot, seed=seed)

    auc = macro_auc_from_probs(y_true, P, classes)
    result["macro_auc"] = round(auc, 4) if not np.isnan(auc) else float("nan")

    calibration = ece_from_probs(y_true, P, classes)
    result["ece"] = calibration["ece"]
    result["calibration"] = calibration

    result["per_class"] = _per_class_metrics(y_true, P, classes, binary=binary)
    result["macro_f1"] = round(
        float(np.mean([m["f1"] for m in result["per_class"].values() if m["support"] > 0]))
        if result["per_class"] else 0.0,
        4,
    )
    result["_y_true"] = y_true
    result["_P"] = P
    return result


def _per_class_metrics(y_true: np.ndarray, P: np.ndarray, classes: list, binary: bool = False) -> dict:
    """Precision / recall / F1 / support per class, from top-1 predictions."""
    if len(y_true) == 0:
        return {}

    class_array = np.array(classes, dtype=object)
    predicted = class_array[np.argmax(P, axis=1)]
    labels = BINARY_LABELS if binary else DISPOSITION_LABELS

    per_class = {}
    for cls in classes:
        tp = int(np.sum((y_true == cls) & (predicted == cls)))
        fp = int(np.sum((y_true != cls) & (predicted == cls)))
        fn = int(np.sum((y_true == cls) & (predicted != cls)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        try:
            label = labels.get(int(cls), str(cls))
        except (TypeError, ValueError):
            label = str(cls)

        per_class[cls] = {
            "label": label,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": int(np.sum(y_true == cls)),
        }
    return per_class


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_results_table(results: list[dict], ks=(1, 3, 5)):
    """
    Print the headline comparison table: one row per model, CIs included.

    The marginal baseline belongs in this table permanently. It is the number
    that makes every other number interpretable.
    """
    if not results:
        return

    n_test = results[0].get("n_test", 0)
    task = "binary affirm/reverse" if results[0].get("binary") else \
           f"{results[0].get('n_classes', '?')}-class disposition"

    print(f"\n── Results ({task}, n = {n_test}) ─────────────────────────")
    header = f"  {'Model':<26}"
    for k in ks:
        header += f" {'Top-' + str(k):>16}"
    header += f" {'LogLoss':>9} {'Brier':>7} {'ECE':>7} {'AUC':>6}"
    print(header)
    print(f"  {'─' * (26 + 17 * len(ks) + 32)}")

    for r in results:
        line = f"  {r['name'][:26]:<26}"
        for k in ks:
            key = f"top_{k}"
            if key in r:
                lo, hi = r.get(f"{key}_ci", (float('nan'), float('nan')))
                line += f" {r[key]*100:>6.1f} [{lo*100:>4.1f},{hi*100:>4.1f}]"
            else:
                line += f" {'—':>16}"
        auc = r.get("macro_auc", float("nan"))
        auc_str = f"{auc:>6.3f}" if not np.isnan(auc) else f"{'—':>6}"
        line += f" {r['log_loss']:>9.4f} {r['brier']:>7.4f} {r['ece']:>7.4f} {auc_str}"
        print(line)

    print(f"\n  Top-k cells show the point estimate with a bootstrap 95% CI.")
    print(f"  Log-loss and Brier are proper scoring rules — lower is better.")


def print_classification_report(report: dict):
    """Pretty-print per-class precision / recall / F1."""
    per_class = report.get("per_class", report)
    n_test = report.get("n_test", "?")

    print(f"\n── Classification Report (n={n_test}) ──────────────────")
    print(f"  {'Class':<40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*70}")

    for cls in sorted(per_class.keys(), key=_sort_key):
        m = per_class[cls]
        print(f"  {m['label'][:40]:<40} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['support']:>8}")

    print(f"  {'─'*70}")
    if "macro_f1" in report:
        print(f"  {'Macro F1':<40} {'':>6} {'':>6} {report['macro_f1']:>6.3f}")
    if "top_1" in report:
        print(f"  {'Top-1 Accuracy':<40} {'':>6} {'':>6} {report['top_1']:>6.3f}")


def print_calibration_report(cal: dict):
    """Pretty-print the calibration analysis."""
    cal = cal.get("calibration", cal)

    print(f"\n── Calibration Analysis ─────────────────────────────────")
    print(f"  Expected Calibration Error (ECE): {cal['ece']:.4f}")
    print(f"  Mean Confidence:  {cal.get('mean_confidence', 0):.4f}")
    print(f"  Overall Accuracy: {cal.get('overall_accuracy', 0):.4f}")

    if cal.get("bin_counts"):
        print(f"\n  {'Bin':>10} {'Confidence':>12} {'Accuracy':>10} {'Count':>7} {'Gap':>7}")
        print(f"  {'─' * 50}")
        edges = cal["bin_edges"]
        for i, (conf, acc, cnt) in enumerate(
            zip(cal["bin_confidences"], cal["bin_accuracies"], cal["bin_counts"])
        ):
            if cnt > 0:
                print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}   {conf:>10.3f}   "
                      f"{acc:>8.3f}   {cnt:>5}   {abs(acc - conf):>5.3f}")


def distribution_summary(
    engine,
    evidence: dict,
    n_samples: int = 5000,
    query_var: str = "final_disposition",
):
    """Print the full predicted probability distribution for a single case."""
    dist = engine.query(query_var, evidence, n_samples=n_samples)
    print(f"\n  Predicted distribution for {query_var}:")
    for val, prob in sorted(dist.items(), key=lambda x: -x[1]):
        try:
            label = DISPOSITION_LABELS.get(int(val), str(val))
            code = f"{int(val):>2}"
        except (TypeError, ValueError):
            label, code = str(val), str(val)
        bar = "█" * int(prob * 40)
        print(f"    {code}: {label[:35]:<35} {prob*100:5.1f}%  {bar}")
    return dist


# ---------------------------------------------------------------------------
# Backwards-compatible single-metric helpers
# ---------------------------------------------------------------------------


def top_k_accuracy(
    df_test: pd.DataFrame,
    sampler,
    k: int = 3,
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    verbose: bool = True,
    evidence_vars=None,
) -> float:
    """Top-k accuracy on a test set. Prints per-case detail when `verbose`."""
    y_true, P, classes = predict_distributions(
        df_test, sampler, n_samples=n_samples,
        query_var=query_var, evidence_vars=evidence_vars,
    )
    accuracy = top_k_from_probs(y_true, P, classes, k)

    if verbose:
        class_array = np.array(classes, dtype=object)
        order = np.argsort(-P, axis=1)[:, :k]
        for i in range(len(y_true)):
            preds = list(class_array[order[i]])
            correct = y_true[i] in preds
            label = DISPOSITION_LABELS.get(int(y_true[i]), str(y_true[i]))
            print(f"  Case {i+1:>3}: true={int(y_true[i])} ({label[:30]})  "
                  f"top-{k}={[int(p) for p in preds]}  {'✓' if correct else '✗'}")

    n_correct = int(round(accuracy * len(y_true)))
    print(f"\n  Top-{k} Accuracy: {n_correct}/{len(y_true)} = {accuracy*100:.1f}%")
    return accuracy


def multi_k_accuracy(
    df_test: pd.DataFrame,
    sampler,
    ks=(1, 3, 5),
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    evidence_vars=None,
) -> dict:
    """Top-k accuracy for several k from a single inference pass."""
    y_true, P, classes = predict_distributions(
        df_test, sampler, n_samples=n_samples,
        query_var=query_var, evidence_vars=evidence_vars,
    )
    return {k: top_k_from_probs(y_true, P, classes, k) for k in ks}


def classification_report(
    df_test: pd.DataFrame,
    sampler,
    n_samples: int = 1000,
    query_var: str = "final_disposition",
    evidence_vars=None,
) -> dict:
    """Per-class precision, recall and F1 for top-1 predictions."""
    y_true, P, classes = predict_distributions(
        df_test, sampler, n_samples=n_samples,
        query_var=query_var, evidence_vars=evidence_vars,
    )
    per_class = _per_class_metrics(y_true, P, classes)
    f1s = [m["f1"] for m in per_class.values() if m["support"] > 0]

    return {
        "per_class": per_class,
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 3),
        "accuracy": round(top_k_from_probs(y_true, P, classes, 1), 3),
        "top_1": round(top_k_from_probs(y_true, P, classes, 1), 3),
        "n_test": int(len(y_true)),
    }


def calibration_analysis(
    df_test: pd.DataFrame,
    sampler,
    n_samples: int = 1000,
    n_bins: int = 10,
    query_var: str = "final_disposition",
    evidence_vars=None,
) -> dict:
    """Expected Calibration Error and reliability-diagram data."""
    y_true, P, classes = predict_distributions(
        df_test, sampler, n_samples=n_samples,
        query_var=query_var, evidence_vars=evidence_vars,
    )
    return ece_from_probs(y_true, P, classes, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Cross-validation and engine comparison
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
    evidence_vars=None,
    engine_cls=None,
) -> dict:
    """
    Run k-fold cross-validation.

    `engine_cls` defaults to the exact engine when available, since it is both
    faster and noise-free; pass a sampler class to cross-validate that instead.
    """
    from src.exact import VariableEliminationEngine

    if engine_cls is None:
        engine_cls = (
            VariableEliminationEngine if use_likelihood_weighting
            else RejectionSampler
        )

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
        engine = engine_cls(builder, random_state=seed + fold)

        y_true, P, classes = predict_distributions(
            df_test, engine, n_samples=n_samples, evidence_vars=evidence_vars
        )
        acc = top_k_from_probs(y_true, P, classes, k)
        fold_accuracies.append(acc)

        if verbose:
            print(f"  Fold {fold+1}/{n_folds}: Top-{k} = {acc*100:.1f}% "
                  f"(train={len(df_train)}, test={len(df_test)})")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))

    if verbose:
        print(f"\n  {n_folds}-Fold CV: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

    return {
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "mean": round(mean_acc, 4),
        "std": round(std_acc, 4),
    }


def compare_inference_methods(
    df_test: pd.DataFrame,
    builder: CPTBuilder,
    k: int = 3,
    n_samples: int = 1000,
    seed: int = 42,
    evidence_vars=None,
) -> dict:
    """
    Compare all four inference engines on the same test set.

    Exact inference is included as ground truth: the samplers' accuracy is only
    interesting relative to the answer they are approximating.
    """
    import time

    from src.exact import VariableEliminationEngine, total_variation_distance

    engines = [
        ("Exact (Var. Elimination)", VariableEliminationEngine),
        ("Rejection Sampling", RejectionSampler),
        ("Likelihood Weighting", LikelihoodWeightingSampler),
        ("Gibbs Sampling (MCMC)", GibbsSampler),
    ]

    results = {}
    exact_P = None

    for name, cls in engines:
        engine = cls(builder, random_state=seed)
        t0 = time.perf_counter()
        y_true, P, classes = predict_distributions(
            df_test, engine, n_samples=n_samples, evidence_vars=evidence_vars
        )
        elapsed = time.perf_counter() - t0
        acc = top_k_from_probs(y_true, P, classes, k)

        entry = {"accuracy": round(acc, 4), "time_seconds": round(elapsed, 2)}

        if exact_P is None:
            exact_P = P
        elif P.shape == exact_P.shape:
            tv = float(np.mean(0.5 * np.abs(P - exact_P).sum(axis=1)))
            entry["tv_from_exact"] = round(tv, 4)

        results[name] = entry
        tv_note = (f"  TV from exact: {entry['tv_from_exact']:.4f}"
                   if "tv_from_exact" in entry else "  (reference)")
        print(f"  {name:<26} Top-{k}: {acc*100:5.1f}%  ({elapsed:6.2f}s){tv_note}")

    return results


def sampler_convergence(
    builder: CPTBuilder,
    evidence: dict,
    sample_counts=(100, 500, 1000, 5000),
    query_var: str = "final_disposition",
    seed: int = 42,
) -> dict:
    """
    Measure how each sampler's estimate approaches the exact posterior.

    Returns {engine name: [TV distance at each sample count]}. This is the
    experiment that justifies having three samplers: it shows them converging
    to a known answer at different rates.
    """
    from src.exact import VariableEliminationEngine, total_variation_distance

    exact = VariableEliminationEngine(builder).query(query_var, evidence)

    curves = {}
    for name, cls in [
        ("Rejection Sampling", RejectionSampler),
        ("Likelihood Weighting", LikelihoodWeightingSampler),
        ("Gibbs Sampling (MCMC)", GibbsSampler),
    ]:
        distances = []
        for n in sample_counts:
            engine = cls(builder, random_state=seed)
            approx = engine.query(query_var, evidence, n_samples=n)
            distances.append(round(total_variation_distance(exact, approx), 4))
        curves[name] = distances

    curves["_sample_counts"] = list(sample_counts)
    curves["_exact"] = exact
    return curves
