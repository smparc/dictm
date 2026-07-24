"""
main.py
-------
Entry point for the Supreme Court Decision Prediction Model.


Usage
-----
    # Train on SCDB data and evaluate:
    python main.py --mode train_eval


    # Predict a single case interactively:
    python main.py --mode predict


    # Just evaluate a saved model:
    python main.py --mode eval --model_path data/cpts.json
"""


import os
import argparse
import logging
import pandas as pd
import numpy as np


from src.network_structure import COLUMN_MAP, DISPOSITION_LABELS, TOPOLOGICAL_ORDER
from src.cpt_builder import CPTBuilder
from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler
from src.evaluate import (
    top_k_accuracy, distribution_summary, build_evidence,
    multi_k_accuracy, classification_report, print_classification_report,
    k_fold_cross_validation, compare_inference_methods,
    calibration_analysis, print_calibration_report,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "SCDB_2023_01_caseCentered_Citation.csv")
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "data", "cpts.json")
TEST_SIZE   = 30     # cases to evaluate
N_SAMPLES   = 1000   # rejection samples per case
TOP_K       = 3      # top-k accuracy threshold
RANDOM_SEED = 42



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SCDB dataset not found at '{path}'.\n\n"
            "Download from: http://scdb.wustl.edu/data.php\n"
            "Select: 'Case Centered Data | Citation' and place the CSV at:\n"
            "  data/SCDB_2023_01_caseCentered_Citation.csv"
        )
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Loaded SCDB: {len(df)} cases, {len(df.columns)} columns")


    from src.preprocessing import preprocess
    df = preprocess(df, verbose=True)
    return df



def train_test_split(df: pd.DataFrame, test_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return df.iloc[train_idx], df.iloc[test_idx]



def chronological_split(df: pd.DataFrame, test_size: int, date_col: str = "dateDecision"):
    """
    Split data chronologically: train on earlier cases, test on the most recent.


    This is the methodologically correct approach for temporal data — we never
    train on future cases to predict past ones.
    """
    if date_col not in df.columns:
        # Fallback: use row order (SCDB is already roughly chronological)
        print(f"  Warning: '{date_col}' column not found, using row order as proxy")
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]
    else:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        train = df_sorted.iloc[:-test_size]
        test = df_sorted.iloc[-test_size:]


    return train, test



# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def mode_train_eval(args):
    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(DATA_PATH)


    print("\n── Splitting Data ───────────────────────────────────────")
    if args.split == "chronological":
        df_train, df_test = chronological_split(df, TEST_SIZE)
        print(f"  Chronological split: Train on earlier cases, test on most recent")
    else:
        df_train, df_test = train_test_split(df, TEST_SIZE, RANDOM_SEED)
        print(f"  Random split (seed={RANDOM_SEED})")
    print(f"  Train: {len(df_train)} cases  |  Test: {len(df_test)} cases")


    print("\n── Building CPTs ────────────────────────────────────────")
    builder = CPTBuilder(alpha=1.0)
    builder.fit(df_train)
    print(f"  CPTs built for {len(builder.cpts)} nodes")


    for node in ["issue_area", "final_disposition"]:
        vals = builder.get_values(node)
        print(f"  {node}: {len(vals)} unique values")


    builder.save(args.model_path)


    print("\n── Comparing Inference Methods ──────────────────────────")
    compare_inference_methods(df_test, builder, k=TOP_K, n_samples=N_SAMPLES)


    print("\n── Evaluating (Likelihood Weighting) ────────────────────")
    sampler = LikelihoodWeightingSampler(builder, random_state=RANDOM_SEED)


    # Multi-k accuracy
    mk = multi_k_accuracy(df_test, sampler, ks=[1, 3, 5], n_samples=N_SAMPLES)
    for k, acc in mk.items():
        print(f"  Top-{k} Accuracy: {acc*100:.1f}%")


    # Detailed per-case results
    print()
    acc = top_k_accuracy(df_test, sampler, k=TOP_K, n_samples=N_SAMPLES, verbose=True)


    # Per-class precision/recall/F1
    report = classification_report(df_test, sampler, n_samples=N_SAMPLES)
    print_classification_report(report)


    # Calibration analysis
    cal = calibration_analysis(df_test, sampler, n_samples=N_SAMPLES)
    print_calibration_report(cal)


    # Generate visualizations
    if args.visualize:
        from src.visualize import (
            plot_network_dag, plot_confusion_matrix, plot_calibration_diagram,
            plot_method_comparison,
        )
        print("\n── Generating Visualizations ────────────────────────────")
        plot_network_dag()
        plot_confusion_matrix(df_test, sampler, n_samples=N_SAMPLES)
        plot_calibration_diagram(cal)


    return acc



def mode_predict(args):
    if not os.path.exists(args.model_path):
        print(f"No saved model found at {args.model_path}. Run --mode train_eval first.")
        return


    print("\n── Loading Model ────────────────────────────────────────")
    builder = CPTBuilder.load(args.model_path)
    sampler = LikelihoodWeightingSampler(builder, random_state=RANDOM_SEED)


    print("\nEnter case details (press Enter to skip a field):\n")
    evidence = {}


    node_prompts = {
        "chief_justice":           "Chief Justice (e.g., 'Roberts')",
        "issue_area":              "Issue Area code (1=Criminal, 2=Civil Rights, 8=Due Process ...)",
        "law_type":                "Law Type code (1=Constitution, 5=Statutory ...)",
        "case_supplement":         "Case Supplement code (leave blank if unsure)",
        "lower_court_disposition": "Lower Court Disposition code",
        "decision_type":           "Decision Type code (1=majority opinion ...)",
        "split_vote":              "Split Vote? (0=No, 1=Yes)",
        "unconstitutional":        "Unconstitutionality declared? (0=No, 1=Yes)",
        "precedent_alteration":    "Precedent altered? (0=No, 1=Yes)",
    }


    for node_name, prompt in node_prompts.items():
        try:
            raw = input(f"  {prompt}: ").strip()
            if raw:
                try:
                    evidence[node_name] = float(raw)
                except ValueError:
                    evidence[node_name] = raw
        except (EOFError, KeyboardInterrupt):
            break


    print(f"\n  Evidence provided: {evidence}")
    print(f"\n── Running Inference ({N_SAMPLES} samples, Likelihood Weighting) ──")
    distribution_summary(sampler, evidence, n_samples=N_SAMPLES)



def mode_eval(args):
    if not os.path.exists(args.model_path):
        print(f"No saved model at {args.model_path}. Run --mode train_eval first.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.")
        return


    df = load_data(DATA_PATH)
    if args.split == "chronological":
        _, df_test = chronological_split(df, TEST_SIZE)
    else:
        _, df_test = train_test_split(df, TEST_SIZE, RANDOM_SEED)


    builder = CPTBuilder.load(args.model_path)
    sampler = LikelihoodWeightingSampler(builder, random_state=RANDOM_SEED)
    top_k_accuracy(df_test, sampler, k=TOP_K, n_samples=N_SAMPLES, verbose=True)


    report = classification_report(df_test, sampler, n_samples=N_SAMPLES)
    print_classification_report(report)



def mode_cross_validate(args):
    """Run k-fold cross-validation for a robust accuracy estimate."""
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.")
        return


    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(DATA_PATH)


    print("\n── K-Fold Cross-Validation ──────────────────────────────")
    result = k_fold_cross_validation(
        df,
        n_folds=5,
        k=TOP_K,
        n_samples=N_SAMPLES,
        use_likelihood_weighting=True,
        verbose=True,
    )
    return result



def mode_dependency(args):
    """Run dependency analysis to validate the network structure."""
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.")
        return


    from src.structure_learning import print_dependency_report


    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(DATA_PATH)


    print_dependency_report(df, method="deviation")
    print()
    print_dependency_report(df, method="mutual_info")



# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Supreme Court Decision Predictor")
    parser.add_argument(
        "--mode",
        choices=["train_eval", "predict", "eval", "cross_validate", "dependency"],
        default="train_eval",
        help="train_eval: train + evaluate | predict: interactive | "
             "eval: evaluate saved model | cross_validate: k-fold CV | "
             "dependency: analyze variable dependencies",
    )
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument(
        "--split",
        choices=["random", "chronological"],
        default="chronological",
        help="Data split strategy: random or chronological (default: chronological)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate publication-quality plots in figures/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output",
    )
    args = parser.parse_args()


    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Supreme Court Decision Prediction — Bayesian Net  ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.mode == "train_eval":
        mode_train_eval(args)
    elif args.mode == "predict":
        mode_predict(args)
    elif args.mode == "eval":
        mode_eval(args)
    elif args.mode == "cross_validate":
        mode_cross_validate(args)
    elif args.mode == "dependency":
        mode_dependency(args)


if __name__ == "__main__":
    main()