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

import sys
import os
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network_structure import COLUMN_MAP, DISPOSITION_LABELS, TOPOLOGICAL_ORDER
from cpt_builder import CPTBuilder
from inference import RejectionSampler
from evaluate import top_k_accuracy, distribution_summary, build_evidence

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
    return df


def train_test_split(df: pd.DataFrame, test_size: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return df.iloc[train_idx], df.iloc[test_idx]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_train_eval(args):
    print("\n── Loading Data ─────────────────────────────────────────")
    df = load_data(DATA_PATH)

    print("\n── Splitting Data ───────────────────────────────────────")
    df_train, df_test = train_test_split(df, TEST_SIZE, RANDOM_SEED)
    print(f"  Train: {len(df_train)} cases  |  Test: {len(df_test)} cases")

    print("\n── Building CPTs ────────────────────────────────────────")
    builder = CPTBuilder(alpha=1.0)
    builder.fit(df_train)
    print(f"  CPTs built for {len(builder.cpts)} nodes")

    # Show value counts for key nodes
    for node in ["issue_area", "final_disposition"]:
        vals = builder.get_values(node)
        print(f"  {node}: {len(vals)} unique values")

    # Save model
    builder.save(MODEL_PATH)

    print("\n── Evaluating (Rejection Sampling) ──────────────────────")
    sampler = RejectionSampler(builder, random_state=RANDOM_SEED)
    acc = top_k_accuracy(
        df_test, sampler,
        k=TOP_K,
        n_samples=N_SAMPLES,
        verbose=True,
    )
    return acc


def mode_predict(args):
    if not os.path.exists(MODEL_PATH):
        print(f"No saved model found at {MODEL_PATH}. Run --mode train_eval first.")
        return

    print("\n── Loading Model ────────────────────────────────────────")
    builder = CPTBuilder.load(MODEL_PATH)
    sampler = RejectionSampler(builder, random_state=RANDOM_SEED)

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
                # Try to parse as number
                try:
                    evidence[node_name] = float(raw)
                except ValueError:
                    evidence[node_name] = raw
        except (EOFError, KeyboardInterrupt):
            break

    print(f"\n  Evidence provided: {evidence}")
    print(f"\n── Running Inference ({N_SAMPLES} samples) ─────────────────")
    distribution_summary(sampler, evidence, n_samples=N_SAMPLES)


def mode_eval(args):
    if not os.path.exists(MODEL_PATH):
        print(f"No saved model at {MODEL_PATH}. Run --mode train_eval first.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"No data at {DATA_PATH}.")
        return

    df = load_data(DATA_PATH)
    _, df_test = train_test_split(df, TEST_SIZE, RANDOM_SEED)

    builder = CPTBuilder.load(MODEL_PATH)
    sampler = RejectionSampler(builder, random_state=RANDOM_SEED)
    top_k_accuracy(df_test, sampler, k=TOP_K, n_samples=N_SAMPLES, verbose=True)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Supreme Court Decision Predictor")
    parser.add_argument("--mode", choices=["train_eval", "predict", "eval"],
                        default="train_eval")
    parser.add_argument("--model_path", default=MODEL_PATH)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Supreme Court Decision Prediction — Bayesian Net  ║")
    print("╚══════════════════════════════════════════════════════╝")

    if args.mode == "train_eval":
        mode_train_eval(args)
    elif args.mode == "predict":
        mode_predict(args)
    elif args.mode == "eval":
        mode_eval(args)


if __name__ == "__main__":
    main()
