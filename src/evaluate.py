"""
evaluate.py
-----------
Evaluate the Bayesian Network using Top-k accuracy.

Top-k Accuracy
--------------
For each test case:
1. Run rejection sampling to get a probability distribution over dispositions.
2. Check whether the true disposition appears among the top-k predictions.
3. Average over all test cases.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from network_structure import COLUMN_MAP, TOPOLOGICAL_ORDER, DISPOSITION_LABELS
from cpt_builder import CPTBuilder
from inference import RejectionSampler


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
    sampler: RejectionSampler,
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
