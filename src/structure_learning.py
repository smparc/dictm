"""
structure_learning.py
---------------------
Automated structure learning and dependency analysis for the Bayesian Network.


Implements the dependency test from the paper:
    D(A, B) = |P(A, B) - P(A)P(B)|


Where D(A,B) represents the distance from perfect independence. Variable pairs
with D above a threshold are considered dependent and form edges in the network.


Additionally computes mutual information I(A; B) as a more theoretically
grounded measure of statistical dependence.
"""


import itertools
import logging
from collections import defaultdict


import numpy as np
import pandas as pd


from src.network_structure import COLUMN_MAP, TOPOLOGICAL_ORDER


log = logging.getLogger(__name__)



def compute_dependency_score(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> float:
    """
    Compute D(A,B) = |P(A,B) - P(A)P(B)| averaged over all value pairs.


    This is the independence test described in the paper (Equation 1).
    A value of 0 indicates perfect independence; higher values indicate
    stronger dependence.


    Parameters
    ----------
    df    : DataFrame with columns var_a and var_b
    var_a : column name for variable A
    var_b : column name for variable B


    Returns
    -------
    float — average absolute deviation from independence
    """
    sub = df[[var_a, var_b]].dropna()
    n = len(sub)
    if n == 0:
        return 0.0


    # Marginal probabilities
    p_a = sub[var_a].value_counts(normalize=True).to_dict()
    p_b = sub[var_b].value_counts(normalize=True).to_dict()


    # Joint probabilities
    joint = sub.groupby([var_a, var_b]).size() / n
    joint_dict = joint.to_dict()


    # Average |P(A,B) - P(A)P(B)| over all observed value pairs
    total_dev = 0.0
    n_pairs = 0
    for (a_val, b_val), p_ab in joint_dict.items():
        p_a_val = p_a.get(a_val, 0)
        p_b_val = p_b.get(b_val, 0)
        total_dev += abs(p_ab - p_a_val * p_b_val)
        n_pairs += 1


    return total_dev / max(1, n_pairs)



def compute_mutual_information(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
) -> float:
    """
    Compute mutual information I(A; B) = Σ P(a,b) log[P(a,b) / (P(a)P(b))].


    A more theoretically grounded measure of statistical dependence than
    the simple deviation score. Always non-negative; 0 iff A ⊥ B.


    Parameters
    ----------
    df    : DataFrame
    var_a : column name
    var_b : column name


    Returns
    -------
    float — mutual information in nats
    """
    sub = df[[var_a, var_b]].dropna()
    n = len(sub)
    if n == 0:
        return 0.0


    p_a = sub[var_a].value_counts(normalize=True).to_dict()
    p_b = sub[var_b].value_counts(normalize=True).to_dict()
    joint = sub.groupby([var_a, var_b]).size() / n


    mi = 0.0
    for (a_val, b_val), p_ab in joint.items():
        p_a_val = p_a.get(a_val, 0)
        p_b_val = p_b.get(b_val, 0)
        if p_ab > 0 and p_a_val > 0 and p_b_val > 0:
            mi += p_ab * np.log(p_ab / (p_a_val * p_b_val))


    return max(0.0, mi)



def dependency_matrix(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    method: str = "deviation",
) -> pd.DataFrame:
    """
    Compute a pairwise dependency matrix for all variables.


    Parameters
    ----------
    df        : DataFrame with SCDB data
    variables : list of column names (defaults to COLUMN_MAP values)
    method    : "deviation" for D(A,B) or "mutual_info" for I(A;B)


    Returns
    -------
    pd.DataFrame — symmetric matrix of dependency scores
    """
    if variables is None:
        variables = [col for col in COLUMN_MAP.values() if col in df.columns]


    score_fn = compute_dependency_score if method == "deviation" else compute_mutual_information
    n = len(variables)
    matrix = np.zeros((n, n))


    for i, j in itertools.combinations(range(n), 2):
        score = score_fn(df, variables[i], variables[j])
        matrix[i, j] = score
        matrix[j, i] = score


    return pd.DataFrame(matrix, index=variables, columns=variables)



def find_top_dependencies(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    threshold: float = 0.01,
    method: str = "deviation",
    top_n: int = 20,
) -> list[tuple[str, str, float]]:
    """
    Find the strongest pairwise dependencies in the dataset.


    Parameters
    ----------
    df        : DataFrame
    variables : list of column names to compare (defaults to COLUMN_MAP values)
    threshold : minimum score to include
    method    : "deviation" or "mutual_info"
    top_n     : max edges to return


    Returns
    -------
    list of (var_a, var_b, score) sorted by score descending
    """
    if variables is None:
        variables = [col for col in COLUMN_MAP.values() if col in df.columns]
    score_fn = compute_dependency_score if method == "deviation" else compute_mutual_information


    edges = []
    for va, vb in itertools.combinations(variables, 2):
        score = score_fn(df, va, vb)
        if score >= threshold:
            edges.append((va, vb, round(score, 6)))


    edges.sort(key=lambda x: -x[2])
    return edges[:top_n]



def print_dependency_report(df: pd.DataFrame, method: str = "deviation"):
    """Print a formatted dependency analysis report."""
    label = "D(A,B)" if method == "deviation" else "I(A;B)"
    print(f"\n── Dependency Analysis ({label}) ────────────────────────")


    edges = find_top_dependencies(df, threshold=0.001, method=method, top_n=20)
    if not edges:
        print("  No significant dependencies found.")
        return


    max_score = edges[0][2] if edges else 1.0
    for va, vb, score in edges:
        bar_len = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "█" * bar_len
        print(f"  {va:<25} ↔ {vb:<25} {score:.6f}  {bar}")


    print(f"\n  Total edges above threshold: {len(edges)}")