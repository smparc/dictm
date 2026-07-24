"""
preprocessing.py
-----------------
Data preprocessing pipeline for SCDB (Supreme Court Database) data.


Handles:
- Column validation: ensures required SCDB columns exist
- Missing data analysis: reports per-variable missingness rates
- Outlier detection: flags rare/unexpected category codes
- Row filtering: drops rows where critical variables are absent
- Data summary: prints a clean overview of the processed dataset
"""


import logging
from collections import Counter


import numpy as np
import pandas as pd


from src.network_structure import COLUMN_MAP, TOPOLOGICAL_ORDER, NODES


log = logging.getLogger(__name__)



def validate_columns(df: pd.DataFrame) -> list[str]:
    """
    Check that all required SCDB columns are present.


    Returns
    -------
    list of missing column names (empty if all present)
    """
    required = set(COLUMN_MAP.values())
    present = set(df.columns)
    missing = required - present
    if missing:
        log.warning("Missing required columns: %s", missing)
    return list(missing)



def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-variable missing data statistics.


    Returns
    -------
    DataFrame with columns: variable, column, n_missing, pct_missing
    """
    rows = []
    for node_name in TOPOLOGICAL_ORDER:
        col = COLUMN_MAP.get(node_name)
        if col and col in df.columns:
            n_missing = int(df[col].isna().sum())
            pct = n_missing / len(df) * 100
            rows.append({
                "variable": node_name,
                "column": col,
                "n_missing": n_missing,
                "pct_missing": round(pct, 2),
            })
    return pd.DataFrame(rows)



def detect_rare_categories(
    df: pd.DataFrame,
    min_count: int = 5,
) -> dict[str, list]:
    """
    Identify category values that appear fewer than `min_count` times.


    These rare categories may cause sparse CPTs and unreliable inference.


    Returns
    -------
    dict {node_name: [(value, count), ...]}
    """
    rare = {}
    for node_name in TOPOLOGICAL_ORDER:
        col = COLUMN_MAP.get(node_name)
        if col and col in df.columns:
            counts = df[col].value_counts()
            rare_vals = [(val, int(c)) for val, c in counts.items() if c < min_count]
            if rare_vals:
                rare[node_name] = sorted(rare_vals, key=lambda x: x[1])
    return rare



def preprocess(
    df: pd.DataFrame,
    drop_missing_target: bool = True,
    drop_missing_threshold: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.


    Steps:
    1. Validate columns
    2. Drop rows missing the target variable (final_disposition)
    3. Drop rows with too many missing features
    4. Report data quality summary


    Parameters
    ----------
    df                     : raw SCDB DataFrame
    drop_missing_target    : if True, drop rows with no target value
    drop_missing_threshold : drop rows where > this fraction of features are NaN
    verbose                : print summary


    Returns
    -------
    Cleaned DataFrame
    """
    n_original = len(df)


    # 1. Validate columns
    missing_cols = validate_columns(df)
    if missing_cols and verbose:
        print(f"  ⚠ Missing columns: {missing_cols}")


    # 2. Drop rows without target
    target_col = COLUMN_MAP["final_disposition"]
    if drop_missing_target and target_col in df.columns:
        df = df.dropna(subset=[target_col])


    # 3. Drop rows with too many missing features
    feature_cols = [COLUMN_MAP[n] for n in TOPOLOGICAL_ORDER
                    if n != "final_disposition" and COLUMN_MAP.get(n) in df.columns]
    if feature_cols:
        missing_frac = df[feature_cols].isna().mean(axis=1)
        df = df[missing_frac <= drop_missing_threshold]


    n_final = len(df)


    if verbose:
        print(f"  Preprocessing: {n_original} → {n_final} rows "
              f"({n_original - n_final} dropped)")


        # Missing data summary
        report = missing_data_report(df)
        has_missing = report[report["n_missing"] > 0]
        if not has_missing.empty:
            print(f"\n  Remaining missing values:")
            for _, r in has_missing.iterrows():
                print(f"    {r['variable']:<25} {r['n_missing']:>6} ({r['pct_missing']:.1f}%)")


        # Rare categories
        rare = detect_rare_categories(df)
        if rare:
            total_rare = sum(len(v) for v in rare.values())
            print(f"\n  Rare categories (<5 occurrences): {total_rare} values across "
                  f"{len(rare)} variables")


    return df.reset_index(drop=True)



def print_data_summary(df: pd.DataFrame):
    """Print a formatted summary of the dataset for the target variable."""
    target_col = COLUMN_MAP["final_disposition"]
    if target_col not in df.columns:
        return


    print(f"\n── Data Summary ─────────────────────────────────────────")
    print(f"  Total cases: {len(df)}")


    from src.network_structure import DISPOSITION_LABELS
    counts = df[target_col].value_counts().sort_index()
    print(f"\n  Disposition distribution:")
    for val, count in counts.items():
        label = DISPOSITION_LABELS.get(int(val), str(val))
        pct = count / len(df) * 100
        bar = "█" * int(pct)
        print(f"    {int(val):>2}: {label[:35]:<35} {count:>5} ({pct:5.1f}%)  {bar}")