"""
cpt_builder.py
--------------
Build Conditional Probability Tables (CPTs) from historical Supreme Court data.

For root nodes (no parents):
    P(X = x) = Count(X = x) / Total cases

For child nodes with parents:
    P(X = x | Parents(X) = pa) = Count(X = x, Parents(X) = pa)
                                  / Count(Parents(X) = pa)

CPT storage format
------------------
Root node  : { value: probability }
Child node : { (parent1_val, parent2_val, ..., child_val): probability }

Laplace smoothing is applied to avoid zero probabilities.
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from network_structure import NODES, TOPOLOGICAL_ORDER, COLUMN_MAP


# ---------------------------------------------------------------------------
# CPT Construction
# ---------------------------------------------------------------------------

class CPTBuilder:
    """Learns CPTs from a pandas DataFrame of historical case data."""

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing parameter (pseudo-count added to every cell).
        """
        self.alpha = alpha
        self.cpts: dict = {}
        self._value_sets: dict = {}  # {node_name: set of observed values}

    def fit(self, df: pd.DataFrame) -> "CPTBuilder":
        """
        Learn all CPTs from df.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns listed in COLUMN_MAP values.

        Returns
        -------
        self
        """
        # Rename columns to node names
        rename = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
        data = df.rename(columns=rename).dropna(
            subset=list(rename.values())
        )

        # Build value sets
        for node_name in TOPOLOGICAL_ORDER:
            if node_name in data.columns:
                self._value_sets[node_name] = set(data[node_name].dropna().unique())

        # Build each CPT
        for node_name in TOPOLOGICAL_ORDER:
            if node_name not in data.columns:
                continue
            node = NODES[node_name]
            if node.is_root():
                self.cpts[node_name] = self._build_root_cpt(data, node_name)
            else:
                valid_parents = [p for p in node.parents if p in data.columns]
                self.cpts[node_name] = self._build_child_cpt(data, node_name, valid_parents)

        return self

    def _build_root_cpt(self, df: pd.DataFrame, node_name: str) -> dict:
        counts = defaultdict(float)
        total = 0.0
        values = self._value_sets[node_name]

        for v in df[node_name].dropna():
            counts[v] += 1
            total += 1

        # Laplace smoothing
        n_values = len(values)
        cpt = {}
        for v in values:
            cpt[v] = (counts[v] + self.alpha) / (total + self.alpha * n_values)
        return cpt

    def _build_child_cpt(
        self, df: pd.DataFrame, node_name: str, parent_names: list
    ) -> dict:
        if not parent_names:
            return self._build_root_cpt(df, node_name)

        child_values = self._value_sets[node_name]
        n_child = len(child_values)

        # Count (parent_config, child_value) occurrences
        joint_counts = defaultdict(lambda: defaultdict(float))
        parent_counts = defaultdict(float)

        cols = parent_names + [node_name]
        sub = df[cols].dropna()

        for _, row in sub.iterrows():
            parent_key = tuple(row[p] for p in parent_names)
            child_val = row[node_name]
            joint_counts[parent_key][child_val] += 1
            parent_counts[parent_key] += 1

        # Build CPT with Laplace smoothing
        cpt = {}
        for parent_key, child_counts in joint_counts.items():
            denom = parent_counts[parent_key] + self.alpha * n_child
            for child_val in child_values:
                key = parent_key + (child_val,)
                cpt[key] = (child_counts.get(child_val, 0) + self.alpha) / denom

        return cpt

    def query_root(self, node_name: str, value) -> float:
        """P(node = value)"""
        cpt = self.cpts.get(node_name, {})
        return cpt.get(value, self.alpha / (len(cpt) * self.alpha + self.alpha))

    def query_child(self, node_name: str, parent_values: tuple, child_value) -> float:
        """P(node = child_value | parents = parent_values)"""
        cpt = self.cpts.get(node_name, {})
        key = parent_values + (child_value,)
        if key in cpt:
            return cpt[key]
        # Fallback: uniform over known child values
        n = len(self._value_sets.get(node_name, {None}))
        return self.alpha / (self.alpha * n + self.alpha)

    def get_values(self, node_name: str) -> set:
        return self._value_sets.get(node_name, set())

    def save(self, path: str):
        """Serialize CPTs to JSON."""
        # Convert tuple keys to strings
        serializable = {}
        for node, cpt in self.cpts.items():
            serializable[node] = {str(k): v for k, v in cpt.items()}
        with open(path, "w") as f:
            json.dump({"cpts": serializable, "values": {k: list(v) for k, v in self._value_sets.items()}}, f, indent=2)
        print(f"CPTs saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CPTBuilder":
        """Load CPTs from JSON."""
        with open(path) as f:
            data = json.load(f)
        builder = cls()
        builder._value_sets = {k: set(v) for k, v in data["values"].items()}
        builder.cpts = {}
        for node, cpt_raw in data["cpts"].items():
            cpt = {}
            for k_str, v in cpt_raw.items():
                # Attempt to parse tuple keys
                try:
                    k_eval = eval(k_str)  # safe: only numbers/strings
                    cpt[k_eval] = v
                except Exception:
                    cpt[k_str] = v
            builder.cpts[node] = cpt
        return builder
