"""
cpt_builder.py
--------------
Build Conditional Probability Tables (CPTs) from historical Supreme Court data.

For root nodes (no parents), a Laplace-smoothed marginal:

    P(X = x) = (Count(X = x) + alpha) / (N + alpha * |X|)

For child nodes, the maximum-likelihood conditional

    P_ML(X = x | Pa = pa) = Count(X = x, Pa = pa) / Count(Pa = pa)

is unusable on its own here. `final_disposition` has five parents, which yields
thousands of parent configurations against ~9,000 cases: most configurations are
observed a handful of times, and many are never observed at all.

We therefore use **hierarchical backoff** (Jelinek-Mercer interpolation). Write
pa_k for the first k parents. Then, from the marginal upward:

    P_0(x)            = Laplace-smoothed marginal
    P_k(x | pa_k)     = lam * P_ML(x | pa_k) + (1 - lam) * P_{k-1}(x | pa_{k-1})
    lam               = Count(pa_k) / (Count(pa_k) + m)

`m` is an equivalent-sample-size constant: a configuration seen far more than
`m` times is trusted almost entirely, while a rare one is pulled toward the
estimate that conditions on fewer parents. Configurations never observed at all
fall back at query time to the deepest table that does contain them, and
ultimately to the marginal.

This matters because the alternative the module previously used — a *uniform*
distribution over the child's values — is the least informative guess available,
and it was applied silently. `backoff_rate` reports how often the fallback path
is taken so the cost is visible rather than hidden.

CPT storage format
------------------
Root node  : { value: probability }
Child node : { (parent1_val, parent2_val, ..., child_val): probability }
"""

import json
import logging

import numpy as np
import pandas as pd

from src.network_structure import COLUMN_MAP, NODES, TOPOLOGICAL_ORDER

log = logging.getLogger(__name__)


def _py(obj):
    """Convert numpy scalars to plain Python types so keys hash/serialize consistently."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.str_):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _sort_key(value):
    """Order values of mixed type deterministically (numbers first, then strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, str(value))


# ---------------------------------------------------------------------------
# CPT Construction
# ---------------------------------------------------------------------------


class CPTBuilder:
    """Learns CPTs from a pandas DataFrame of historical case data."""

    def __init__(
        self,
        alpha: float = 1.0,
        backoff: bool = True,
        equivalent_sample_size: float = 5.0,
    ):
        """
        Parameters
        ----------
        alpha : float
            Laplace smoothing pseudo-count, applied to marginals and (when
            `backoff` is False) to every conditional cell.
        backoff : bool
            Use hierarchical backoff for child nodes. When False, reproduces the
            plain per-configuration Laplace estimate.
        equivalent_sample_size : float
            The `m` in lam = n / (n + m). Larger values shrink harder toward the
            lower-order estimate. Ignored when `backoff` is False.
        """
        self.alpha = alpha
        self.backoff = backoff
        self.equivalent_sample_size = equivalent_sample_size

        self.cpts: dict = {}
        self._value_sets: dict = {}   # node -> set of observed values
        self._parents: dict = {}      # node -> ordered parents actually used
        self._marginals: dict = {}    # node -> {value: smoothed marginal}
        self._levels: dict = {}       # node -> [level-0 table, ..., level-k table]

        self._backoff_hits = 0
        self._backoff_queries = 0

    # -----------------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "CPTBuilder":
        """
        Learn all CPTs from df.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain the columns listed in COLUMN_MAP values.

        Returns
        -------
        self
        """
        rename = {v: k for k, v in COLUMN_MAP.items() if v in df.columns}
        data = df.rename(columns=rename).dropna(subset=list(rename.values()))

        for node_name in TOPOLOGICAL_ORDER:
            if node_name in data.columns:
                values = {_py(v) for v in data[node_name].dropna().unique()}
                self._value_sets[node_name] = values

        for node_name in TOPOLOGICAL_ORDER:
            if node_name not in data.columns:
                continue

            node = NODES[node_name]
            self._marginals[node_name] = self._build_marginal(data, node_name)

            valid_parents = [p for p in node.parents if p in data.columns]
            self._parents[node_name] = valid_parents

            if not valid_parents:
                self.cpts[node_name] = self._marginals[node_name]
                self._levels[node_name] = [self._marginals[node_name]]
            else:
                levels = self._build_levels(data, node_name, valid_parents)
                self._levels[node_name] = levels
                self.cpts[node_name] = levels[-1]

        return self

    def _build_marginal(self, df: pd.DataFrame, node_name: str) -> dict:
        """Laplace-smoothed unconditional distribution for a node."""
        values = self._value_sets[node_name]
        counts = df[node_name].dropna().value_counts()
        total = float(counts.sum())
        n_values = len(values)

        return {
            v: (float(counts.get(v, 0)) + self.alpha) / (total + self.alpha * n_values)
            for v in values
        }

    def _build_levels(
        self, df: pd.DataFrame, node_name: str, parent_names: list
    ) -> list[dict]:
        """
        Build the backoff hierarchy for a child node.

        Returns a list whose element k is the table conditioned on the first k
        parents. Element 0 is the marginal (keyed by bare child value); elements
        1..K are keyed by (parent_1, ..., parent_k, child_value).
        """
        child_values = sorted(self._value_sets[node_name], key=_sort_key)
        n_child = len(child_values)

        levels: list[dict] = [self._marginals[node_name]]

        for depth in range(1, len(parent_names) + 1):
            prefix = parent_names[:depth]
            sub = df[prefix + [node_name]].dropna()

            # Vectorised: one groupby, unstacked into a (configs x values) matrix.
            counts = (
                sub.groupby(prefix + [node_name], observed=True)
                .size()
                .unstack(node_name, fill_value=0)
                .reindex(columns=child_values, fill_value=0)
            )

            totals = counts.to_numpy().sum(axis=1).astype(float)

            if self.backoff:
                # lam = n / (n + m); interpolate toward the shallower table.
                lam = totals / (totals + self.equivalent_sample_size)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ml = np.divide(
                        counts.to_numpy(dtype=float),
                        totals[:, None],
                        out=np.zeros(counts.shape, dtype=float),
                        where=totals[:, None] > 0,
                    )
                lower = self._lower_level_matrix(levels[depth - 1], counts.index, child_values)
                probs = lam[:, None] * ml + (1.0 - lam)[:, None] * lower
            else:
                probs = (counts.to_numpy(dtype=float) + self.alpha) / (
                    totals[:, None] + self.alpha * n_child
                )

            levels.append(self._matrix_to_table(counts.index, child_values, probs))

        return levels

    @staticmethod
    def _lower_level_matrix(lower: dict, index, child_values) -> np.ndarray:
        """
        Look up the shallower table's row for each configuration in `index`.

        The shallower table is keyed by the configuration with its last parent
        dropped (or, at depth 1, by the bare child value).
        """
        rows = []
        for key in index:
            key = (key,) if not isinstance(key, tuple) else key
            key = tuple(_py(k) for k in key)
            shorter = key[:-1]
            if shorter:
                rows.append([lower.get(shorter + (v,), 0.0) for v in child_values])
            else:
                rows.append([lower.get(v, 0.0) for v in child_values])

        matrix = np.asarray(rows, dtype=float)
        # Guard against an all-zero row so every distribution stays proper.
        totals = matrix.sum(axis=1, keepdims=True)
        blank = totals[:, 0] <= 0
        if blank.any():
            matrix[blank] = 1.0 / len(child_values)
            totals = matrix.sum(axis=1, keepdims=True)
        return matrix / totals

    @staticmethod
    def _matrix_to_table(index, child_values, probs: np.ndarray) -> dict:
        """Flatten a (configs x values) matrix into the nested-key dict format."""
        table = {}
        for i, key in enumerate(index):
            key = (key,) if not isinstance(key, tuple) else key
            key = tuple(_py(k) for k in key)
            row = probs[i]
            for j, value in enumerate(child_values):
                table[key + (value,)] = float(row[j])
        return table

    # -----------------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------------

    def query_root(self, node_name: str, value) -> float:
        """P(node = value)."""
        marginal = self._marginals.get(node_name) or self.cpts.get(node_name, {})
        value = _py(value)
        if value in marginal:
            return marginal[value]
        n = max(1, len(self._value_sets.get(node_name, ())))
        return self.alpha / (self.alpha * n + self.alpha)

    def query_child(self, node_name: str, parent_values: tuple, child_value) -> float:
        """
        P(node = child_value | parents = parent_values).

        Falls back through progressively shorter parent prefixes and finally to
        the marginal, rather than to a uniform distribution.
        """
        self._backoff_queries += 1

        parent_values = tuple(_py(v) for v in parent_values)
        child_value = _py(child_value)

        levels = self._levels.get(node_name)
        if not levels:
            return self.query_root(node_name, child_value)

        depth = min(len(parent_values), len(levels) - 1)
        for k in range(depth, 0, -1):
            key = parent_values[:k] + (child_value,)
            hit = levels[k].get(key)
            if hit is not None:
                if k < depth:
                    self._backoff_hits += 1
                return hit

        self._backoff_hits += 1
        return self.query_root(node_name, child_value)

    def get_values(self, node_name: str) -> set:
        return self._value_sets.get(node_name, set())

    def get_parents(self, node_name: str) -> list:
        """The parents actually used for this node (those present in the data)."""
        return self._parents.get(node_name, [])

    @property
    def backoff_rate(self) -> float:
        """
        Fraction of conditional queries that could not be answered at full depth.

        A high rate means the network is asking for parent configurations the
        training data never contained — a direct measure of CPT sparsity.
        """
        if self._backoff_queries == 0:
            return 0.0
        return self._backoff_hits / self._backoff_queries

    def reset_backoff_stats(self):
        self._backoff_hits = 0
        self._backoff_queries = 0

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------
    #
    # Keys are tuples, which JSON cannot represent. Rather than stringify them
    # and parse the result back as a Python literal, each entry is stored as an
    # explicit [key_parts, probability] pair. That round-trips value types
    # faithfully and never interprets file contents as code.

    @staticmethod
    def _table_to_json(table: dict) -> list:
        entries = []
        for key, prob in table.items():
            parts = list(key) if isinstance(key, tuple) else [key]
            entries.append([[_py(p) for p in parts], float(prob)])
        return entries

    @staticmethod
    def _table_from_json(entries: list, tuple_keys: bool) -> dict:
        table = {}
        for parts, prob in entries:
            key = tuple(parts) if tuple_keys else parts[0]
            table[key] = float(prob)
        return table

    def save(self, path: str):
        """Serialize CPTs to JSON."""
        payload = {
            "format": 2,
            "alpha": self.alpha,
            "backoff": self.backoff,
            "equivalent_sample_size": self.equivalent_sample_size,
            "values": {k: [_py(x) for x in v] for k, v in self._value_sets.items()},
            "parents": self._parents,
            "marginals": {
                node: self._table_to_json(table) for node, table in self._marginals.items()
            },
            "levels": {
                node: [self._table_to_json(t) for t in tables]
                for node, tables in self._levels.items()
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        log.info("CPTs saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "CPTBuilder":
        """Load CPTs from JSON."""
        with open(path) as f:
            data = json.load(f)

        builder = cls(
            alpha=data.get("alpha", 1.0),
            backoff=data.get("backoff", True),
            equivalent_sample_size=data.get("equivalent_sample_size", 5.0),
        )
        builder._value_sets = {k: set(v) for k, v in data["values"].items()}
        builder._parents = data.get("parents", {})
        builder._marginals = {
            node: cls._table_from_json(entries, tuple_keys=False)
            for node, entries in data["marginals"].items()
        }
        builder._levels = {
            node: [
                cls._table_from_json(entries, tuple_keys=(i > 0))
                for i, entries in enumerate(tables)
            ]
            for node, tables in data["levels"].items()
        }
        builder.cpts = {node: tables[-1] for node, tables in builder._levels.items()}
        return builder
