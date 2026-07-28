"""
exact.py
--------
Exact inference by variable elimination.

The three sampling engines in `inference.py` approximate the posterior. This
module computes it. Two reasons that matters:

1. **Ground truth.** With an exact answer available, the samplers can be
   *validated* rather than merely compared to each other: total-variation
   distance from the exact posterior must shrink as the sample count grows.
   That turns "we implemented three algorithms" into a convergence result.

2. **It is cheaper.** The network has ten discrete nodes, so elimination runs in
   well under a millisecond — orders of magnitude faster than drawing a thousand
   samples per case, and with no Monte Carlo noise.

The engine exposes the same `query` / `top_k_predictions` interface as the
samplers, so it drops into the existing evaluation code unchanged. `n_samples`
is accepted and ignored, purely for interface compatibility.

Algorithm
---------
Every node contributes a factor phi(X, Pa(X)) = P(X | Pa(X)). Evidence is
applied by slicing each factor down to the observed values. Non-query variables
are then eliminated one at a time — multiply together every factor mentioning
the variable, sum it out — using a greedy min-degree ordering. The surviving
factors are multiplied and normalised.
"""

import logging

import numpy as np

from src.cpt_builder import CPTBuilder, _py, _sort_key
from src.network_structure import NODES, TOPOLOGICAL_ORDER

log = logging.getLogger(__name__)


class Factor:
    """A discrete factor over a set of variables, backed by a dense array."""

    __slots__ = ("variables", "domains", "table")

    def __init__(self, variables: list[str], domains: dict[str, list], table: np.ndarray):
        self.variables = list(variables)
        self.domains = domains
        self.table = table

    def __repr__(self) -> str:
        return f"Factor({', '.join(self.variables)}; shape={self.table.shape})"

    def multiply(self, other: "Factor") -> "Factor":
        """Pointwise product, broadcasting both operands onto the union of variables."""
        variables = list(self.variables)
        variables += [v for v in other.variables if v not in self.variables]

        left = self._aligned(variables)
        right = other._aligned(variables)
        return Factor(variables, self.domains, left * right)

    def _aligned(self, variables: list[str]) -> np.ndarray:
        """Reshape this factor's table so it broadcasts against `variables`."""
        order = [self.variables.index(v) for v in variables if v in self.variables]
        table = np.transpose(self.table, order)

        shape = []
        for v in variables:
            shape.append(len(self.domains[v]) if v in self.variables else 1)
        return table.reshape(shape)

    def sum_out(self, variable: str) -> "Factor":
        """Marginalise `variable` away."""
        axis = self.variables.index(variable)
        variables = [v for v in self.variables if v != variable]
        return Factor(variables, self.domains, self.table.sum(axis=axis))

    def reduce(self, evidence: dict) -> "Factor":
        """Slice the factor down to the observed values of any evidence variables."""
        variables = list(self.variables)
        table = self.table

        for var, value in evidence.items():
            if var not in variables:
                continue
            domain = self.domains[var]
            if value not in domain:
                # Evidence value never observed in training: this factor cannot
                # support it. Leave the factor untouched and let the caller's
                # backoff handle it, rather than producing an all-zero posterior.
                log.debug("Evidence %s=%r outside observed domain; ignoring", var, value)
                continue
            axis = variables.index(var)
            table = np.take(table, domain.index(value), axis=axis)
            variables = [v for v in variables if v != var]

        return Factor(variables, self.domains, table)


class VariableEliminationEngine:
    """Exact posterior inference for the Bayesian Network."""

    def __init__(self, cpt_builder: CPTBuilder, random_state: int | None = None):
        # random_state is accepted for interface parity with the samplers; exact
        # inference is deterministic and does not use it.
        self.cpt = cpt_builder
        self.domains: dict[str, list] = {
            node: sorted(cpt_builder.get_values(node), key=_sort_key)
            for node in TOPOLOGICAL_ORDER
            if cpt_builder.get_values(node)
        }
        self._factors = self._build_factors()

    # -----------------------------------------------------------------------
    # Factor construction
    # -----------------------------------------------------------------------

    def _build_factors(self) -> list[Factor]:
        """Materialise one factor per node from its CPT."""
        factors = []

        for node_name in TOPOLOGICAL_ORDER:
            if node_name not in self.domains:
                continue

            # The builder records the parents it actually fitted on; fall back to
            # the declared structure only for models saved before it did.
            declared = self.cpt.get_parents(node_name) or NODES[node_name].parents
            parents = [p for p in declared if p in self.domains]

            variables = parents + [node_name]
            shape = tuple(len(self.domains[v]) for v in variables)
            table = np.empty(shape, dtype=float)

            child_values = self.domains[node_name]

            if not parents:
                for i, value in enumerate(child_values):
                    table[i] = self.cpt.query_root(node_name, value)
            else:
                parent_domains = [self.domains[p] for p in parents]
                for index in np.ndindex(*shape[:-1]):
                    parent_values = tuple(
                        parent_domains[d][i] for d, i in enumerate(index)
                    )
                    for j, value in enumerate(child_values):
                        table[index + (j,)] = self.cpt.query_child(
                            node_name, parent_values, value
                        )

            # Guard against a degenerate all-zero conditional.
            totals = table.sum(axis=-1, keepdims=True)
            table = np.where(totals > 0, table / np.where(totals > 0, totals, 1.0),
                             1.0 / len(child_values))

            factors.append(Factor(variables, self.domains, table))

        return factors

    # -----------------------------------------------------------------------
    # Elimination
    # -----------------------------------------------------------------------

    @staticmethod
    def _elimination_order(factors: list[Factor], keep: str) -> list[str]:
        """Greedy min-degree ordering: repeatedly eliminate the least-connected variable."""
        neighbours: dict[str, set] = {}
        for factor in factors:
            for var in factor.variables:
                neighbours.setdefault(var, set()).update(
                    v for v in factor.variables if v != var
                )

        remaining = {v for v in neighbours if v != keep}
        order = []
        while remaining:
            var = min(remaining, key=lambda v: len(neighbours[v] & remaining))
            order.append(var)
            remaining.discard(var)
        return order

    def query(self, query_var: str, evidence: dict, n_samples: int | None = None) -> dict:
        """
        Compute P(query_var | evidence) exactly.

        Parameters
        ----------
        query_var : str
        evidence  : dict {variable: observed value}
        n_samples : ignored; present so this engine is interchangeable with the
                    sampling engines.

        Returns
        -------
        dict {value: probability}, ordered by probability descending.
        """
        if query_var not in self.domains:
            return {}

        evidence = {
            var: _py(value)
            for var, value in evidence.items()
            if var in self.domains and var != query_var
        }

        factors = [f.reduce(evidence) for f in self._factors]
        factors = [f for f in factors if f.variables]

        for var in self._elimination_order(factors, keep=query_var):
            involved = [f for f in factors if var in f.variables]
            if not involved:
                continue
            factors = [f for f in factors if var not in f.variables]

            product = involved[0]
            for factor in involved[1:]:
                product = product.multiply(factor)
            reduced = product.sum_out(var)
            if reduced.variables:
                factors.append(reduced)

        result = None
        for factor in factors:
            if query_var not in factor.variables:
                continue
            result = factor if result is None else result.multiply(factor)

        domain = self.domains[query_var]
        if result is None:
            return {v: 1.0 / len(domain) for v in domain}

        # Any variable that survived elimination (disconnected components can
        # leave one behind) is marginalised away here.
        for var in [v for v in result.variables if v != query_var]:
            result = result.sum_out(var)

        table = result.table.reshape(-1)
        total = table.sum()
        if total <= 0:
            return {v: 1.0 / len(domain) for v in domain}

        probs = table / total
        return {
            value: float(prob)
            for value, prob in sorted(
                zip(domain, probs, strict=True), key=lambda pair: -pair[1]
            )
        }

    def top_k_predictions(
        self,
        query_var: str,
        evidence: dict,
        k: int = 3,
        n_samples: int | None = None,
    ) -> list[tuple]:
        """Return the top-k most probable values for query_var given evidence."""
        dist = self.query(query_var, evidence)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]


def total_variation_distance(p: dict, q: dict) -> float:
    """
    Total-variation distance between two distributions given as dicts.

    TV(p, q) = 1/2 * sum_x |p(x) - q(x)|, in [0, 1]. Used to measure how far a
    sampler's estimate sits from the exact posterior.
    """
    support = set(p) | set(q)
    return 0.5 * sum(abs(p.get(x, 0.0) - q.get(x, 0.0)) for x in support)
