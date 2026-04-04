"""
inference.py
------------
Rejection sampling for probabilistic inference in the Bayesian Network.

Rejection Sampling
------------------
To compute P(query | evidence):

1. Sample a full variable assignment from the joint distribution
   by ancestral sampling (following topological order, sampling
   each variable given its parents' sampled values).

2. If the sample is consistent with the evidence, accept it.
   Otherwise, reject it.

3. Repeat for N samples. The query distribution is estimated
   from the accepted samples.

This is exact in the limit of infinite samples.
"""

import numpy as np
from collections import defaultdict
from network_structure import NODES, TOPOLOGICAL_ORDER
from cpt_builder import CPTBuilder


class RejectionSampler:
    """Perform probabilistic inference via rejection sampling."""

    def __init__(self, cpt_builder: CPTBuilder, random_state: int = 42):
        self.cpt = cpt_builder
        self.rng = np.random.default_rng(random_state)

    # -----------------------------------------------------------------------
    # Ancestral sampling
    # -----------------------------------------------------------------------

    def _sample_node(self, node_name: str, assignment: dict):
        """
        Sample a value for node_name given the current full assignment.

        Parameters
        ----------
        node_name  : str
        assignment : dict — values already sampled for parent nodes

        Returns
        -------
        sampled value
        """
        node = NODES[node_name]
        values = list(self.cpt.get_values(node_name))
        if not values:
            return None

        if node.is_root():
            probs = np.array([self.cpt.query_root(node_name, v) for v in values])
        else:
            valid_parents = [p for p in node.parents if p in assignment]
            parent_vals = tuple(assignment[p] for p in valid_parents)
            probs = np.array([
                self.cpt.query_child(node_name, parent_vals, v) for v in values
            ])

        # Normalize (handle floating-point issues)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(values)) / len(values)
        else:
            probs = probs / total

        idx = self.rng.choice(len(values), p=probs)
        return values[idx]

    def _ancestral_sample(self) -> dict:
        """
        Draw one complete assignment from the joint distribution
        by sampling variables in topological order.
        """
        assignment = {}
        for node_name in TOPOLOGICAL_ORDER:
            if node_name in self.cpt.cpts:
                assignment[node_name] = self._sample_node(node_name, assignment)
        return assignment

    # -----------------------------------------------------------------------
    # Rejection sampling
    # -----------------------------------------------------------------------

    def query(
        self,
        query_var: str,
        evidence: dict,
        n_samples: int = 5000,
    ) -> dict:
        """
        Estimate P(query_var | evidence) via rejection sampling.

        Parameters
        ----------
        query_var : str — the variable to predict (e.g., 'final_disposition')
        evidence  : dict — {variable_name: observed_value}
        n_samples : int — total samples to draw

        Returns
        -------
        dict — {value: estimated_probability}  (normalized)
        """
        counts = defaultdict(float)
        n_accepted = 0

        for _ in range(n_samples):
            sample = self._ancestral_sample()

            # Check evidence consistency
            consistent = all(
                sample.get(var) == val
                for var, val in evidence.items()
                if var in sample
            )
            if not consistent:
                continue

            # Accepted sample — record query variable value
            q_val = sample.get(query_var)
            if q_val is not None:
                counts[q_val] += 1
                n_accepted += 1

        if n_accepted == 0:
            # No accepted samples — return uniform over known values
            values = list(self.cpt.get_values(query_var))
            return {v: 1.0 / len(values) for v in values} if values else {}

        # Normalize
        total = sum(counts.values())
        return {v: c / total for v, c in sorted(counts.items(), key=lambda x: -x[1])}

    def top_k_predictions(
        self,
        query_var: str,
        evidence: dict,
        k: int = 3,
        n_samples: int = 5000,
    ) -> list[tuple]:
        """
        Return the top-k most probable values for query_var given evidence.

        Returns
        -------
        list of (value, probability) sorted by probability descending
        """
        dist = self.query(query_var, evidence, n_samples=n_samples)
        sorted_preds = sorted(dist.items(), key=lambda x: -x[1])
        return sorted_preds[:k]
