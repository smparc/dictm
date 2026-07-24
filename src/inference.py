"""
inference.py
------------
Probabilistic inference engines for the Bayesian Network.


Three algorithms are implemented:


1. **Rejection Sampling** (baseline from the paper)
   - Generates full ancestral samples, rejects inconsistent ones
   - Simple but exponentially slow with many evidence variables


2. **Likelihood Weighting** (improved)
   - Always accepts samples by fixing evidence variables and weighting
     each sample by the product of P(evidence_var | parents)
   - Much more efficient: no wasted samples, convergence in fewer iterations
   - Standard improvement over rejection sampling in BN textbooks
     (Russell & Norvig, Ch. 14)


3. **Gibbs Sampling** (MCMC)
   - Markov Chain Monte Carlo: initializes consistent state, then repeatedly
     re-samples each non-evidence variable conditioned on its Markov blanket
   - Converges to the exact posterior; best for complex networks with many
     inter-dependencies
   - Includes burn-in period to reach stationary distribution
"""


import logging
import numpy as np
from collections import defaultdict
from src.network_structure import NODES, TOPOLOGICAL_ORDER
from src.cpt_builder import CPTBuilder


log = logging.getLogger(__name__)



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



class LikelihoodWeightingSampler:
    """
    Likelihood Weighting — a more efficient alternative to rejection sampling.


    Instead of rejecting samples that don't match evidence, we:
    1. Fix evidence variables to their observed values
    2. Sample non-evidence variables normally from P(X | parents)
    3. Weight each sample by the product of P(evidence_var = observed | parents)


    This guarantees every sample is "accepted" (with a weight), so we never
    waste computation. With many evidence variables, this can be orders of
    magnitude faster than rejection sampling.
    """


    def __init__(self, cpt_builder: CPTBuilder, random_state: int = 42):
        self.cpt = cpt_builder
        self.rng = np.random.default_rng(random_state)


    def _sample_node(self, node_name: str, assignment: dict):
        """Sample a value for a non-evidence node given its parents."""
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


        total = probs.sum()
        if total <= 0:
            probs = np.ones(len(values)) / len(values)
        else:
            probs = probs / total


        idx = self.rng.choice(len(values), p=probs)
        return values[idx]


    def _weighted_sample(self, evidence: dict) -> tuple[dict, float]:
        """
        Generate one weighted sample.


        Returns
        -------
        (assignment, weight) — the full variable assignment and its likelihood weight
        """
        assignment = {}
        weight = 1.0


        for node_name in TOPOLOGICAL_ORDER:
            if node_name not in self.cpt.cpts:
                continue


            node = NODES[node_name]


            if node_name in evidence:
                # Fix to observed value; multiply weight by P(observed | parents)
                assignment[node_name] = evidence[node_name]


                if node.is_root():
                    w = self.cpt.query_root(node_name, evidence[node_name])
                else:
                    valid_parents = [p for p in node.parents if p in assignment]
                    parent_vals = tuple(assignment[p] for p in valid_parents)
                    w = self.cpt.query_child(node_name, parent_vals, evidence[node_name])


                weight *= w
            else:
                # Sample normally
                assignment[node_name] = self._sample_node(node_name, assignment)


        return assignment, weight


    def query(
        self,
        query_var: str,
        evidence: dict,
        n_samples: int = 5000,
    ) -> dict:
        """
        Estimate P(query_var | evidence) via likelihood weighting.


        Parameters
        ----------
        query_var : str
        evidence  : dict
        n_samples : int


        Returns
        -------
        dict — {value: estimated_probability}
        """
        weighted_counts = defaultdict(float)
        total_weight = 0.0


        for _ in range(n_samples):
            sample, weight = self._weighted_sample(evidence)
            q_val = sample.get(query_var)
            if q_val is not None:
                weighted_counts[q_val] += weight
                total_weight += weight


        if total_weight == 0:
            values = list(self.cpt.get_values(query_var))
            return {v: 1.0 / len(values) for v in values} if values else {}


        return {
            v: w / total_weight
            for v, w in sorted(weighted_counts.items(), key=lambda x: -x[1])
        }


    def top_k_predictions(
        self,
        query_var: str,
        evidence: dict,
        k: int = 3,
        n_samples: int = 5000,
    ) -> list[tuple]:
        """Return the top-k most probable values."""
        dist = self.query(query_var, evidence, n_samples=n_samples)
        sorted_preds = sorted(dist.items(), key=lambda x: -x[1])
        return sorted_preds[:k]



class GibbsSampler:
    """
    Gibbs Sampling — MCMC inference for Bayesian Networks.


    Algorithm:
    1. Initialize all non-evidence variables to random consistent values
    2. For each iteration, sweep through non-evidence variables in
       topological order, re-sampling each from P(X | Markov Blanket(X))
    3. After a burn-in period, collect samples for the query variable


    The Markov blanket of a node X consists of:
    - X's parents
    - X's children
    - The other parents of X's children (co-parents)


    Gibbs sampling is guaranteed to converge to the true posterior
    distribution as the number of samples → ∞.
    """


    def __init__(self, cpt_builder: CPTBuilder, random_state: int = 42):
        self.cpt = cpt_builder
        self.rng = np.random.default_rng(random_state)
        self._children_cache: dict[str, list[str]] = {}
        self._build_children_cache()


    def _build_children_cache(self):
        """Pre-compute the children of each node for Markov blanket lookups."""
        for node_name in TOPOLOGICAL_ORDER:
            self._children_cache[node_name] = []
        for node_name in TOPOLOGICAL_ORDER:
            node = NODES[node_name]
            for parent in node.parents:
                if parent in self._children_cache:
                    self._children_cache[parent].append(node_name)


    def _initialize_state(self, evidence: dict) -> dict:
        """Create an initial assignment: evidence fixed, rest sampled randomly."""
        state = {}
        for node_name in TOPOLOGICAL_ORDER:
            if node_name not in self.cpt.cpts:
                continue
            if node_name in evidence:
                state[node_name] = evidence[node_name]
            else:
                values = list(self.cpt.get_values(node_name))
                if values:
                    state[node_name] = self.rng.choice(values)
        return state


    def _compute_full_conditional(self, node_name: str, state: dict) -> dict[object, float]:
        """
        Compute P(X = x | Markov Blanket) ∝ P(X | parents) × ∏ P(child | child_parents).


        This is the key equation for Gibbs sampling: the probability of a
        variable given its Markov blanket decomposes into the product of
        its own CPT entry times the CPT entries of all its children.
        """
        node = NODES[node_name]
        values = list(self.cpt.get_values(node_name))
        if not values:
            return {}


        scores = {}
        for val in values:
            # P(X = val | parents(X))
            if node.is_root():
                p = self.cpt.query_root(node_name, val)
            else:
                valid_parents = [p for p in node.parents if p in state]
                parent_vals = tuple(state[p] for p in valid_parents)
                p = self.cpt.query_child(node_name, parent_vals, val)


            # Multiply by P(child | child_parents) for each child
            for child_name in self._children_cache.get(node_name, []):
                child_node = NODES[child_name]
                child_val = state.get(child_name)
                if child_val is None:
                    continue


                # Build parent values for the child, substituting our candidate val
                child_parent_vals = []
                for cp in child_node.parents:
                    if cp == node_name:
                        child_parent_vals.append(val)
                    elif cp in state:
                        child_parent_vals.append(state[cp])
                child_parent_vals = tuple(child_parent_vals)


                p *= self.cpt.query_child(child_name, child_parent_vals, child_val)


            scores[val] = p


        return scores


    def _gibbs_step(self, state: dict, evidence: dict):
        """One full sweep: re-sample each non-evidence variable."""
        for node_name in TOPOLOGICAL_ORDER:
            if node_name in evidence or node_name not in self.cpt.cpts:
                continue


            scores = self._compute_full_conditional(node_name, state)
            if not scores:
                continue


            values = list(scores.keys())
            probs = np.array([scores[v] for v in values], dtype=float)
            total = probs.sum()
            if total <= 0:
                probs = np.ones(len(values)) / len(values)
            else:
                probs /= total


            state[node_name] = values[self.rng.choice(len(values), p=probs)]


    def query(
        self,
        query_var: str,
        evidence: dict,
        n_samples: int = 5000,
        burn_in: int = 500,
    ) -> dict:
        """
        Estimate P(query_var | evidence) via Gibbs sampling.


        Parameters
        ----------
        query_var : str
        evidence  : dict
        n_samples : int — samples to collect after burn-in
        burn_in   : int — initial samples to discard (reaching stationary dist)


        Returns
        -------
        dict — {value: estimated_probability}
        """
        state = self._initialize_state(evidence)
        counts = defaultdict(float)


        # Burn-in phase
        for _ in range(burn_in):
            self._gibbs_step(state, evidence)


        # Collection phase
        for _ in range(n_samples):
            self._gibbs_step(state, evidence)
            q_val = state.get(query_var)
            if q_val is not None:
                counts[q_val] += 1


        total = sum(counts.values())
        if total == 0:
            values = list(self.cpt.get_values(query_var))
            return {v: 1.0 / len(values) for v in values} if values else {}


        return {
            v: c / total
            for v, c in sorted(counts.items(), key=lambda x: -x[1])
        }


    def top_k_predictions(
        self,
        query_var: str,
        evidence: dict,
        k: int = 3,
        n_samples: int = 5000,
    ) -> list[tuple]:
        """Return the top-k most probable values."""
        dist = self.query(query_var, evidence, n_samples=n_samples)
        sorted_preds = sorted(dist.items(), key=lambda x: -x[1])
        return sorted_preds[:k]