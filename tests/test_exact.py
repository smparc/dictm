"""
Tests for exact inference, and for the samplers measured against it.

The convergence test here is the load-bearing correctness check for the whole
inference layer: it either validates all three sampling engines at once or
exposes a bug in them. Previously the samplers could only be compared to each
other, which cannot distinguish "all three are right" from "all three share a
bug".
"""

import numpy as np
import pytest

from src.cpt_builder import CPTBuilder
from src.exact import (
    Factor,
    VariableEliminationEngine,
    total_variation_distance,
)
from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler


class TestFactor:
    """Unit tests for the factor algebra elimination is built on."""

    def test_multiply_broadcasts_over_union_of_variables(self):
        domains = {"a": [0, 1], "b": [0, 1, 2]}
        fa = Factor(["a"], domains, np.array([0.25, 0.75]))
        fb = Factor(["b"], domains, np.array([0.2, 0.3, 0.5]))

        product = fa.multiply(fb)
        assert set(product.variables) == {"a", "b"}
        assert product.table.shape == (2, 3)
        assert product.table.sum() == pytest.approx(1.0)

    def test_sum_out_removes_variable(self):
        domains = {"a": [0, 1], "b": [0, 1, 2]}
        table = np.arange(6, dtype=float).reshape(2, 3)
        factor = Factor(["a", "b"], domains, table)

        reduced = factor.sum_out("b")
        assert reduced.variables == ["a"]
        assert reduced.table == pytest.approx(table.sum(axis=1))

    def test_reduce_slices_on_evidence(self):
        domains = {"a": [0, 1], "b": [0, 1, 2]}
        table = np.arange(6, dtype=float).reshape(2, 3)
        factor = Factor(["a", "b"], domains, table)

        reduced = factor.reduce({"a": 1})
        assert reduced.variables == ["b"]
        assert reduced.table == pytest.approx(table[1])

    def test_reduce_ignores_unobserved_evidence_value(self):
        """An evidence value outside the observed domain must not zero the factor."""
        domains = {"a": [0, 1]}
        factor = Factor(["a"], domains, np.array([0.4, 0.6]))
        assert factor.reduce({"a": 99}).table == pytest.approx([0.4, 0.6])


class TestVariableElimination:
    """Tests for the exact engine itself."""

    def test_returns_normalized_distribution(self, exact_engine):
        dist = exact_engine.query("final_disposition", {})
        assert dist
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(p >= 0 for p in dist.values())

    def test_is_deterministic(self, exact_engine):
        evidence = {"issue_area": 1, "law_type": 1}
        assert exact_engine.query("final_disposition", evidence) == \
               exact_engine.query("final_disposition", evidence)

    def test_evidence_changes_posterior(self, exact_engine):
        """A direct parent of the outcome must visibly move the posterior."""
        a = exact_engine.query("final_disposition", {"decision_type": 1})
        b = exact_engine.query("final_disposition", {"decision_type": 6})
        assert total_variation_distance(a, b) > 0.05

    def test_lower_court_disposition_is_structurally_bottlenecked(self, exact_engine):
        """
        Documents a real weakness of the hand-crafted DAG rather than asserting
        it away.

        `lower_court_disposition` is plausibly the most informative pre-decision
        variable there is — whether the court below affirmed or reversed says a
        great deal about what the Supreme Court will do. But in this structure
        its only path to `final_disposition` runs through `precedent_alteration`,
        a near-constant binary. Conditioning on it therefore barely moves the
        posterior, and this test pins that behaviour so that anyone who later
        adds a direct edge sees this expectation fail and knows why.
        """
        a = exact_engine.query("final_disposition", {"lower_court_disposition": 1})
        b = exact_engine.query("final_disposition", {"lower_court_disposition": 3})
        assert total_variation_distance(a, b) < 0.02

    def test_root_query_matches_marginal_cpt(self, fitted_builder, exact_engine):
        """With no evidence, a root node's posterior is exactly its CPT."""
        dist = exact_engine.query("issue_area", {})
        for value in fitted_builder.get_values("issue_area"):
            assert dist[value] == pytest.approx(
                fitted_builder.query_root("issue_area", value), abs=1e-9
            )

    def test_observed_query_variable_is_ignored(self, exact_engine):
        """Passing the query variable as evidence must not collapse the posterior."""
        dist = exact_engine.query("final_disposition", {"final_disposition": 2})
        assert sum(dist.values()) == pytest.approx(1.0)
        assert len(dist) > 1

    def test_top_k_predictions_sorted(self, exact_engine):
        preds = exact_engine.top_k_predictions("final_disposition", {}, k=3)
        assert len(preds) == 3
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True)

    def test_leaf_with_all_parents_observed_equals_cpt_lookup(self, fitted_builder, exact_engine):
        """
        When every parent of the leaf is observed, the posterior is a single CPT
        row. This is the degenerate case the sampling engines were previously
        spending 1000 samples to approximate.
        """
        parents = fitted_builder.get_parents("final_disposition")
        evidence = {p: sorted(fitted_builder.get_values(p), key=str)[0] for p in parents}

        dist = exact_engine.query("final_disposition", evidence)
        parent_values = tuple(evidence[p] for p in parents)
        expected = {
            v: fitted_builder.query_child("final_disposition", parent_values, v)
            for v in fitted_builder.get_values("final_disposition")
        }
        total = sum(expected.values())
        for value, prob in expected.items():
            assert dist[value] == pytest.approx(prob / total, abs=1e-9)


class TestTotalVariationDistance:
    def test_identical_distributions_are_zero(self):
        p = {1: 0.5, 2: 0.5}
        assert total_variation_distance(p, p) == pytest.approx(0.0)

    def test_disjoint_distributions_are_one(self):
        assert total_variation_distance({1: 1.0}, {2: 1.0}) == pytest.approx(1.0)

    def test_handles_missing_support(self):
        d = total_variation_distance({1: 0.5, 2: 0.5}, {1: 1.0})
        assert d == pytest.approx(0.5)


class TestSamplerConvergence:
    """Every sampler must approach the exact posterior as sample count grows."""

    EVIDENCE = {"issue_area": 1, "law_type": 1}

    @pytest.mark.parametrize(
        "sampler_cls",
        [RejectionSampler, LikelihoodWeightingSampler, GibbsSampler],
    )
    def test_converges_toward_exact(self, fitted_builder, exact_engine, sampler_cls):
        exact = exact_engine.query("final_disposition", self.EVIDENCE)

        few = sampler_cls(fitted_builder, random_state=0).query(
            "final_disposition", self.EVIDENCE, n_samples=50
        )
        many = sampler_cls(fitted_builder, random_state=0).query(
            "final_disposition", self.EVIDENCE, n_samples=20000
        )

        near = total_variation_distance(exact, many)
        far = total_variation_distance(exact, few)

        # The large-sample estimate must be close in absolute terms, and no
        # worse than the small-sample one by more than sampling slack.
        assert near < 0.05, f"{sampler_cls.__name__} TV={near:.4f} at 20k samples"
        assert near <= far + 0.02

    @pytest.mark.parametrize(
        "sampler_cls",
        [RejectionSampler, LikelihoodWeightingSampler, GibbsSampler],
    )
    def test_agrees_with_exact_without_evidence(self, fitted_builder, exact_engine, sampler_cls):
        exact = exact_engine.query("final_disposition", {})
        approx = sampler_cls(fitted_builder, random_state=7).query(
            "final_disposition", {}, n_samples=20000
        )
        assert total_variation_distance(exact, approx) < 0.05


class TestSamplerConvergenceHelper:
    def test_returns_curve_per_engine(self, fitted_builder):
        from src.evaluate import sampler_convergence

        curves = sampler_convergence(
            fitted_builder, {"issue_area": 1}, sample_counts=(100, 2000)
        )
        engines = [k for k in curves if not k.startswith("_")]
        assert len(engines) == 3
        for name in engines:
            assert len(curves[name]) == 2
            assert all(0.0 <= d <= 1.0 for d in curves[name])
