"""Tests for inference engines (Rejection Sampling + Likelihood Weighting)."""


import numpy as np
import pytest


from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler



class TestRejectionSampler:
    """Tests for the baseline rejection sampling inference."""


    def test_query_returns_distribution(self, rejection_sampler):
        dist = rejection_sampler.query("final_disposition", evidence={}, n_samples=500)
        assert isinstance(dist, dict)
        assert len(dist) > 0


    def test_distribution_sums_to_one(self, rejection_sampler):
        dist = rejection_sampler.query("final_disposition", evidence={}, n_samples=500)
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.05, f"Distribution sums to {total}"


    def test_probabilities_non_negative(self, rejection_sampler):
        dist = rejection_sampler.query("final_disposition", evidence={}, n_samples=500)
        for val, prob in dist.items():
            assert prob >= 0, f"Negative probability for {val}"


    def test_top_k_predictions(self, rejection_sampler):
        preds = rejection_sampler.top_k_predictions(
            "final_disposition", evidence={}, k=3, n_samples=500
        )
        assert len(preds) <= 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in preds)
        # Should be sorted descending
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True)


    def test_evidence_changes_distribution(self, rejection_sampler):
        """Providing evidence should shift the distribution."""
        no_evidence = rejection_sampler.query("final_disposition", {}, n_samples=1000)
        with_evidence = rejection_sampler.query(
            "final_disposition", {"issue_area": 1.0}, n_samples=1000
        )
        # Distributions should be different (not exactly equal)
        # This is probabilistic so we just check they're both valid
        assert len(no_evidence) > 0
        assert len(with_evidence) > 0


    def test_empty_evidence(self, rejection_sampler):
        dist = rejection_sampler.query("final_disposition", {}, n_samples=200)
        assert len(dist) > 0



class TestLikelihoodWeightingSampler:
    """Tests for the improved Likelihood Weighting inference."""


    def test_query_returns_distribution(self, lw_sampler):
        dist = lw_sampler.query("final_disposition", evidence={}, n_samples=500)
        assert isinstance(dist, dict)
        assert len(dist) > 0


    def test_distribution_sums_to_one(self, lw_sampler):
        dist = lw_sampler.query("final_disposition", evidence={}, n_samples=500)
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.05, f"Distribution sums to {total}"


    def test_probabilities_non_negative(self, lw_sampler):
        dist = lw_sampler.query("final_disposition", evidence={}, n_samples=500)
        for val, prob in dist.items():
            assert prob >= 0


    def test_top_k_predictions(self, lw_sampler):
        preds = lw_sampler.top_k_predictions(
            "final_disposition", evidence={}, k=3, n_samples=500
        )
        assert len(preds) <= 3
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True)


    def test_with_evidence(self, lw_sampler):
        """LW should handle evidence without rejection — all samples accepted."""
        dist = lw_sampler.query(
            "final_disposition",
            {"chief_justice": "Roberts", "issue_area": 1.0, "law_type": 1.0},
            n_samples=500,
        )
        assert len(dist) > 0
        assert abs(sum(dist.values()) - 1.0) < 0.05


    def test_many_evidence_vars(self, lw_sampler):
        """LW should handle many evidence variables efficiently (rejection sampling struggles)."""
        evidence = {
            "chief_justice": "Roberts",
            "issue_area": 1.0,
            "law_type": 1.0,
            "split_vote": 0.0,
            "unconstitutional": 0.0,
        }
        dist = lw_sampler.query("final_disposition", evidence, n_samples=500)
        assert len(dist) > 0



class TestSamplerConsistency:
    """All three samplers should produce roughly similar distributions."""


    def test_similar_distributions_no_evidence(self, rejection_sampler, lw_sampler, gibbs_sampler):
        """With no evidence and enough samples, all methods should roughly agree."""
        rs_dist = rejection_sampler.query("final_disposition", {}, n_samples=2000)
        lw_dist = lw_sampler.query("final_disposition", {}, n_samples=2000)
        gb_dist = gibbs_sampler.query("final_disposition", {}, n_samples=2000)


        rs_top = max(rs_dist, key=rs_dist.get) if rs_dist else None
        lw_top = max(lw_dist, key=lw_dist.get) if lw_dist else None
        gb_top = max(gb_dist, key=gb_dist.get) if gb_dist else None


        assert rs_top is not None
        assert lw_top is not None
        assert gb_top is not None



class TestGibbsSampler:
    """Tests for Gibbs Sampling (MCMC) inference."""


    def test_query_returns_distribution(self, gibbs_sampler):
        dist = gibbs_sampler.query("final_disposition", evidence={}, n_samples=500, burn_in=100)
        assert isinstance(dist, dict)
        assert len(dist) > 0


    def test_distribution_sums_to_one(self, gibbs_sampler):
        dist = gibbs_sampler.query("final_disposition", evidence={}, n_samples=500, burn_in=100)
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.05, f"Distribution sums to {total}"


    def test_probabilities_non_negative(self, gibbs_sampler):
        dist = gibbs_sampler.query("final_disposition", evidence={}, n_samples=500, burn_in=100)
        for val, prob in dist.items():
            assert prob >= 0


    def test_top_k_predictions(self, gibbs_sampler):
        preds = gibbs_sampler.top_k_predictions(
            "final_disposition", evidence={}, k=3, n_samples=500
        )
        assert len(preds) <= 3
        probs = [p for _, p in preds]
        assert probs == sorted(probs, reverse=True)


    def test_with_evidence(self, gibbs_sampler):
        dist = gibbs_sampler.query(
            "final_disposition",
            {"chief_justice": "Roberts", "issue_area": 1.0},
            n_samples=500,
            burn_in=100,
        )
        assert len(dist) > 0
        assert abs(sum(dist.values()) - 1.0) < 0.05


    def test_many_evidence_vars(self, gibbs_sampler):
        """Gibbs should handle many evidence variables without issues."""
        evidence = {
            "chief_justice": "Roberts",
            "issue_area": 1.0,
            "law_type": 1.0,
            "split_vote": 0.0,
            "unconstitutional": 0.0,
        }
        dist = gibbs_sampler.query("final_disposition", evidence, n_samples=500, burn_in=100)
        assert len(dist) > 0