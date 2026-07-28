"""
Tests for the reference models.

The marginal baseline is the load-bearing one: it is the number that decides
whether the Bayesian Network has learned anything, so its correctness matters as
much as the network's.
"""

import numpy as np
import pytest

from src.baselines import (
    MarginalBaseline,
    MajorityClassBaseline,
    LogisticRegressionBaseline,
    GradientBoostingBaseline,
    build_baselines,
)
from src.evaluate import evaluate_model, predict_distributions, top_k_from_probs
from src.network_structure import COLUMN_MAP


class TestMarginalBaseline:
    def test_reproduces_training_class_frequencies(self, prepared_df):
        baseline = MarginalBaseline(prepared_df)
        expected = prepared_df[COLUMN_MAP["final_disposition"]].value_counts(normalize=True)

        dist = baseline.query("final_disposition", {})
        assert sum(dist.values()) == pytest.approx(1.0)
        for value, prob in expected.items():
            assert dist[value] == pytest.approx(prob)

    def test_ignores_evidence_entirely(self, prepared_df):
        """It uses no features, so every case must get the same distribution."""
        baseline = MarginalBaseline(prepared_df)
        assert baseline.query("final_disposition", {}) == \
               baseline.query("final_disposition", {"issue_area": 1, "law_type": 5})

    def test_top_k_is_the_k_most_common_classes(self, prepared_df):
        baseline = MarginalBaseline(prepared_df)
        counts = prepared_df[COLUMN_MAP["final_disposition"]].value_counts()

        preds = [v for v, _ in baseline.top_k_predictions("final_disposition", {}, k=3)]
        assert preds == list(counts.index[:3])

    def test_top_k_accuracy_equals_combined_base_rate(self, prepared_df):
        """
        The property that makes this baseline strong: its Top-k accuracy is the
        summed base rate of the k most common classes.
        """
        baseline = MarginalBaseline(prepared_df)
        counts = prepared_df[COLUMN_MAP["final_disposition"]].value_counts(normalize=True)
        expected = counts.iloc[:3].sum()

        y_true, P, classes = predict_distributions(prepared_df, baseline)
        assert top_k_from_probs(y_true, P, classes, 3) == pytest.approx(expected, abs=0.01)


class TestMajorityClassBaseline:
    def test_predicts_the_modal_class(self, prepared_df):
        baseline = MajorityClassBaseline(prepared_df)
        expected = prepared_df[COLUMN_MAP["final_disposition"]].value_counts().index[0]
        assert baseline.top_k_predictions("final_disposition", {}, k=1)[0][0] == expected

    def test_distribution_is_normalized_and_finite(self, prepared_df):
        """Mass is left on the other classes so log-loss cannot become infinite."""
        dist = MajorityClassBaseline(prepared_df).query("final_disposition", {})
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(p > 0 for p in dist.values())

    def test_scores_worse_than_marginal_on_log_loss(self, prepared_df):
        """
        Confident and wrong. This is the case that justifies reporting a proper
        scoring rule next to accuracy.
        """
        majority = evaluate_model(prepared_df, MajorityClassBaseline(prepared_df),
                                  name="majority", n_boot=20)
        marginal = evaluate_model(prepared_df, MarginalBaseline(prepared_df),
                                  name="marginal", n_boot=20)
        assert majority["log_loss"] > marginal["log_loss"]
        assert majority["top_1"] == pytest.approx(marginal["top_1"], abs=1e-9)


class TestSklearnBaselines:
    @pytest.mark.parametrize("cls", [LogisticRegressionBaseline, GradientBoostingBaseline])
    def test_fits_and_returns_normalized_distribution(self, prepared_df, cls):
        model = cls(prepared_df, evidence_vars="ex_ante")
        dist = model.query("final_disposition", {"issue_area": 1, "law_type": 1})
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in dist.values())

    def test_class_labels_keep_their_original_type(self, prepared_df):
        """
        Labels are fitted as strings internally; the public surface must hand
        back the original values or every downstream lookup silently misses.
        """
        model = LogisticRegressionBaseline(prepared_df, evidence_vars="ex_ante")
        observed = set(prepared_df[COLUMN_MAP["final_disposition"]].unique())
        assert set(model.classes) <= observed

    def test_handles_evidence_with_missing_fields(self, prepared_df):
        """Absent variables become an explicit category rather than an error."""
        model = LogisticRegressionBaseline(prepared_df, evidence_vars="ex_ante")
        assert sum(model.query("final_disposition", {}).values()) == pytest.approx(1.0, abs=1e-6)


class TestBuildBaselines:
    def test_returns_the_standard_set(self, prepared_df):
        baselines = build_baselines(prepared_df, evidence_vars="ex_ante")
        names = [b.name for b in baselines]
        assert any(n.startswith("Marginal") for n in names)
        assert any(n.startswith("Majority") for n in names)
        assert len(baselines) == 4

    def test_can_skip_discriminative_models(self, prepared_df):
        assert len(build_baselines(prepared_df, include_sklearn=False)) == 2

    def test_every_baseline_satisfies_the_engine_interface(self, prepared_df):
        for baseline in build_baselines(prepared_df, evidence_vars="ex_ante"):
            dist = baseline.query("final_disposition", {"issue_area": 1})
            assert isinstance(dist, dict) and dist
            preds = baseline.top_k_predictions("final_disposition", {"issue_area": 1}, k=2)
            assert len(preds) <= 2
            assert [p for _, p in preds] == sorted([p for _, p in preds], reverse=True)
