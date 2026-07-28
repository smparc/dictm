"""
Tests for the scoring metrics.

These are checked against hand-computed values rather than against each other,
because a metric that is merely self-consistent can still be wrong — and every
conclusion in this project rests on them.
"""

import numpy as np
import pytest

from src.evaluate import (
    bootstrap_ci,
    brier_from_probs,
    build_evidence,
    ece_from_probs,
    evaluate_model,
    log_loss_from_probs,
    macro_auc_from_probs,
    predict_distributions,
    top_k_from_probs,
)
from src.network_structure import COLUMN_MAP, FEATURE_SETS


CLASSES = [1, 2, 3]


class TestTopK:
    def test_perfect_prediction(self):
        y = np.array([1, 2, 3])
        P = np.eye(3)
        assert top_k_from_probs(y, P, CLASSES, 1) == 1.0

    def test_always_wrong_at_k1_but_right_at_k2(self):
        y = np.array([2, 2])
        P = np.array([[0.5, 0.4, 0.1], [0.5, 0.4, 0.1]])
        assert top_k_from_probs(y, P, CLASSES, 1) == 0.0
        assert top_k_from_probs(y, P, CLASSES, 2) == 1.0

    def test_k_is_clamped_to_class_count(self):
        y = np.array([1])
        P = np.array([[0.4, 0.35, 0.25]])
        assert top_k_from_probs(y, P, CLASSES, 99) == 1.0

    def test_monotonic_in_k(self):
        rng = np.random.default_rng(0)
        y = rng.choice(CLASSES, size=50)
        P = rng.dirichlet(np.ones(3), size=50)
        scores = [top_k_from_probs(y, P, CLASSES, k) for k in (1, 2, 3)]
        assert scores == sorted(scores)

    def test_empty_input(self):
        assert top_k_from_probs(np.array([]), np.zeros((0, 3)), CLASSES, 1) == 0.0


class TestLogLoss:
    def test_confident_and_correct_is_near_zero(self):
        y = np.array([1])
        P = np.array([[1.0, 0.0, 0.0]])
        assert log_loss_from_probs(y, P, CLASSES) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_equals_log_of_class_count(self):
        y = np.array([1, 2, 3])
        P = np.full((3, 3), 1 / 3)
        assert log_loss_from_probs(y, P, CLASSES) == pytest.approx(np.log(3))

    def test_confident_and_wrong_is_large_but_finite(self):
        y = np.array([3])
        P = np.array([[1.0, 0.0, 0.0]])
        value = log_loss_from_probs(y, P, CLASSES)
        assert value > 20 and np.isfinite(value)

    def test_penalizes_overconfidence_relative_to_hedging(self):
        y = np.array([1, 2])
        confident = np.array([[0.99, 0.005, 0.005], [0.99, 0.005, 0.005]])
        hedged = np.array([[0.5, 0.4, 0.1], [0.5, 0.4, 0.1]])
        assert log_loss_from_probs(y, confident, CLASSES) > log_loss_from_probs(y, hedged, CLASSES)


class TestBrier:
    def test_perfect_prediction_is_zero(self):
        assert brier_from_probs(np.array([1]), np.array([[1.0, 0.0, 0.0]]), CLASSES) == \
               pytest.approx(0.0)

    def test_hand_computed_value(self):
        # (0.6-1)^2 + (0.3-0)^2 + (0.1-0)^2 = 0.16 + 0.09 + 0.01 = 0.26
        y = np.array([1])
        P = np.array([[0.6, 0.3, 0.1]])
        assert brier_from_probs(y, P, CLASSES) == pytest.approx(0.26)

    def test_worst_case_is_two(self):
        assert brier_from_probs(np.array([3]), np.array([[1.0, 0.0, 0.0]]), CLASSES) == \
               pytest.approx(2.0)


class TestMacroAUC:
    def test_perfect_separation_is_one(self):
        y = np.array([1, 1, 2, 2])
        P = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1],
                      [0.1, 0.8, 0.1], [0.05, 0.9, 0.05]])
        assert macro_auc_from_probs(y, P, CLASSES) == pytest.approx(1.0)

    def test_single_class_is_undefined(self):
        y = np.array([1, 1])
        P = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
        assert np.isnan(macro_auc_from_probs(y, P, CLASSES))


class TestECE:
    def test_perfectly_calibrated_is_near_zero(self):
        """Confidence 1.0 and always correct: no calibration gap."""
        y = np.array([1, 2, 3])
        P = np.eye(3)
        assert ece_from_probs(y, P, CLASSES)["ece"] == pytest.approx(0.0, abs=1e-6)

    def test_confident_and_always_wrong_is_near_one(self):
        y = np.array([2, 2])
        P = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert ece_from_probs(y, P, CLASSES)["ece"] == pytest.approx(1.0, abs=1e-6)

    def test_bin_count_matches_request(self):
        rng = np.random.default_rng(1)
        y = rng.choice(CLASSES, size=40)
        P = rng.dirichlet(np.ones(3), size=40)
        result = ece_from_probs(y, P, CLASSES, n_bins=5)
        assert len(result["bin_counts"]) == 5
        assert sum(result["bin_counts"]) == 40


class TestBootstrapCI:
    def test_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        y = rng.choice(CLASSES, size=200)
        P = rng.dirichlet(np.ones(3), size=200)

        point = top_k_from_probs(y, P, CLASSES, 1)
        lo, hi = bootstrap_ci(lambda a, b, c: top_k_from_probs(a, b, c, 1),
                              y, P, CLASSES, n_boot=300)
        assert lo <= point <= hi

    def test_deterministic_given_a_seed(self):
        rng = np.random.default_rng(4)
        y = rng.choice(CLASSES, size=50)
        P = rng.dirichlet(np.ones(3), size=50)
        fn = lambda a, b, c: top_k_from_probs(a, b, c, 1)
        assert bootstrap_ci(fn, y, P, CLASSES, n_boot=100, seed=7) == \
               bootstrap_ci(fn, y, P, CLASSES, n_boot=100, seed=7)

    def test_narrows_as_sample_size_grows(self):
        rng = np.random.default_rng(5)
        fn = lambda a, b, c: top_k_from_probs(a, b, c, 1)

        widths = []
        for n in (40, 800):
            y = rng.choice(CLASSES, size=n)
            P = rng.dirichlet(np.ones(3), size=n)
            lo, hi = bootstrap_ci(fn, y, P, CLASSES, n_boot=300)
            widths.append(hi - lo)
        assert widths[1] < widths[0]

    def test_empty_input_is_nan(self):
        lo, hi = bootstrap_ci(lambda a, b, c: top_k_from_probs(a, b, c, 1),
                              np.array([]), np.zeros((0, 3)), CLASSES, n_boot=10)
        assert np.isnan(lo) and np.isnan(hi)


class TestBuildEvidence:
    def test_excludes_the_query_variable(self, prepared_df):
        evidence = build_evidence(prepared_df.iloc[0])
        assert "final_disposition" not in evidence
        assert evidence

    def test_values_come_from_the_row(self, prepared_df):
        row = prepared_df.iloc[0]
        for node, value in build_evidence(row).items():
            assert value == row[COLUMN_MAP[node]]

    def test_track_name_restricts_the_evidence_set(self, prepared_df):
        row = prepared_df.iloc[0]
        ex_ante = build_evidence(row, evidence_vars="ex_ante")
        assert set(ex_ante) <= set(FEATURE_SETS["ex_ante"])

    def test_ex_ante_excludes_post_decision_variables(self, prepared_df):
        """The whole point of the track: none of these may leak in."""
        evidence = build_evidence(prepared_df.iloc[0], evidence_vars="ex_ante")
        for leaked in ("decision_type", "split_vote", "unconstitutional",
                       "precedent_alteration", "case_supplement"):
            assert leaked not in evidence

    def test_explanatory_is_a_superset_of_ex_ante(self, prepared_df):
        row = prepared_df.iloc[0]
        assert set(build_evidence(row, evidence_vars="ex_ante")) <= \
               set(build_evidence(row, evidence_vars="explanatory"))

    def test_explicit_variable_list_is_honoured(self, prepared_df):
        evidence = build_evidence(prepared_df.iloc[0], evidence_vars=["issue_area"])
        assert set(evidence) == {"issue_area"}


class TestEvaluateModel:
    def test_reports_every_metric_with_intervals(self, prepared_df, exact_engine):
        result = evaluate_model(prepared_df.head(60), exact_engine, name="bn", n_boot=50)
        for key in ("top_1", "top_3", "log_loss", "brier", "ece", "macro_f1", "n_test"):
            assert key in result
        for key in ("top_1_ci", "top_3_ci", "log_loss_ci"):
            lo, hi = result[key]
            assert lo <= hi

    def test_binary_task_collapses_to_two_classes(self, prepared_df, exact_engine):
        result = evaluate_model(prepared_df, exact_engine, name="bn", binary=True,
                                ks=(1,), n_boot=20)
        assert result["n_classes"] == 2
        assert set(result["classes"]) <= {0, 1}
        assert result["binary"] is True

    def test_binary_drops_cases_outside_both_classes(self, prepared_df, exact_engine):
        """Disposition 1 belongs to neither affirm nor reverse and must be excluded."""
        multi = evaluate_model(prepared_df, exact_engine, name="bn", n_boot=10)
        binary = evaluate_model(prepared_df, exact_engine, name="bn", binary=True,
                                ks=(1,), n_boot=10)
        assert binary["n_test"] < multi["n_test"]

    def test_posterior_rows_are_normalized(self, prepared_df, exact_engine):
        _, P, _ = predict_distributions(prepared_df.head(30), exact_engine)
        assert P.sum(axis=1) == pytest.approx(np.ones(P.shape[0]))
