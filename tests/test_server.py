"""
Tests for the web surface.

These exercise the pure functions (`build_schema`, `predict`) directly rather
than going through HTTP, so the suite does not require FastAPI to be installed.
"""

import pytest

from app.server import build_schema, predict
from src.network_structure import FEATURE_SETS


class TestBuildSchema:
    def test_describes_every_explanatory_variable(self, exact_engine):
        schema = build_schema(exact_engine)
        assert {f["name"] for f in schema["fields"]} == set(FEATURE_SETS["explanatory"])

    def test_marks_which_fields_are_ex_ante(self, exact_engine):
        schema = build_schema(exact_engine)
        flagged = {f["name"] for f in schema["fields"] if f["ex_ante"]}
        assert flagged == set(FEATURE_SETS["ex_ante"])

    def test_options_cover_the_observed_values(self, exact_engine, fitted_builder):
        schema = build_schema(exact_engine)
        field = next(f for f in schema["fields"] if f["name"] == "issue_area")
        assert {o["value"] for o in field["options"]} == fitted_builder.get_values("issue_area")

    def test_options_carry_human_readable_labels(self, exact_engine):
        schema = build_schema(exact_engine)
        field = next(f for f in schema["fields"] if f["name"] == "issue_area")
        labels = {o["label"] for o in field["options"]}
        assert "Criminal Procedure" in labels


class TestPredict:
    def test_returns_a_normalized_ranked_distribution(self, exact_engine):
        result = predict(exact_engine, {"issue_area": 1}, track="ex_ante")
        probs = [d["probability"] for d in result["distribution"]]
        assert sum(probs) == pytest.approx(1.0, abs=1e-4)
        assert probs == sorted(probs, reverse=True)

    def test_drops_evidence_outside_the_selected_track(self, exact_engine):
        """A post-decision variable submitted under ex_ante must be ignored, not used."""
        result = predict(exact_engine, {"issue_area": 1, "decision_type": 1}, track="ex_ante")
        assert "decision_type" not in result["evidence_used"]
        assert "decision_type" in result["ignored"]

    def test_explanatory_track_accepts_post_decision_variables(self, exact_engine):
        result = predict(exact_engine, {"decision_type": 1}, track="explanatory")
        assert "decision_type" in result["evidence_used"]
        assert result["ignored"] == []

    def test_binary_summary_sums_to_one(self, exact_engine):
        binary = predict(exact_engine, {"issue_area": 1}, track="ex_ante")["binary"]
        assert binary["affirmed"] + binary["reversed"] == pytest.approx(1.0, abs=1e-4)

    def test_empty_evidence_returns_the_prior(self, exact_engine):
        result = predict(exact_engine, {}, track="ex_ante")
        assert result["evidence_used"] == {}
        assert len(result["distribution"]) > 1

    def test_none_values_are_skipped(self, exact_engine):
        result = predict(exact_engine, {"issue_area": None, "law_type": 1}, track="ex_ante")
        assert "issue_area" not in result["evidence_used"]
        assert "law_type" in result["evidence_used"]

    def test_dispositions_are_labelled(self, exact_engine):
        result = predict(exact_engine, {}, track="ex_ante")
        assert all(d["label"] for d in result["distribution"])
