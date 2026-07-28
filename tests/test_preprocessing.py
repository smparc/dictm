"""Tests for the SCDB preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.network_structure import COLUMN_MAP
from src.preprocessing import (
    add_derived_columns,
    detect_rare_categories,
    missing_data_report,
    preprocess,
    validate_columns,
)


class TestAddDerivedColumns:
    def test_derives_vote_split_from_min_votes(self):
        df = pd.DataFrame({"minVotes": [0, 1, 3, 0], "majVotes": [9, 8, 6, 9]})
        assert add_derived_columns(df)["voteSplit"].tolist() == [0, 1, 1, 0]

    def test_does_not_mutate_the_input(self):
        df = pd.DataFrame({"minVotes": [0, 1]})
        add_derived_columns(df)
        assert "voteSplit" not in df.columns

    def test_existing_column_is_left_alone(self):
        df = pd.DataFrame({"minVotes": [0, 1], "voteSplit": [9, 9]})
        assert add_derived_columns(df)["voteSplit"].tolist() == [9, 9]

    def test_falls_back_to_raw_split_vote_and_warns_when_constant(self, caplog):
        """
        SCDB's own `splitVote` is constant, which is exactly why the derived
        column exists. If we ever have to fall back to it, that must be loud.
        """
        df = pd.DataFrame({"splitVote": [1, 1, 1, 1]})
        with caplog.at_level("WARNING"):
            result = add_derived_columns(df)
        assert result["voteSplit"].nunique() == 1
        assert any("no information" in r.message for r in caplog.records)

    def test_missing_min_votes_treated_as_unanimous(self):
        df = pd.DataFrame({"minVotes": [np.nan, 2.0]})
        assert add_derived_columns(df)["voteSplit"].tolist() == [0, 1]


class TestValidateColumns:
    def test_reports_missing_columns(self):
        assert set(validate_columns(pd.DataFrame({"chief": ["Roberts"]}))) == \
               set(COLUMN_MAP.values()) - {"chief"}

    def test_complete_frame_reports_nothing(self, prepared_df):
        assert validate_columns(prepared_df) == []


class TestMissingDataReport:
    def test_counts_missing_values_per_variable(self):
        df = pd.DataFrame({
            "caseDisposition": [1.0, np.nan, 3.0],
            "issueArea": [1.0, 2.0, 3.0],
        })
        report = missing_data_report(df).set_index("variable")
        assert report.loc["final_disposition", "n_missing"] == 1
        assert report.loc["issue_area", "n_missing"] == 0

    def test_percentage_is_computed(self):
        df = pd.DataFrame({"caseDisposition": [1.0, np.nan]})
        assert missing_data_report(df).iloc[0]["pct_missing"] == pytest.approx(50.0)


class TestDetectRareCategories:
    def test_flags_values_below_the_threshold(self):
        df = pd.DataFrame({"issueArea": [1] * 20 + [7]})
        rare = detect_rare_categories(df, min_count=5)
        assert "issue_area" in rare
        assert (7, 1) in rare["issue_area"]

    def test_common_values_are_not_flagged(self):
        df = pd.DataFrame({"issueArea": [1] * 20})
        assert "issue_area" not in detect_rare_categories(df, min_count=5)


class TestPreprocess:
    def test_drops_rows_without_a_target(self, prepared_df):
        df = prepared_df.copy()
        df.loc[:4, COLUMN_MAP["final_disposition"]] = np.nan
        assert len(preprocess(df, verbose=False)) == len(prepared_df) - 5

    def test_adds_derived_columns(self, sample_df):
        assert "voteSplit" in preprocess(sample_df, verbose=False).columns

    def test_index_is_reset(self, prepared_df):
        df = prepared_df.copy()
        df.loc[:2, COLUMN_MAP["final_disposition"]] = np.nan
        result = preprocess(df, verbose=False)
        assert result.index.tolist() == list(range(len(result)))

    def test_drops_rows_with_too_many_missing_features(self, prepared_df):
        df = prepared_df.copy()
        feature_cols = [COLUMN_MAP[n] for n in COLUMN_MAP if n != "final_disposition"]
        df.loc[0, feature_cols] = np.nan
        assert len(preprocess(df, verbose=False)) == len(prepared_df) - 1

    def test_is_idempotent(self, prepared_df):
        once = preprocess(prepared_df, verbose=False)
        assert len(preprocess(once, verbose=False)) == len(once)
