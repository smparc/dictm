"""Tests for evaluation utilities."""


import numpy as np
import pandas as pd
import pytest


from src.evaluate import (
    top_k_accuracy,
    build_evidence,
    distribution_summary,
    multi_k_accuracy,
    classification_report,
    calibration_analysis,
)
from src.network_structure import COLUMN_MAP



class TestTopKAccuracy:
    """Tests for the top-k accuracy metric using actual sampler."""


    def test_returns_float(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        acc = top_k_accuracy(df_test, lw_sampler, k=3, n_samples=200, verbose=False)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


    def test_top3_geq_top1(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        acc1 = top_k_accuracy(df_test, lw_sampler, k=1, n_samples=200, verbose=False)
        acc3 = top_k_accuracy(df_test, lw_sampler, k=3, n_samples=200, verbose=False)
        assert acc1 <= acc3 + 0.01  # allow float tolerance



class TestBuildEvidence:
    """Tests for evidence extraction from dataframe rows."""


    def test_build_evidence_excludes_target(self, sample_df):
        row = sample_df.iloc[0]
        evidence = build_evidence(row, exclude_var="final_disposition")
        assert "final_disposition" not in evidence
        assert len(evidence) > 0


    def test_build_evidence_values_are_from_row(self, sample_df):
        row = sample_df.iloc[0]
        evidence = build_evidence(row, exclude_var="final_disposition")
        for node, value in evidence.items():
            col = COLUMN_MAP[node]
            assert value == row[col]



class TestDistributionSummary:
    """Tests for distribution_summary."""


    def test_runs_without_error(self, lw_sampler):
        """distribution_summary prints; just verify it doesn't crash."""
        distribution_summary(lw_sampler, evidence={}, n_samples=100)



class TestMultiKAccuracy:
    """Tests for multi_k_accuracy."""


    def test_returns_dict(self, sample_df, lw_sampler):
        df_test = sample_df.head(3)
        result = multi_k_accuracy(df_test, lw_sampler, ks=[1, 3], n_samples=200)
        assert isinstance(result, dict)
        assert 1 in result
        assert 3 in result


    def test_k1_leq_k3(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        result = multi_k_accuracy(df_test, lw_sampler, ks=[1, 3], n_samples=200)
        assert result[1] <= result[3] + 0.01



class TestClassificationReport:
    """Tests for per-class classification report."""


    def test_returns_dict(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        report = classification_report(df_test, lw_sampler, n_samples=200)
        assert isinstance(report, dict)
        assert "accuracy" in report
        assert "per_class" in report
        assert "macro_f1" in report


    def test_accuracy_in_range(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        report = classification_report(df_test, lw_sampler, n_samples=200)
        assert 0.0 <= report["accuracy"] <= 1.0



class TestCalibrationAnalysis:
    """Tests for calibration (ECE) analysis."""


    def test_returns_ece(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        cal = calibration_analysis(df_test, lw_sampler, n_samples=200)
        assert "ece" in cal
        assert cal["ece"] >= 0


    def test_returns_bins(self, sample_df, lw_sampler):
        df_test = sample_df.head(5)
        cal = calibration_analysis(df_test, lw_sampler, n_samples=200, n_bins=5)
        assert len(cal["bin_counts"]) == 5
        assert len(cal["bin_accuracies"]) == 5