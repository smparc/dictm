"""Tests for the CLI plumbing and the report printers."""

import numpy as np
import pandas as pd
import pytest

import main as cli
from src.baselines import MarginalBaseline
from src.evaluate import (
    compare_inference_methods,
    evaluate_model,
    k_fold_cross_validation,
    print_calibration_report,
    print_classification_report,
    print_results_table,
)


class TestArgumentParsing:
    def test_defaults(self):
        args = cli.build_parser().parse_args([])
        assert args.mode == "train_eval"
        assert args.track == "both"
        assert args.task == "multiclass"
        assert args.split == "chronological"

    def test_track_and_task_are_settable(self):
        args = cli.build_parser().parse_args(["--track", "ex_ante", "--task", "binary"])
        assert args.track == "ex_ante"
        assert args.task == "binary"

    def test_rejects_unknown_track(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["--track", "nonsense"])

    def test_test_fraction_is_configurable(self):
        assert cli.build_parser().parse_args(["--test-frac", "0.3"]).test_frac == 0.3


class TestTracksFor:
    def test_both_expands_to_every_policy(self):
        args = cli.build_parser().parse_args(["--track", "both"])
        assert set(cli.tracks_for(args)) == {"explanatory", "ex_ante"}

    def test_single_track_stays_single(self):
        args = cli.build_parser().parse_args(["--track", "ex_ante"])
        assert cli.tracks_for(args) == ["ex_ante"]


class TestSplits:
    def test_chronological_split_puts_later_terms_in_test(self):
        df = pd.DataFrame({"term": list(range(100)), "x": range(100)})
        train, test = cli.chronological_split(df, test_fraction=0.2)
        assert len(test) == 20
        assert train["term"].max() < test["term"].min()

    def test_chronological_split_is_deterministic(self):
        df = pd.DataFrame({"term": list(range(50))})
        a, _ = cli.chronological_split(df, 0.2)
        b, _ = cli.chronological_split(df, 0.2)
        assert a.equals(b)

    def test_random_split_partitions_without_overlap(self):
        df = pd.DataFrame({"term": list(range(100)), "x": range(100)})
        train, test = cli.random_split(df, test_fraction=0.25, seed=1)
        assert len(train) + len(test) == 100
        assert not set(train["x"]) & set(test["x"])

    def test_random_split_is_seeded(self):
        df = pd.DataFrame({"term": list(range(60)), "x": range(60)})
        a, _ = cli.random_split(df, 0.2, seed=3)
        b, _ = cli.random_split(df, 0.2, seed=3)
        assert a.equals(b)

    def test_split_without_a_date_column_falls_back_to_row_order(self):
        df = pd.DataFrame({"x": range(20)})
        train, test = cli.chronological_split(df, 0.25)
        assert len(test) == 5


class TestDatasetDiscovery:
    def test_missing_directory_raises_with_a_download_hint(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError, match="fetch_data"):
            cli.find_dataset()

    def test_prefers_the_newest_release(self, tmp_path, monkeypatch):
        for name in ("SCDB_2023_01_caseCentered_Citation.csv",
                     "SCDB_2024_01_caseCentered_Citation.csv"):
            (tmp_path / name).write_text("x")
        monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path))
        assert "2024_01" in cli.find_dataset()

    def test_ignores_unrelated_csvs(self, tmp_path, monkeypatch):
        (tmp_path / "notes.csv").write_text("x")
        monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            cli.find_dataset()


class TestReportPrinters:
    """The printers must not raise on any shape of result they are handed."""

    @pytest.fixture
    def results(self, prepared_df, exact_engine):
        return [
            evaluate_model(prepared_df.head(60), MarginalBaseline(prepared_df),
                           name="Marginal (no features)", n_boot=20),
            evaluate_model(prepared_df.head(60), exact_engine,
                           name="Bayesian Network", n_boot=20),
        ]

    def test_results_table(self, results, capsys):
        print_results_table(results)
        out = capsys.readouterr().out
        assert "Marginal (no features)" in out
        assert "Bayesian Network" in out
        assert "LogLoss" in out

    def test_results_table_handles_binary(self, prepared_df, exact_engine, capsys):
        result = evaluate_model(prepared_df, exact_engine, name="bn",
                                binary=True, ks=(1,), n_boot=10)
        print_results_table([result], ks=(1,))
        assert "binary affirm/reverse" in capsys.readouterr().out

    def test_results_table_tolerates_empty_input(self):
        print_results_table([])

    def test_classification_report(self, results, capsys):
        print_classification_report(results[-1])
        assert "Macro F1" in capsys.readouterr().out

    def test_calibration_report(self, results, capsys):
        print_calibration_report(results[-1])
        assert "Expected Calibration Error" in capsys.readouterr().out

    def test_verdict_names_the_marginal_comparison(self, results, capsys):
        cli._print_track_verdict({"ex_ante": results}, ks=(1, 3))
        out = capsys.readouterr().out
        assert "marginal" in out
        assert "beats" in out or "does not beat" in out


class TestCrossValidationAndComparison:
    def test_k_fold_returns_mean_and_spread(self, prepared_df, capsys):
        result = k_fold_cross_validation(prepared_df, n_folds=3, k=3,
                                         n_samples=100, verbose=False)
        assert len(result["fold_accuracies"]) == 3
        assert 0.0 <= result["mean"] <= 1.0
        assert result["std"] >= 0.0

    def test_compare_inference_methods_covers_all_engines(self, prepared_df, fitted_builder, capsys):
        results = compare_inference_methods(prepared_df.head(12), fitted_builder,
                                            k=3, n_samples=120)
        assert len(results) == 4
        assert "Exact (Var. Elimination)" in results
        # Every sampler is scored against exact as the reference.
        for name, entry in results.items():
            if name.startswith("Exact"):
                assert "tv_from_exact" not in entry
            else:
                assert 0.0 <= entry["tv_from_exact"] <= 1.0
