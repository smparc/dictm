"""
Smoke tests for the plotting code.

Every figure is generated end-to-end into a temporary directory. These are
deliberately shallow — they assert that a file appears, not what it looks like —
because the failure mode they exist to catch is a plot function raising. That is
not hypothetical: `plot_network_dag` passed `boxstyle` to a `Rectangle`, which
does not accept it, so `--visualize` crashed on its first call.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import src.visualize as viz
from src.evaluate import evaluate_model, sampler_convergence
from src.structure_learning import dependency_matrix


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(viz, "OUTPUT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def result(prepared_df, exact_engine):
    return evaluate_model(prepared_df.head(80), exact_engine, name="bn", n_boot=20)


def test_network_dag(output_dir):
    viz.plot_network_dag()
    assert (output_dir / "network_dag.png").exists()


def test_confusion_matrix(output_dir, result):
    viz.plot_confusion_matrix(result)
    assert (output_dir / "confusion_matrix.png").exists()


def test_calibration_diagram(output_dir, result):
    viz.plot_calibration_diagram(result["calibration"])
    assert (output_dir / "calibration_diagram.png").exists()


def test_results_comparison(output_dir, prepared_df, exact_engine):
    from src.baselines import MarginalBaseline

    results = [
        evaluate_model(prepared_df.head(80), MarginalBaseline(prepared_df),
                       name="Marginal (no features)", n_boot=20),
        evaluate_model(prepared_df.head(80), exact_engine, name="Bayesian Network", n_boot=20),
    ]
    viz.plot_results_comparison(results)
    assert (output_dir / "results_comparison.png").exists()


def test_convergence(output_dir, fitted_builder):
    curves = sampler_convergence(fitted_builder, {"issue_area": 1}, sample_counts=(50, 200))
    viz.plot_convergence(curves)
    assert (output_dir / "sampler_convergence.png").exists()


def test_method_comparison(output_dir):
    results = {
        "Exact (Var. Elimination)": {"accuracy": 0.8, "time_seconds": 0.1},
        "Rejection Sampling": {"accuracy": 0.78, "time_seconds": 5.0},
        "Likelihood Weighting": {"accuracy": 0.79, "time_seconds": 3.0},
        "Gibbs Sampling (MCMC)": {"accuracy": 0.79, "time_seconds": 8.0},
    }
    viz.plot_method_comparison(results)
    assert (output_dir / "method_comparison.png").exists()


def test_dependency_heatmap(output_dir, prepared_df):
    matrix = dependency_matrix(prepared_df, method="mutual_info")
    viz.plot_dependency_heatmap(matrix, method="mutual_info")
    assert (output_dir / "dependency_heatmap_mutual_info.png").exists()


def test_prediction_distribution(output_dir, exact_engine):
    dist = exact_engine.query("final_disposition", {})
    viz.plot_prediction_distribution(dist, true_val=2)
    assert (output_dir / "prediction_distribution.png").exists()
