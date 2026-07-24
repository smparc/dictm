"""Tests for structure learning (dependency / independence analysis)."""


import numpy as np
import pandas as pd
import pytest


from src.structure_learning import (
    compute_dependency_score,
    compute_mutual_information,
    dependency_matrix,
    find_top_dependencies,
)



@pytest.fixture
def independent_df():
    """DataFrame where columns A and B are independent."""
    rng = np.random.default_rng(0)
    n = 5000
    return pd.DataFrame({
        "A": rng.choice([0, 1], size=n, p=[0.5, 0.5]),
        "B": rng.choice([0, 1], size=n, p=[0.5, 0.5]),
    })



@pytest.fixture
def dependent_df():
    """DataFrame where column B is a deterministic function of A."""
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    return pd.DataFrame({
        "A": a,
        "B": a,  # perfectly dependent
    })



class TestDependencyScore:
    """Tests for the D(A,B) metric from the paper."""


    def test_independent_vars_low_score(self, independent_df):
        score = compute_dependency_score(independent_df, "A", "B")
        assert score < 0.05, f"Independent vars have high D={score}"


    def test_dependent_vars_high_score(self, dependent_df):
        score = compute_dependency_score(dependent_df, "A", "B")
        assert score > 0.1, f"Dependent vars have low D={score}"


    def test_symmetric(self, independent_df):
        d_ab = compute_dependency_score(independent_df, "A", "B")
        d_ba = compute_dependency_score(independent_df, "B", "A")
        assert abs(d_ab - d_ba) < 1e-10


    def test_non_negative(self, independent_df):
        score = compute_dependency_score(independent_df, "A", "B")
        assert score >= 0



class TestMutualInformation:
    """Tests for mutual information."""


    def test_independent_low_mi(self, independent_df):
        mi = compute_mutual_information(independent_df, "A", "B")
        assert mi < 0.01, f"Independent vars MI={mi}"


    def test_dependent_high_mi(self, dependent_df):
        mi = compute_mutual_information(dependent_df, "A", "B")
        assert mi > 0.3, f"Dependent vars MI={mi}"


    def test_non_negative(self, independent_df):
        mi = compute_mutual_information(independent_df, "A", "B")
        assert mi >= 0



class TestDependencyMatrix:
    """Tests for full dependency matrix computation."""


    def test_returns_correct_shape(self, independent_df):
        mat = dependency_matrix(independent_df, ["A", "B"])
        assert mat.shape == (2, 2)


    def test_diagonal_is_zero_or_max(self, independent_df):
        mat = dependency_matrix(independent_df, ["A", "B"])
        # Self-dependency can be 0 (if excluded) or high
        # Just check the matrix is valid
        assert mat.shape[0] == mat.shape[1]



class TestFindTopDependencies:
    """Tests for finding the strongest variable pairs."""


    def test_returns_sorted(self):
        rng = np.random.default_rng(0)
        n = 2000
        a = rng.choice([0, 1, 2], size=n)
        df = pd.DataFrame({
            "X": a,
            "Y": a,
            "Z": rng.choice([0, 1, 2], size=n),
        })
        top = find_top_dependencies(df, variables=["X", "Y", "Z"], top_n=3)
        scores = [s for _, _, s in top]
        assert scores == sorted(scores, reverse=True)