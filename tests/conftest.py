"""Shared fixtures for the dictm test suite."""


import pytest
import pandas as pd
import numpy as np


from src.network_structure import COLUMN_MAP, TOPOLOGICAL_ORDER
from src.cpt_builder import CPTBuilder
from src.inference import RejectionSampler, LikelihoodWeightingSampler, GibbsSampler



@pytest.fixture
def sample_df():
    """Small synthetic dataset mimicking SCDB structure."""
    rng = np.random.default_rng(42)
    n = 200


    data = {
        "chief": rng.choice(["Roberts", "Rehnquist", "Burger"], size=n, p=[0.5, 0.3, 0.2]),
        "issueArea": rng.choice([1, 2, 3, 8, 10], size=n),
        "lawType": rng.choice([1, 2, 5, 9], size=n),
        "caseDispositionUnusual": rng.choice([0, 1, 2], size=n),
        "lcDisposition": rng.choice([1, 2, 3, 4], size=n),
        "decisionType": rng.choice([1, 2, 6, 7], size=n),
        "precedentAlteration": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "splitVote": rng.choice([0, 1], size=n, p=[0.6, 0.4]),
        "declarationUncon": rng.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "caseDisposition": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.3, 0.25, 0.25, 0.15]),
    }
    return pd.DataFrame(data)



@pytest.fixture
def fitted_builder(sample_df):
    """CPTBuilder fitted on the sample data."""
    builder = CPTBuilder(alpha=1.0)
    builder.fit(sample_df)
    return builder



@pytest.fixture
def rejection_sampler(fitted_builder):
    return RejectionSampler(fitted_builder, random_state=42)



@pytest.fixture
def lw_sampler(fitted_builder):
    return LikelihoodWeightingSampler(fitted_builder, random_state=42)



@pytest.fixture
def gibbs_sampler(fitted_builder):
    return GibbsSampler(fitted_builder, random_state=42)