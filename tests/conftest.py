"""Shared fixtures for the dictm test suite."""

import numpy as np
import pandas as pd
import pytest

from src.cpt_builder import CPTBuilder
from src.exact import VariableEliminationEngine
from src.inference import GibbsSampler, LikelihoodWeightingSampler, RejectionSampler
from src.preprocessing import add_derived_columns


@pytest.fixture
def sample_df():
    """
    Small synthetic dataset mimicking SCDB structure.

    `caseDisposition` is generated as a function of `lcDisposition` and
    `decisionType` rather than independently, so the network has real signal to
    recover — a dataset of pure noise would let a broken model pass.
    """
    rng = np.random.default_rng(42)
    n = 400

    issue_area = rng.choice([1, 2, 3, 8, 10], size=n)
    law_type = rng.choice([1, 2, 5, 9], size=n)
    lc_disposition = rng.choice([1, 2, 3, 4], size=n)
    decision_type = rng.choice([1, 2, 6, 7], size=n)
    min_votes = rng.choice([0, 1, 2, 3], size=n, p=[0.4, 0.2, 0.2, 0.2])

    disposition = np.empty(n, dtype=int)
    for i in range(n):
        if lc_disposition[i] in (1, 2):
            weights = [0.05, 0.45, 0.25, 0.15, 0.10]
        else:
            weights = [0.05, 0.15, 0.20, 0.40, 0.20]
        if decision_type[i] == 6:
            weights = [0.20, 0.30, 0.20, 0.20, 0.10]
        disposition[i] = rng.choice([1, 2, 3, 4, 5], p=weights)

    return pd.DataFrame({
        "chief": rng.choice(["Roberts", "Rehnquist", "Burger"], size=n, p=[0.5, 0.3, 0.2]),
        "issueArea": issue_area,
        "lawType": law_type,
        "caseDispositionUnusual": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        "lcDisposition": lc_disposition,
        "decisionType": decision_type,
        "precedentAlteration": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "minVotes": min_votes,
        "majVotes": 9 - min_votes,
        "declarationUncon": rng.choice([1, 2, 3], size=n, p=[0.7, 0.2, 0.1]),
        "caseDisposition": disposition,
    })


@pytest.fixture
def prepared_df(sample_df):
    """`sample_df` with the derived columns the network expects (voteSplit)."""
    return add_derived_columns(sample_df)


@pytest.fixture
def fitted_builder(prepared_df):
    """CPTBuilder fitted on the sample data."""
    return CPTBuilder(alpha=1.0).fit(prepared_df)


@pytest.fixture
def rejection_sampler(fitted_builder):
    return RejectionSampler(fitted_builder, random_state=42)


@pytest.fixture
def lw_sampler(fitted_builder):
    return LikelihoodWeightingSampler(fitted_builder, random_state=42)


@pytest.fixture
def gibbs_sampler(fitted_builder):
    return GibbsSampler(fitted_builder, random_state=42)


@pytest.fixture
def exact_engine(fitted_builder):
    return VariableEliminationEngine(fitted_builder)
