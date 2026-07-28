"""
baselines.py
------------
Reference models the Bayesian Network must be measured against.

Why this module exists
----------------------
This project previously reported 73.3% Top-3 accuracy against "~27% expected by
random chance". Random guessing is not the baseline a reader applies. The
Supreme Court affirms, reverses, or reverses-and-remands in about 80% of cases,
so a model that ignores every feature and always names those three dispositions
scores **~79.6% Top-3** on SCDB — comfortably above the figure the network was
reporting.

`MarginalBaseline` is that model. It belongs in every results table
permanently, because it is the number that makes every other number
interpretable.

The discriminative baselines (logistic regression, gradient boosting) answer a
different question: given the same features, how much accuracy does the
Bayesian Network give up in exchange for its structure? Expect it to give up
some. The network's compensating advantages — an interpretable graph, a full
calibrated posterior, and the ability to answer queries under *partial*
evidence, which a fitted sklearn model cannot do without imputation — are real,
but they are advantages worth naming honestly rather than hiding behind a
favourable comparison.

Every class here implements the same `query` / `top_k_predictions` interface as
the inference engines, so they drop straight into `evaluate_model`.
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from src.cpt_builder import _py, _sort_key
from src.network_structure import COLUMN_MAP, FEATURE_SETS

log = logging.getLogger(__name__)


class _BaseBaseline:
    """Shared `top_k_predictions` implementation."""

    name = "baseline"

    def query(self, query_var: str, evidence: dict, n_samples: int | None = None) -> dict:
        raise NotImplementedError

    def top_k_predictions(
        self, query_var: str, evidence: dict, k: int = 3, n_samples: int | None = None
    ) -> list[tuple]:
        dist = self.query(query_var, evidence)
        return sorted(dist.items(), key=lambda x: -x[1])[:k]


class MarginalBaseline(_BaseBaseline):
    """
    Predicts the training-set class distribution, identically for every case.

    Uses no features at all. On the multi-class disposition task its Top-3
    accuracy is the combined base rate of the three most common dispositions —
    the bar any feature-based model has to clear.
    """

    name = "Marginal (no features)"

    def __init__(self, df_train: pd.DataFrame, query_var: str = "final_disposition"):
        col = COLUMN_MAP.get(query_var, query_var)
        counts = df_train[col].dropna().value_counts(normalize=True)
        self.distribution = {_py(v): float(p) for v, p in counts.items()}

    def query(self, query_var: str, evidence: dict, n_samples: int | None = None) -> dict:
        return dict(self.distribution)


class MajorityClassBaseline(_BaseBaseline):
    """
    Puts all mass on the single most common training class.

    Deliberately badly calibrated — it exists as a Top-1 reference point, and as
    an illustration of why log-loss is reported alongside accuracy: this model
    can look respectable on Top-1 while scoring terribly as a probabilistic
    forecast.
    """

    name = "Majority class"

    def __init__(self, df_train: pd.DataFrame, query_var: str = "final_disposition"):
        col = COLUMN_MAP.get(query_var, query_var)
        counts = df_train[col].dropna().value_counts()
        self.majority = _py(counts.index[0])
        self.classes = [_py(v) for v in counts.index]

    def query(self, query_var: str, evidence: dict, n_samples: int | None = None) -> dict:
        # A little mass is left on the other classes so log-loss stays finite;
        # an infinite penalty for one confident mistake would say more about the
        # metric's edge case than about the model.
        floor = 0.01 / max(1, len(self.classes) - 1)
        dist = {c: floor for c in self.classes}
        dist[self.majority] = 0.99
        return dist


class _SklearnBaseline(_BaseBaseline):
    """
    Shared plumbing for the discriminative baselines.

    Features are the evidence variables under the active track, one-hot encoded.
    Missing values become their own "absent" category rather than being imputed,
    which keeps the comparison against the Bayesian Network fair — the network
    also handles missingness natively.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        evidence_vars=None,
        query_var: str = "final_disposition",
        model=None,
        name: str = "sklearn",
    ):
        self.name = name
        self.query_var = query_var

        if isinstance(evidence_vars, str):
            evidence_vars = FEATURE_SETS[evidence_vars]
        if evidence_vars is None:
            evidence_vars = [n for n in COLUMN_MAP if n != query_var]
        self.evidence_vars = [n for n in evidence_vars if n != query_var]

        self.feature_cols = [
            COLUMN_MAP[n] for n in self.evidence_vars if COLUMN_MAP.get(n) in df_train.columns
        ]

        target_col = COLUMN_MAP.get(query_var, query_var)
        train = df_train.dropna(subset=[target_col])

        X = self._encode(train[self.feature_cols])
        self._columns = X.columns
        y = np.array([_py(v) for v in train[target_col]], dtype=object)

        self.model = model
        self.model.fit(X.to_numpy(), y)
        self.classes = [_py(c) for c in self.model.classes_]

    @staticmethod
    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(frame.astype("object").fillna("__absent__"), dummy_na=False)

    def _encode_evidence(self, evidence: dict) -> np.ndarray:
        row = {}
        for node in self.evidence_vars:
            col = COLUMN_MAP.get(node)
            if col in self.feature_cols:
                row[col] = evidence.get(node, "__absent__")
        encoded = self._encode(pd.DataFrame([row]))
        encoded = encoded.reindex(columns=self._columns, fill_value=0)
        return encoded.to_numpy()

    def query(self, query_var: str, evidence: dict, n_samples: int | None = None) -> dict:
        probs = self.model.predict_proba(self._encode_evidence(evidence))[0]
        return {c: float(p) for c, p in zip(self.classes, probs)}


class LogisticRegressionBaseline(_SklearnBaseline):
    """Multinomial logistic regression on one-hot encoded evidence variables."""

    def __init__(self, df_train, evidence_vars=None, query_var="final_disposition", seed=42):
        from sklearn.linear_model import LogisticRegression

        super().__init__(
            df_train, evidence_vars, query_var,
            model=LogisticRegression(max_iter=1000, random_state=seed),
            name="Logistic regression",
        )


class GradientBoostingBaseline(_SklearnBaseline):
    """Histogram gradient boosting — the strong discriminative reference."""

    def __init__(self, df_train, evidence_vars=None, query_var="final_disposition", seed=42):
        from sklearn.ensemble import HistGradientBoostingClassifier

        super().__init__(
            df_train, evidence_vars, query_var,
            model=HistGradientBoostingClassifier(random_state=seed),
            name="Gradient boosting",
        )


def build_baselines(
    df_train: pd.DataFrame,
    evidence_vars=None,
    query_var: str = "final_disposition",
    include_sklearn: bool = True,
    seed: int = 42,
) -> list:
    """
    Construct the standard baseline set for a results table.

    scikit-learn models are skipped with a warning if the package is missing, so
    the marginal comparison — the important one — is always available.
    """
    baselines = [
        MarginalBaseline(df_train, query_var),
        MajorityClassBaseline(df_train, query_var),
    ]

    if include_sklearn:
        try:
            baselines.append(
                LogisticRegressionBaseline(df_train, evidence_vars, query_var, seed)
            )
            baselines.append(
                GradientBoostingBaseline(df_train, evidence_vars, query_var, seed)
            )
        except ImportError:
            log.warning("scikit-learn not installed; skipping discriminative baselines")
        except Exception as exc:
            log.warning("Could not fit discriminative baselines: %s", exc)

    return baselines
