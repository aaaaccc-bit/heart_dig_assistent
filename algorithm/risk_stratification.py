"""Model-based risk stratification for the Cleveland dataset.

The prediction pages answer "how good is this model"; this module answers a
different question: *how many patients land in each risk band, and how did the
model actually split them?*

Probabilities come from a two-model ensemble (logistic regression + random
forest) evaluated with out-of-fold predictions, so a patient's band never
depends on a model that has already seen that patient. The band cut points are
fixed probability thresholds rather than quantiles: a "low risk" patient is one
whose out-of-fold probability is below 25%, which is a statement a clinician
can check, and the resulting band sizes are reported as they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from algorithm.dataset import Dataset
from algorithm.schema import AGE_BANDS, COLUMN_BY_NAME, TARGET_COLUMN


class RiskStratificationError(ValueError):
    """Raised when a risk stratification cannot be configured or fitted."""


ALGORITHM_VERSION = "1.0.0-risk-stratification"

CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

#: All dataset columns except the outcome, in schema order.
PREDICTORS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)

#: Fixed probability cut points; a band is ``[lower, upper)``.
THRESHOLDS = (0.25, 0.50, 0.75)

#: ``(key, label, lower, upper)`` for the four displayed bands.
STRATA = (
    ("low", "低风险", 0.0, 0.25),
    ("medium", "中风险", 0.25, 0.50),
    ("high", "高风险", 0.50, 0.75),
    ("critical", "极高风险", 0.75, 1.0),
)

#: Axes of the high-risk profile radar. Every axis is a share of patients, so
#: the six points are on the same 0-100% scale and no re-scaling is implied.
_PROFILE_SPECS = (
    ("sex", "男性占比", lambda frame: frame["sex"] == 1),
    ("age", "年龄 ≥55 占比", lambda frame: frame["age"] >= 55),
    ("cp", "无症状胸痛占比", lambda frame: frame["cp"] == 4),
    ("exang", "运动诱发心绞痛占比", lambda frame: frame["exang"] == 1),
    ("ca", "主要血管数 >0 占比", lambda frame: frame["ca"] > 0),
    ("oldpeak", "ST 压低 >0 占比", lambda frame: frame["oldpeak"] > 0),
)

#: How many of the highest-risk patients the result carries for the table.
CASE_LIMIT = 8


@dataclass(frozen=True)
class RiskStratum:
    key: str
    label: str
    lower: float
    upper: float
    count: int
    percent: float
    positive: int
    positive_rate: float
    age_share: tuple[float, ...]


@dataclass(frozen=True)
class StratumFlow:
    """One band's split into the patients who did and did not have disease."""

    key: str
    positive: int
    negative: int


@dataclass(frozen=True)
class FactorContribution:
    """A feature's share of the log-odds gap between high and low risk bands."""

    key: str
    label: str
    value: float
    share: float


@dataclass(frozen=True)
class ProfileAxis:
    """One radar axis: the share of an indicator in high risk vs all patients."""

    key: str
    label: str
    high_rate: float
    overall_rate: float
    high_count: int
    overall_count: int

    @property
    def high_value(self) -> float:
        return self.high_rate / 100.0

    @property
    def overall_value(self) -> float:
        return self.overall_rate / 100.0


@dataclass(frozen=True)
class RiskCase:
    rank: int
    age: int
    sex: int
    chest_pain: int
    thalach: int
    oldpeak: float
    ca: float
    probability: float


@dataclass(frozen=True)
class RiskStratificationResult:
    n: int
    positive: int
    negative: int
    strata: tuple[RiskStratum, ...]
    flows: tuple[StratumFlow, ...]
    factors: tuple[FactorContribution, ...]
    profile: tuple[ProfileAxis, ...]
    cases: tuple[RiskCase, ...]
    thresholds: tuple[float, ...]
    roc_auc: float
    brier_score: float
    model: str
    folds: int
    seed: int
    dataset_fingerprint: str
    algorithm_version: str

    @property
    def critical_count(self) -> int:
        return next((item.count for item in self.strata if item.key == "critical"), 0)


def _preprocessor(names: Sequence[str]) -> ColumnTransformer:
    categorical = [name for name in names if name in CATEGORICAL]
    numeric = [name for name in names if name not in CATEGORICAL]
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="most_frequent"), numeric))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _logistic(names: Sequence[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(names)),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
        ]
    )


def _forest(names: Sequence[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(names)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _feature_base(column: str) -> str:
    """Map an encoded column (``categorical__cp_4``) back to its source field."""

    bare = column.split("__", 1)[-1]
    return bare.rsplit("_", 1)[0] if "_" in bare else bare


def _validate(dataset: Dataset, predictors: Sequence[str] | None, folds: int) -> tuple[str, ...]:
    if folds < 2:
        raise RiskStratificationError("交叉验证折数至少为 2")
    frame = dataset.frame
    source = PREDICTORS if predictors is None else predictors
    names = tuple(str(name) for name in source)
    if not names:
        raise RiskStratificationError("至少需要一个特征")
    if len(set(names)) != len(names):
        raise RiskStratificationError("特征不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise RiskStratificationError(f"特征不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise RiskStratificationError("target 不能作为分层特征")
    if frame.empty:
        raise RiskStratificationError("数据集为空")
    return names


def _strata(frame: pd.DataFrame, labels: np.ndarray, y: np.ndarray) -> tuple[RiskStratum, ...]:
    total = len(y)
    result = []
    for index, (key, label, lower, upper) in enumerate(STRATA):
        mask = labels == index
        count = int(mask.sum())
        positive = int(y[mask].sum())
        ages = frame.loc[mask, "age"]
        age_share = tuple(
            float(((ages >= low) & (ages <= high)).sum() / count) if count else 0.0
            for _, low, high in AGE_BANDS
        )
        result.append(
            RiskStratum(
                key=key,
                label=label,
                lower=lower,
                upper=upper,
                count=count,
                percent=count / total * 100.0 if total else 0.0,
                positive=positive,
                positive_rate=positive / count * 100.0 if count else 0.0,
                age_share=age_share,
            )
        )
    return tuple(result)


def _flows(strata: Sequence[RiskStratum]) -> tuple[StratumFlow, ...]:
    return tuple(
        StratumFlow(item.key, item.positive, item.count - item.positive)
        for item in strata
    )


def _factors(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    y: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[FactorContribution, ...]:
    """Decompose the log-odds gap (high + critical bands vs low band) per feature.

    For a linear model the identity ``logit(mean_high) - logit(mean_low)`` equals
    ``sum(beta_j * (mean_high_j - mean_low_j))`` over the encoded design, and the
    encoded columns of one clinical field add back up to that field's share.
    """

    reference = labels == 0
    elevated = labels >= 2
    if not reference.any() or not elevated.any():
        return ()

    pipeline = _logistic(names, seed).fit(frame.loc[:, list(names)], y)
    coefficients = pipeline.named_steps["model"].coef_[0]
    design = pipeline.named_steps["preprocess"].transform(frame.loc[:, list(names)])
    design = design.toarray() if hasattr(design, "toarray") else np.asarray(design)
    columns = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]

    delta = design[elevated].mean(axis=0) - design[reference].mean(axis=0)
    aggregated: dict[str, float] = {}
    for column, value in zip(columns, coefficients * delta):
        base = _feature_base(column)
        aggregated[base] = aggregated.get(base, 0.0) + float(value)

    total = sum(abs(value) for value in aggregated.values())
    if not total:
        return ()
    ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:6]
    return tuple(
        FactorContribution(
            key=key,
            label=COLUMN_BY_NAME[key].label if key in COLUMN_BY_NAME else key,
            value=value,
            share=value / total * 100.0,
        )
        for key, value in ranked
    )


def _profile(frame: pd.DataFrame, labels: np.ndarray) -> tuple[ProfileAxis, ...]:
    """Indicator shares for the high + critical bands versus the whole sample."""

    high = labels >= 2
    if not high.any():
        return ()
    high_total = int(high.sum())
    overall_total = len(frame)
    result = []
    for key, label, predicate in _PROFILE_SPECS:
        mask = predicate(frame).to_numpy(dtype=bool)
        high_count = int(np.sum(mask & high))
        overall_count = int(np.sum(mask))
        result.append(
            ProfileAxis(
                key=key,
                label=label,
                high_rate=high_count / high_total * 100.0 if high_total else 0.0,
                overall_rate=overall_count / overall_total * 100.0 if overall_total else 0.0,
                high_count=high_count,
                overall_count=overall_count,
            )
        )
    return tuple(result)


def _cases(frame: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> tuple[RiskCase, ...]:
    elevated = np.flatnonzero(labels == 3)
    if elevated.size == 0:
        return ()
    ordered = elevated[np.argsort(-probabilities[elevated])][:CASE_LIMIT]
    result = []
    for rank, position in enumerate(ordered, start=1):
        row = frame.iloc[int(position)]
        ca = row["ca"]
        result.append(
            RiskCase(
                rank=rank,
                age=int(row["age"]),
                sex=int(row["sex"]),
                chest_pain=int(row["cp"]),
                thalach=int(row["thalach"]),
                oldpeak=float(row["oldpeak"]),
                ca=float(ca) if pd.notna(ca) else float("nan"),
                probability=float(probabilities[position]),
            )
        )
    return tuple(result)


def run_risk_stratification(
    dataset: Dataset,
    predictors: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> RiskStratificationResult:
    """Assign every patient to a risk band from ensemble out-of-fold probabilities."""

    names = _validate(dataset, predictors, folds)
    frame = dataset.frame.copy()
    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise RiskStratificationError("因变量必须同时包含阳性和阴性样本")

    x = frame.loc[:, list(names)]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    logistic = cross_val_predict(_logistic(names, seed), x, y, cv=splitter, method="predict_proba")[:, 1]
    forest = cross_val_predict(_forest(names, seed), x, y, cv=splitter, method="predict_proba")[:, 1]
    probabilities = (logistic + forest) / 2.0

    labels = np.digitize(probabilities, THRESHOLDS)
    strata = _strata(frame, labels, y)

    return RiskStratificationResult(
        n=len(y),
        positive=int(y.sum()),
        negative=int((y == 0).sum()),
        strata=strata,
        flows=_flows(strata),
        factors=_factors(frame, names, y, labels, seed),
        profile=_profile(frame, labels),
        cases=_cases(frame, labels, probabilities),
        thresholds=THRESHOLDS,
        roc_auc=float(roc_auc_score(y, probabilities)),
        brier_score=float(brier_score_loss(y, probabilities)),
        model="逻辑回归 + 随机森林 集成（概率平均）",
        folds=folds,
        seed=seed,
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "CASE_LIMIT",
    "FactorContribution",
    "PREDICTORS",
    "ProfileAxis",
    "RiskCase",
    "RiskStratificationError",
    "RiskStratificationResult",
    "RiskStratum",
    "STRATA",
    "StratumFlow",
    "THRESHOLDS",
    "run_risk_stratification",
]
