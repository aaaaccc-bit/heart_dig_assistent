"""Feature-level insight for the Cleveland dataset.

Two questions are answered side by side:

* **How much does the model rely on a feature?**  Permutation importance: inside
  every cross-validation fold the fitted random forest is asked to score the
  held-out patients again with one column shuffled. The mean AUC drop is the
  feature's contribution to the model, on a scale that is comparable across
  numeric and categorical columns.
* **How far does a feature get on its own?**  A one-feature logistic model is
  cross-validated and scored by AUC. This ignores the other twelve columns, so
  it says how separable the outcome is by that indicator alone.

The two rankings often disagree, and the disagreement is informative: a feature
can be highly discriminative yet almost useless to the model when a correlated
partner already carries the same signal (``thalach`` vs ``oldpeak`` here).

The module also returns the positive/negative group means with Cohen's d for the
continuous columns and the Spearman correlation matrix among them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class FeatureInsightError(ValueError):
    """Raised when the feature analysis cannot be configured or fitted."""


ALGORITHM_VERSION = "1.0.0-feature-insight"

CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

#: All dataset columns except the outcome, in schema order.
FEATURES = (
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

#: Continuous columns used for the correlation matrix.
NUMERIC = ("age", "trestbps", "chol", "thalach", "oldpeak")


@dataclass(frozen=True)
class FeatureSignal:
    key: str
    label: str
    unit: str
    kind: str  # "numeric" | "categorical"
    permutation: float
    univariate_auc: float
    positive_mean: float
    negative_mean: float
    effect_size: float

    @property
    def group_delta(self) -> float:
        return self.positive_mean - self.negative_mean


@dataclass(frozen=True)
class CorrelationMatrix:
    keys: tuple[str, ...]
    labels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FeatureInsightResult:
    n: int
    positive: int
    negative: int
    features: tuple[FeatureSignal, ...]
    correlation: CorrelationMatrix
    folds: int
    seed: int
    dataset_fingerprint: str
    algorithm_version: str

    def signal(self, key: str) -> FeatureSignal:
        return next(feature for feature in self.features if feature.key == key)


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
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _forest(names: Sequence[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(names)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _single_feature_auc(x: pd.DataFrame, feature: str, y: np.ndarray, splitter) -> float:
    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor([feature])),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    probabilities = cross_val_predict(
        pipeline, x[[feature]], y, cv=splitter, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, probabilities))


def _permutation_drops(
    x: pd.DataFrame, y: np.ndarray, names: Sequence[str], splitter, seed: int
) -> dict[str, float]:
    """Mean AUC drop when one column is shuffled inside each validation fold."""

    drops: dict[str, list[float]] = {name: [] for name in names}
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        pipeline = _forest(names, seed).fit(x.iloc[train], y[train])
        held_out = x.iloc[test]
        target = y[test]
        baseline = roc_auc_score(target, pipeline.predict_proba(held_out)[:, 1])
        generator = np.random.default_rng(seed + fold)
        for name in names:
            shuffled = held_out.copy()
            values = shuffled[name].to_numpy().copy()
            generator.shuffle(values)
            shuffled[name] = values
            score = roc_auc_score(target, pipeline.predict_proba(shuffled)[:, 1])
            drops[name].append(float(baseline - score))
    return {name: float(np.mean(scores)) for name, scores in drops.items()}


def _effect_size(positive: pd.Series, negative: pd.Series) -> float:
    first, second = positive.dropna(), negative.dropna()
    if len(first) < 2 or len(second) < 2:
        return 0.0
    pooled = math.sqrt(
        ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1))
        / (len(first) + len(second) - 2)
    )
    if not pooled:
        return 0.0
    return float((first.mean() - second.mean()) / pooled)


def _signals(
    frame: pd.DataFrame,
    names: Sequence[str],
    y: np.ndarray,
    drops: dict[str, float],
    aucs: dict[str, float],
) -> tuple[FeatureSignal, ...]:
    from algorithm.schema import COLUMN_BY_NAME

    positive = frame[y == 1]
    negative = frame[y == 0]
    signals = []
    for name in names:
        spec = COLUMN_BY_NAME[name]
        numeric = name not in CATEGORICAL
        if numeric:
            positive_mean = float(positive[name].astype(float).mean())
            negative_mean = float(negative[name].astype(float).mean())
            effect = _effect_size(positive[name].astype(float), negative[name].astype(float))
        else:
            positive_mean = negative_mean = effect = float("nan")
        signals.append(
            FeatureSignal(
                key=name,
                label=spec.label,
                unit=spec.unit,
                kind="numeric" if numeric else "categorical",
                permutation=drops[name],
                univariate_auc=aucs[name],
                positive_mean=positive_mean,
                negative_mean=negative_mean,
                effect_size=effect,
            )
        )
    return tuple(sorted(signals, key=lambda item: item.permutation, reverse=True))


def _correlation(frame: pd.DataFrame, names: Sequence[str]) -> CorrelationMatrix:
    from algorithm.schema import COLUMN_BY_NAME

    keys = tuple(name for name in names if name in NUMERIC)
    if not keys:
        return CorrelationMatrix((), (), ())
    matrix = frame[list(keys)].astype(float).corr(method="spearman")
    values = tuple(
        tuple(float(matrix.loc[first, second]) for second in keys) for first in keys
    )
    return CorrelationMatrix(
        keys=keys,
        labels=tuple(COLUMN_BY_NAME[key].label for key in keys),
        values=values,
    )


def run_feature_insight(
    dataset: Dataset,
    features: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> FeatureInsightResult:
    """Rank features by model reliance and by single-feature discrimination."""

    if folds < 2:
        raise FeatureInsightError("交叉验证折数至少为 2")
    frame = dataset.frame.copy()
    source = FEATURES if features is None else features
    names = tuple(str(name) for name in source)
    if not names:
        raise FeatureInsightError("至少需要一个特征")
    if len(set(names)) != len(names):
        raise FeatureInsightError("特征不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise FeatureInsightError(f"特征不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise FeatureInsightError("target 不能作为特征")
    if frame.empty:
        raise FeatureInsightError("数据集为空")

    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise FeatureInsightError("因变量必须同时包含阳性和阴性样本")

    x = frame.loc[:, list(names)]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    drops = _permutation_drops(x, y, names, splitter, seed)
    aucs = {name: _single_feature_auc(x, name, y, splitter) for name in names}

    return FeatureInsightResult(
        n=len(y),
        positive=int(y.sum()),
        negative=int((y == 0).sum()),
        features=_signals(frame, names, y, drops, aucs),
        correlation=_correlation(frame, names),
        folds=folds,
        seed=seed,
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "CATEGORICAL",
    "CorrelationMatrix",
    "FEATURES",
    "FeatureInsightError",
    "FeatureInsightResult",
    "FeatureSignal",
    "NUMERIC",
    "run_feature_insight",
]
