"""Head-to-head evaluation of the three binary classifiers.

The sample-analysis pages each answer "how does *this* model look".  This module
answers a different question: *with the same folds, the same pre-processing and
the same threshold grid, which model should be used?*

Each model keeps the configuration used by its own page (logistic regression
with one-hot encoding, a 400-tree balanced random forest, and the 32/16 MLP),
and all three are scored on out-of-fold probabilities from one stratified
5-fold split.  Threshold-dependent metrics are pre-computed for 0.20-0.80 so the
page can switch the decision threshold without re-fitting anything.
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class ClassificationEvaluationError(ValueError):
    """Raised when the comparison cannot be configured or fitted."""


ALGORITHM_VERSION = "1.0.0-classification-evaluation"

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

#: Display name per model key; the order is the one shown on the page.
MODELS = (
    ("logistic", "Logistic 回归"),
    ("forest", "随机森林"),
    ("neural", "神经网络（MLP）"),
)

#: Threshold grid pre-computed for every model.
THRESHOLDS = tuple(round(0.20 + index * 0.05, 2) for index in range(13))


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    mcc: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(frozen=True)
class ModelEvaluation:
    key: str
    label: str
    roc_auc: float
    pr_auc: float
    brier_score: float
    roc_fpr: tuple[float, ...]
    roc_tpr: tuple[float, ...]
    pr_recall: tuple[float, ...]
    pr_precision: tuple[float, ...]
    thresholds: tuple[ThresholdMetrics, ...]

    def at(self, threshold: float) -> ThresholdMetrics:
        """Return the pre-computed metrics closest to *threshold*."""

        return min(self.thresholds, key=lambda item: abs(item.threshold - threshold))


@dataclass(frozen=True)
class ClassificationEvaluation:
    n: int
    positive: int
    negative: int
    models: tuple[ModelEvaluation, ...]
    thresholds: tuple[float, ...]
    folds: int
    seed: int
    dataset_fingerprint: str
    algorithm_version: str

    def model(self, key: str) -> ModelEvaluation:
        return next(model for model in self.models if model.key == key)


def _preprocessor(names: Sequence[str], *, scale_numeric: bool = False) -> ColumnTransformer:
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
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
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


def _neural(names: Sequence[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", _preprocessor(names, scale_numeric=True)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=35,
                    random_state=seed,
                ),
            ),
        ]
    )


def _metrics_at(threshold: float, y: np.ndarray, probabilities: np.ndarray) -> ThresholdMetrics:
    predicted = probabilities >= threshold
    true_positive = int(np.sum(predicted & (y == 1)))
    true_negative = int(np.sum(~predicted & (y == 0)))
    false_positive = int(np.sum(predicted & (y == 0)))
    false_negative = int(np.sum(~predicted & (y == 1)))

    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    ppv = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    npv = true_negative / (true_negative + false_negative) if true_negative + false_negative else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else 0.0
    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=float(np.mean(predicted == y)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        ppv=float(ppv),
        npv=float(npv),
        f1=float(f1),
        mcc=float(matthews_corrcoef(y, predicted)),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def _evaluate(key: str, label: str, y: np.ndarray, probabilities: np.ndarray) -> ModelEvaluation:
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    return ModelEvaluation(
        key=key,
        label=label,
        roc_auc=float(roc_auc_score(y, probabilities)),
        pr_auc=float(average_precision_score(y, probabilities)),
        brier_score=float(brier_score_loss(y, probabilities)),
        roc_fpr=tuple(float(value) for value in fpr),
        roc_tpr=tuple(float(value) for value in tpr),
        pr_recall=tuple(float(value) for value in recall),
        pr_precision=tuple(float(value) for value in precision),
        thresholds=tuple(_metrics_at(threshold, y, probabilities) for threshold in THRESHOLDS),
    )


def run_classification_evaluation(
    dataset: Dataset,
    predictors: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 42,
) -> ClassificationEvaluation:
    """Fit every classifier on the same folds and return comparable metrics."""

    if folds < 2:
        raise ClassificationEvaluationError("交叉验证折数至少为 2")
    frame = dataset.frame.copy()
    source = PREDICTORS if predictors is None else predictors
    names = tuple(str(name) for name in source)
    if not names:
        raise ClassificationEvaluationError("至少需要一个自变量")
    if len(set(names)) != len(names):
        raise ClassificationEvaluationError("自变量不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise ClassificationEvaluationError(f"自变量不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise ClassificationEvaluationError("target 不能作为自变量")
    if frame.empty:
        raise ClassificationEvaluationError("数据集为空")

    y = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(y).size != 2:
        raise ClassificationEvaluationError("因变量必须同时包含阳性和阴性样本")

    x = frame.loc[:, list(names)]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    builders = {"logistic": _logistic, "forest": _forest, "neural": _neural}
    evaluated = []
    for key, label in MODELS:
        probabilities = cross_val_predict(
            builders[key](names, seed), x, y, cv=splitter, method="predict_proba"
        )[:, 1]
        evaluated.append(_evaluate(key, label, y, probabilities))

    return ClassificationEvaluation(
        n=len(y),
        positive=int(y.sum()),
        negative=int((y == 0).sum()),
        models=tuple(evaluated),
        thresholds=THRESHOLDS,
        folds=folds,
        seed=seed,
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "ClassificationEvaluation",
    "ClassificationEvaluationError",
    "MODELS",
    "ModelEvaluation",
    "THRESHOLDS",
    "ThresholdMetrics",
    "run_classification_evaluation",
]
