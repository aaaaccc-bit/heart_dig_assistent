"""K-means patient segmentation with k selection and a 2-D projection.

The prediction modules answer "how risky is this patient"; this module asks a
different, unsupervised question: *does the cohort fall into natural groups at
all, and do those groups differ in outcome?*

The data is small (303 rows, 13 mixed-type columns), so the result is reported
honestly: every candidate ``k`` keeps its silhouette / Calinski-Harabasz /
Davies-Bouldin score plus the adjusted Rand index against the binary outcome,
and the page shows the whole k scan instead of hiding it behind one number.
Clusters are numbered by observed positive rate, so "簇 1" is always the
highest-risk group.

Pre-processing (median / most-frequent imputation, standardisation of numeric
columns, one-hot encoding of categorical columns) is fitted on the whole
dataset: unlike a supervised model there is no label leakage to avoid, and the
same transform is reused for every candidate ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from algorithm.dataset import Dataset
from algorithm.schema import TARGET_COLUMN


class ClusteringError(ValueError):
    """Raised when a segmentation cannot be configured or fitted."""


ALGORITHM_VERSION = "1.0.0-clustering"

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

#: Candidate cluster counts scanned in one run.
K_RANGE = (2, 3, 4, 5, 6)

#: Available methods; the first entry is the page default.
METHODS = (
    ("kmeans", "KMeans（数值标准化 + 类别独热）"),
    ("single", "最小距离层次聚类（单链，欧氏距离）"),
)

#: Clinical columns reported per cluster.
PROFILE_FIELDS = (
    ("age", "年龄", "岁"),
    ("thalach", "最大心率", "bpm"),
    ("oldpeak", "ST 压低", "mm"),
    ("ca", "主要血管数", "根"),
    ("trestbps", "静息血压", "mmHg"),
)


@dataclass(frozen=True)
class ClusterMetric:
    k: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    ari: float


@dataclass(frozen=True)
class ClusterFeature:
    key: str
    label: str
    unit: str
    mean: float
    overall: float


@dataclass(frozen=True)
class ClusterGroup:
    index: int
    label: str
    count: int
    percent: float
    positive: int
    positive_rate: float
    features: tuple[ClusterFeature, ...]


@dataclass(frozen=True)
class ClusterRun:
    k: int
    metric: ClusterMetric
    groups: tuple[ClusterGroup, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class ClusteringResult:
    n: int
    positive: int
    negative: int
    features: tuple[str, ...]
    runs: tuple[ClusterRun, ...]
    selected_k: int
    points: tuple[tuple[float, float], ...]
    explained: tuple[float, float]
    method: str
    seed: int
    dataset_fingerprint: str
    algorithm_version: str
    method_key: str = "kmeans"

    @property
    def selected(self) -> ClusterRun:
        return next(run for run in self.runs if run.k == self.selected_k)

    @property
    def has_singletons(self) -> bool:
        """True when at least one cluster holds a single patient."""

        return any(group.count <= 1 for group in self.selected.groups)


def _validate(
    dataset: Dataset,
    features: Sequence[str] | None,
    k_range: Sequence[int],
    method: str,
) -> tuple[str, ...]:
    if method not in dict(METHODS):
        raise ClusteringError(f"未知的聚类方法: {method}")
    frame = dataset.frame
    source = FEATURES if features is None else features
    names = tuple(str(name) for name in source)
    if not names:
        raise ClusteringError("至少需要一个聚类特征")
    if len(set(names)) != len(names):
        raise ClusteringError("聚类特征不能重复")
    absent = [name for name in names if name not in frame.columns]
    if absent:
        raise ClusteringError(f"特征不存在: {', '.join(absent)}")
    if TARGET_COLUMN in names:
        raise ClusteringError("target 不能作为聚类特征")
    if frame.empty:
        raise ClusteringError("数据集为空")

    candidates = tuple(int(k) for k in k_range)
    if not candidates:
        raise ClusteringError("至少需要一个候选簇数")
    if len(set(candidates)) != len(candidates):
        raise ClusteringError("候选簇数不能重复")
    if min(candidates) < 2:
        raise ClusteringError("簇数至少为 2")
    if max(candidates) >= len(frame):
        raise ClusteringError("簇数必须小于样本数")
    return names


def _matrix(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    """Standardised numeric columns plus one-hot categorical columns."""

    numeric = [name for name in names if name not in CATEGORICAL]
    categorical = [name for name in names if name in CATEGORICAL]
    blocks = []
    if numeric:
        values = SimpleImputer(strategy="median").fit_transform(frame[numeric].astype(float))
        blocks.append(StandardScaler().fit_transform(values))
    if categorical:
        values = SimpleImputer(strategy="most_frequent").fit_transform(frame[categorical].astype(object))
        encoded = OneHotEncoder(handle_unknown="ignore").fit_transform(values)
        blocks.append(encoded.toarray() if hasattr(encoded, "toarray") else np.asarray(encoded))
    return np.hstack(blocks)


def _groups(
    frame: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    overall: dict[str, float],
) -> tuple[tuple[ClusterGroup, ...], np.ndarray]:
    positive = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    total = len(frame)

    # Number clusters by observed positive rate so 簇 1 is always the riskiest.
    rates = {index: float(positive[labels == index].mean()) if np.any(labels == index) else 0.0 for index in range(k)}
    order = sorted(range(k), key=lambda index: (rates[index], int(np.sum(labels == index))), reverse=True)
    remap = {old: new for new, old in enumerate(order)}
    ordered = np.array([remap[label] for label in labels])

    groups = []
    for index in range(k):
        mask = ordered == index
        count = int(mask.sum())
        cases = int(positive[mask].sum())
        features = []
        for key, label, unit in PROFILE_FIELDS:
            column = frame.loc[mask, key].astype(float)
            mean = float(column.mean()) if count and column.notna().any() else 0.0
            features.append(
                ClusterFeature(
                    key=key,
                    label=label,
                    unit=unit,
                    mean=mean,
                    overall=overall[key],
                )
            )
        groups.append(
            ClusterGroup(
                index=index,
                label=f"簇 {index + 1}",
                count=count,
                percent=count / total * 100.0 if total else 0.0,
                positive=cases,
                positive_rate=cases / count * 100.0 if count else 0.0,
                features=tuple(features),
            )
        )
    return tuple(groups), ordered


def run_clustering(
    dataset: Dataset,
    features: Sequence[str] | None = None,
    k_range: Sequence[int] = K_RANGE,
    selected_k: int | None = None,
    seed: int = 42,
    method: str = "kmeans",
) -> ClusteringResult:
    """Fit the requested method for every candidate ``k`` and report the scan.

    ``method`` is ``"kmeans"`` or ``"single"`` (single-linkage agglomerative
    clustering). Single linkage is prone to chaining on this dataset, which the
    result exposes through :attr:`ClusteringResult.has_singletons`; the page
    warns instead of hiding it.
    """

    names = _validate(dataset, features, k_range, method)
    method_label = dict(METHODS)[method]
    frame = dataset.frame.copy()
    positive = (frame[TARGET_COLUMN].to_numpy(dtype=float) > 0).astype(int)
    if np.unique(positive).size != 2:
        raise ClusteringError("因变量必须同时包含阳性和阴性样本")

    matrix = _matrix(frame, names)
    projection = PCA(n_components=2, random_state=seed).fit(matrix)
    coordinates = projection.transform(matrix)
    explained = tuple(float(value) for value in projection.explained_variance_ratio_[:2])
    overall = {
        key: float(frame[key].astype(float).mean())
        for key, _, _ in PROFILE_FIELDS
    }

    runs = []
    ordered_labels: dict[int, np.ndarray] = {}
    for k in sorted(int(value) for value in k_range):
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        else:
            model = AgglomerativeClustering(
                n_clusters=k, linkage="single", metric="euclidean"
            )
        labels = model.fit_predict(matrix)
        groups, relabelled = _groups(frame, labels, k, overall)
        ordered_labels[k] = relabelled
        runs.append(
            ClusterRun(
                k=k,
                metric=ClusterMetric(
                    k=k,
                    silhouette=float(silhouette_score(matrix, labels)),
                    calinski_harabasz=float(calinski_harabasz_score(matrix, labels)),
                    davies_bouldin=float(davies_bouldin_score(matrix, labels)),
                    ari=float(adjusted_rand_score(positive, labels)),
                ),
                groups=groups,
                labels=tuple(int(value) for value in relabelled),
            )
        )

    if selected_k is None:
        selected_k = max(runs, key=lambda run: run.metric.silhouette).k
    if selected_k not in ordered_labels:
        raise ClusteringError(f"选定簇数 {selected_k} 不在扫描范围内")

    return ClusteringResult(
        n=len(frame),
        positive=int(positive.sum()),
        negative=int((positive == 0).sum()),
        features=names,
        runs=tuple(runs),
        selected_k=selected_k,
        points=tuple((float(x), float(y)) for x, y in coordinates),
        explained=(explained[0], explained[1]),
        method=method_label,
        seed=seed,
        dataset_fingerprint=dataset.fingerprint,
        algorithm_version=ALGORITHM_VERSION,
        method_key=method,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "CATEGORICAL",
    "ClusterFeature",
    "ClusterGroup",
    "ClusterMetric",
    "ClusterRun",
    "ClusteringError",
    "ClusteringResult",
    "K_RANGE",
    "METHODS",
    "run_clustering",
]
