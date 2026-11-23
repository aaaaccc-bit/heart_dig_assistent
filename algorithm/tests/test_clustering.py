"""Tests for the K-means segmentation module."""

from __future__ import annotations

import unittest

from algorithm.clustering import (
    PROFILE_FIELDS,
    ClusteringError,
    run_clustering,
)
from algorithm.dataset import Dataset, load_dataset


class ClusteringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.result = run_clustering(cls.dataset)

    def test_one_run_per_candidate_k(self) -> None:
        self.assertEqual([run.k for run in self.result.runs], [2, 3, 4, 5, 6])
        self.assertEqual(self.result.selected_k, 2)
        self.assertIs(self.result.selected, self.result.runs[0])
        self.assertEqual(self.result.selected_k, max(
            self.result.runs, key=lambda run: run.metric.silhouette
        ).k)

    def test_every_run_partitions_the_sample(self) -> None:
        for run in self.result.runs:
            self.assertEqual(len(run.labels), self.result.n)
            self.assertEqual(len(run.groups), run.k)
            self.assertEqual(sum(group.count for group in run.groups), self.result.n)
            self.assertEqual(
                sum(group.positive for group in run.groups), self.result.positive
            )
            self.assertAlmostEqual(
                sum(group.percent for group in run.groups), 100.0, places=6
            )
            self.assertEqual(sorted(set(run.labels)), list(range(run.k)))

    def test_clusters_are_numbered_by_risk(self) -> None:
        for run in self.result.runs:
            rates = [group.positive_rate for group in run.groups]
            self.assertEqual(rates, sorted(rates, reverse=True))
            self.assertEqual([group.index for group in run.groups], list(range(run.k)))
            self.assertEqual([group.label for group in run.groups], [f"簇 {i + 1}" for i in range(run.k)])

    def test_metrics_are_sane(self) -> None:
        for run in self.result.runs:
            metric = run.metric
            self.assertEqual(metric.k, run.k)
            self.assertGreater(metric.silhouette, -1.0)
            self.assertLessEqual(metric.silhouette, 1.0)
            self.assertGreater(metric.calinski_harabasz, 0.0)
            self.assertGreater(metric.davies_bouldin, 0.0)
            self.assertGreaterEqual(metric.ari, -1.0)
            self.assertLessEqual(metric.ari, 1.0)
        # The cohort overlaps heavily: two clusters is the only structure the
        # silhouette supports, and even that is weak.
        self.assertGreater(self.result.selected.metric.silhouette, 0.10)
        self.assertLess(self.result.selected.metric.silhouette, 0.40)

    def test_projection_is_two_dimensional(self) -> None:
        self.assertEqual(len(self.result.points), self.result.n)
        self.assertEqual(len(self.result.explained), 2)
        for ratio in self.result.explained:
            self.assertGreater(ratio, 0.0)
            self.assertLess(ratio, 1.0)
        self.assertLess(sum(self.result.explained), 1.0)
        self.assertEqual(self.result.points[0][0], self.result.points[0][0])  # not NaN

    def test_profiles_report_clinical_means(self) -> None:
        overall = {
            key: float(self.dataset.frame[key].astype(float).mean())
            for key, _, _ in PROFILE_FIELDS
        }
        for run in self.result.runs:
            for group in run.groups:
                self.assertEqual(len(group.features), len(PROFILE_FIELDS))
                self.assertEqual(
                    [feature.key for feature in group.features],
                    [key for key, _, _ in PROFILE_FIELDS],
                )
                for feature in group.features:
                    self.assertAlmostEqual(feature.overall, overall[feature.key], places=9)
                    self.assertTrue(feature.unit is not None)

    def test_same_seed_reproduces_the_segmentation(self) -> None:
        repeated = run_clustering(self.dataset, seed=self.result.seed)
        self.assertEqual(repeated.selected.labels, self.result.selected.labels)
        self.assertEqual(
            [group.count for group in repeated.selected.groups],
            [group.count for group in self.result.selected.groups],
        )
        self.assertAlmostEqual(
            repeated.selected.metric.silhouette,
            self.result.selected.metric.silhouette,
            places=12,
        )

    def test_custom_k_range_and_selection(self) -> None:
        result = run_clustering(self.dataset, k_range=(2, 3), selected_k=3)
        self.assertEqual([run.k for run in result.runs], [2, 3])
        self.assertEqual(result.selected_k, 3)
        self.assertEqual(len(result.selected.groups), 3)

    def test_custom_features(self) -> None:
        result = run_clustering(self.dataset, features=("age", "thalach", "oldpeak"), k_range=(2,))
        self.assertEqual(result.features, ("age", "thalach", "oldpeak"))
        self.assertEqual(len(result.selected.labels), 303)

    def test_kmeans_is_the_default_and_has_no_singletons(self) -> None:
        self.assertEqual(self.result.method_key, "kmeans")
        self.assertFalse(self.result.has_singletons)
        for run in self.result.runs:
            self.assertTrue(all(group.count > 1 for group in run.groups))

    def test_single_linkage_chains_on_this_data(self) -> None:
        result = run_clustering(self.dataset, method="single")
        self.assertEqual(result.method_key, "single")
        self.assertIn("最小距离", result.method)
        self.assertEqual([run.k for run in result.runs], [2, 3, 4, 5, 6])
        for run in result.runs:
            self.assertEqual(sum(group.count for group in run.groups), result.n)
            self.assertEqual(sorted(set(run.labels)), list(range(run.k)))
        # The known pathology: one giant cluster plus singleton patients, so the
        # silhouette is inflated and the ARI against the outcome collapses.
        self.assertTrue(result.has_singletons)
        self.assertGreaterEqual(
            max(group.count for group in result.selected.groups), 290
        )
        self.assertLess(abs(result.selected.metric.ari), 0.05)

    def test_single_linkage_can_be_forced_to_a_k(self) -> None:
        result = run_clustering(self.dataset, method="single", k_range=(3,), selected_k=3)
        self.assertEqual(result.selected_k, 3)
        self.assertEqual(len(result.selected.groups), 3)
        self.assertEqual(result.selected.labels, run_clustering(
            self.dataset, method="single", k_range=(3,)
        ).selected.labels)

    def test_unknown_method_is_rejected(self) -> None:
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, method="ward")

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, k_range=(1, 2))
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, k_range=(2, 2))
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, k_range=())
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, features=("age", "target"))
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, features=("age", "age"))
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, features=("age", "missing_column"))
        with self.assertRaises(ClusteringError):
            run_clustering(self.dataset, features=())

    def test_too_few_rows_are_rejected(self) -> None:
        small = Dataset(
            frame=self.dataset.frame.head(5).copy(),
            source=self.dataset.source,
            fingerprint=self.dataset.fingerprint,
            missing=self.dataset.missing,
        )
        with self.assertRaises(ClusteringError):
            run_clustering(small, k_range=(5, 6))


if __name__ == "__main__":
    unittest.main()
