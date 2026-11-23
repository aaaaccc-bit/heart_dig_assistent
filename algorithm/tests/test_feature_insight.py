"""Tests for the feature-insight module."""

from __future__ import annotations

import math
import unittest

from algorithm.dataset import Dataset, load_dataset
from algorithm.feature_insight import (
    FEATURES,
    NUMERIC,
    FeatureInsightError,
    run_feature_insight,
)


class FeatureInsightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.result = run_feature_insight(cls.dataset)

    def test_every_feature_is_scored(self) -> None:
        self.assertEqual(len(self.result.features), len(FEATURES))
        self.assertEqual(
            {feature.key for feature in self.result.features}, set(FEATURES)
        )
        self.assertEqual(self.result.n, 303)
        self.assertEqual(self.result.positive + self.result.negative, self.result.n)
        self.assertEqual(self.result.dataset_fingerprint, self.dataset.fingerprint)

    def test_features_are_sorted_by_model_reliance(self) -> None:
        drops = [feature.permutation for feature in self.result.features]
        self.assertEqual(drops, sorted(drops, reverse=True))

    def test_scores_are_finite_and_bounded(self) -> None:
        for feature in self.result.features:
            self.assertTrue(math.isfinite(feature.permutation))
            # a single feature can never separate the outcome better than the
            # full model, and an AUC below 0.5 is possible for noise
            self.assertGreater(feature.univariate_auc, 0.35)
            self.assertLessEqual(feature.univariate_auc, 1.0)
        best = max(feature.univariate_auc for feature in self.result.features)
        self.assertGreater(best, 0.70)

    def test_numeric_features_carry_group_statistics(self) -> None:
        numeric = [feature for feature in self.result.features if feature.kind == "numeric"]
        self.assertEqual(len(numeric), len(NUMERIC))
        for feature in numeric:
            self.assertTrue(math.isfinite(feature.positive_mean))
            self.assertTrue(math.isfinite(feature.negative_mean))
            self.assertTrue(math.isfinite(feature.effect_size))
            self.assertAlmostEqual(
                feature.group_delta, feature.positive_mean - feature.negative_mean, places=9
            )
        categorical = [feature for feature in self.result.features if feature.kind == "categorical"]
        self.assertTrue(categorical)
        for feature in categorical:
            self.assertTrue(math.isnan(feature.positive_mean))
            self.assertTrue(math.isnan(feature.effect_size))

    def test_correlation_matrix_is_square_and_symmetric(self) -> None:
        matrix = self.result.correlation
        self.assertEqual(matrix.keys, NUMERIC)
        self.assertEqual(len(matrix.labels), len(matrix.keys))
        self.assertEqual(len(matrix.values), len(matrix.keys))
        for row, key in enumerate(matrix.keys):
            self.assertEqual(len(matrix.values[row]), len(matrix.keys))
            self.assertAlmostEqual(matrix.values[row][row], 1.0, places=9)
            for column in range(len(matrix.keys)):
                self.assertAlmostEqual(
                    matrix.values[row][column], matrix.values[column][row], places=9
                )
                self.assertGreaterEqual(matrix.values[row][column], -1.0)
                self.assertLessEqual(matrix.values[row][column], 1.0)

    def test_known_correlations_have_the_expected_sign(self) -> None:
        matrix = self.result.correlation
        keys = list(matrix.keys)
        age = keys.index("age")
        thalach = keys.index("thalach")
        oldpeak = keys.index("oldpeak")
        self.assertLess(matrix.values[age][thalach], 0.0)
        self.assertLess(matrix.values[thalach][oldpeak], 0.0)

    def test_same_seed_reproduces_the_ranking(self) -> None:
        repeated = run_feature_insight(self.dataset, seed=self.result.seed)
        self.assertEqual(
            [feature.key for feature in repeated.features],
            [feature.key for feature in self.result.features],
        )
        self.assertEqual(
            [round(feature.permutation, 12) for feature in repeated.features],
            [round(feature.permutation, 12) for feature in self.result.features],
        )

    def test_custom_features_and_folds(self) -> None:
        result = run_feature_insight(
            self.dataset, features=("age", "thalach", "oldpeak"), folds=3
        )
        self.assertEqual(len(result.features), 3)
        self.assertEqual(result.folds, 3)
        self.assertEqual(result.correlation.keys, ("age", "thalach", "oldpeak"))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(self.dataset, folds=1)
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(self.dataset, features=())
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(self.dataset, features=("age", "target"))
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(self.dataset, features=("age", "age"))
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(self.dataset, features=("age", "missing_column"))

    def test_single_class_data_is_rejected(self) -> None:
        frame = self.dataset.frame.copy()
        frame["target"] = 0
        dataset = Dataset(
            frame=frame,
            source=self.dataset.source,
            fingerprint=self.dataset.fingerprint,
            missing=self.dataset.missing,
        )
        with self.assertRaises(FeatureInsightError):
            run_feature_insight(dataset)


if __name__ == "__main__":
    unittest.main()
