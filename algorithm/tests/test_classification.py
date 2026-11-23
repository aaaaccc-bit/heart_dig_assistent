"""Tests for the head-to-head classification evaluation."""

from __future__ import annotations

import unittest

from algorithm.classification import (
    MODELS,
    THRESHOLDS,
    ClassificationEvaluationError,
    run_classification_evaluation,
)
from algorithm.dataset import Dataset, load_dataset


class ClassificationEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.result = run_classification_evaluation(cls.dataset)

    def test_every_model_is_evaluated(self) -> None:
        self.assertEqual([model.key for model in self.result.models], [key for key, _ in MODELS])
        self.assertEqual([model.label for model in self.result.models], [label for _, label in MODELS])
        self.assertEqual(self.result.n, 303)
        self.assertEqual(self.result.positive + self.result.negative, self.result.n)
        self.assertEqual(self.result.dataset_fingerprint, self.dataset.fingerprint)

    def test_threshold_grid_is_shared(self) -> None:
        self.assertEqual(self.result.thresholds, THRESHOLDS)
        self.assertEqual(list(THRESHOLDS), sorted(THRESHOLDS))
        for model in self.result.models:
            self.assertEqual(
                [item.threshold for item in model.thresholds], list(THRESHOLDS)
            )
            self.assertEqual(model.at(0.5).threshold, 0.5)
            self.assertEqual(model.at(0.49).threshold, 0.5)
            self.assertEqual(model.at(0.01).threshold, min(THRESHOLDS))
            self.assertEqual(model.at(0.99).threshold, max(THRESHOLDS))

    def test_discrimination_metrics_are_reasonable(self) -> None:
        for model in self.result.models:
            self.assertGreater(model.roc_auc, 0.85)
            self.assertLessEqual(model.roc_auc, 1.0)
            self.assertGreater(model.pr_auc, 0.85)
            self.assertGreater(model.brier_score, 0.0)
            self.assertLess(model.brier_score, 0.30)

    def test_confusion_counts_match_the_sample(self) -> None:
        for model in self.result.models:
            for item in model.thresholds:
                self.assertEqual(
                    item.true_positive + item.true_negative + item.false_positive + item.false_negative,
                    self.result.n,
                )
                self.assertEqual(item.true_positive + item.false_negative, self.result.positive)
                self.assertEqual(item.true_negative + item.false_positive, self.result.negative)
                expected_sensitivity = item.true_positive / (item.true_positive + item.false_negative)
                self.assertAlmostEqual(item.sensitivity, expected_sensitivity, places=12)

    def test_metrics_stay_in_range(self) -> None:
        for model in self.result.models:
            for item in model.thresholds:
                for value in (
                    item.accuracy,
                    item.sensitivity,
                    item.specificity,
                    item.ppv,
                    item.npv,
                    item.f1,
                ):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
                self.assertGreaterEqual(item.mcc, -1.0)
                self.assertLessEqual(item.mcc, 1.0)

    def test_roc_and_pr_curves_are_complete(self) -> None:
        for model in self.result.models:
            self.assertEqual(model.roc_fpr[0], 0.0)
            self.assertEqual(model.roc_tpr[0], 0.0)
            self.assertEqual(model.roc_fpr[-1], 1.0)
            self.assertEqual(model.roc_tpr[-1], 1.0)
            self.assertEqual(len(model.roc_fpr), len(model.roc_tpr))
            self.assertEqual(len(model.pr_recall), len(model.pr_precision))
            self.assertGreaterEqual(len(model.pr_recall), 2)

    def test_lower_threshold_trades_specificity_for_sensitivity(self) -> None:
        for model in self.result.models:
            low = model.at(0.3)
            high = model.at(0.7)
            self.assertGreater(low.sensitivity, high.sensitivity)
            self.assertGreater(high.specificity, low.specificity)

    def test_same_folds_reproduce_the_metrics(self) -> None:
        repeated = run_classification_evaluation(self.dataset, seed=self.result.seed)
        self.assertEqual(
            [round(model.roc_auc, 12) for model in repeated.models],
            [round(model.roc_auc, 12) for model in self.result.models],
        )
        self.assertEqual(
            [round(model.brier_score, 12) for model in repeated.models],
            [round(model.brier_score, 12) for model in self.result.models],
        )

    def test_custom_predictors(self) -> None:
        result = run_classification_evaluation(
            self.dataset, predictors=("cp", "oldpeak"), folds=3
        )
        self.assertEqual(result.folds, 3)
        self.assertEqual(len(result.models), len(MODELS))
        self.assertEqual(result.n, 303)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(self.dataset, folds=1)
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(self.dataset, predictors=("age", "target"))
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(self.dataset, predictors=("age", "age"))
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(self.dataset, predictors=("age", "missing_column"))
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(self.dataset, predictors=())

    def test_single_class_data_is_rejected(self) -> None:
        frame = self.dataset.frame.copy()
        frame["target"] = 0
        dataset = Dataset(
            frame=frame,
            source=self.dataset.source,
            fingerprint=self.dataset.fingerprint,
            missing=self.dataset.missing,
        )
        with self.assertRaises(ClassificationEvaluationError):
            run_classification_evaluation(dataset)


if __name__ == "__main__":
    unittest.main()
