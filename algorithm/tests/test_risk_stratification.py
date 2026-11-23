"""Tests for the risk-stratification module (bands, flow, drivers, profile)."""

from __future__ import annotations

import unittest

from algorithm.dataset import load_dataset
from algorithm.risk_stratification import (
    CASE_LIMIT,
    THRESHOLDS,
    RiskStratificationError,
    run_risk_stratification,
)


class RiskStratificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = load_dataset()
        cls.result = run_risk_stratification(cls.dataset)

    def test_sample_totals_are_preserved(self) -> None:
        result = self.result
        self.assertEqual(result.n, 303)
        self.assertEqual(result.positive, 139)
        self.assertEqual(result.positive + result.negative, result.n)
        self.assertEqual(sum(stratum.count for stratum in result.strata), result.n)
        self.assertAlmostEqual(
            sum(stratum.percent for stratum in result.strata), 100.0, places=6
        )

    def test_bands_are_contiguous(self) -> None:
        strata = self.result.strata
        self.assertEqual([stratum.key for stratum in strata], ["low", "medium", "high", "critical"])
        self.assertEqual([stratum.lower for stratum in strata], [0.0, *THRESHOLDS])
        self.assertEqual([stratum.upper for stratum in strata], [*THRESHOLDS, 1.0])
        for lower, upper in zip(strata, strata[1:]):
            self.assertEqual(lower.upper, upper.lower)

    def test_positive_counts_are_internally_consistent(self) -> None:
        for stratum, flow in zip(self.result.strata, self.result.flows):
            self.assertEqual(flow.key, stratum.key)
            self.assertEqual(flow.positive, stratum.positive)
            self.assertEqual(flow.positive + flow.negative, stratum.count)
            expected = stratum.positive / stratum.count * 100.0 if stratum.count else 0.0
            self.assertAlmostEqual(stratum.positive_rate, expected, places=9)
        self.assertEqual(
            sum(flow.positive for flow in self.result.flows), self.result.positive
        )

    def test_age_share_is_a_distribution(self) -> None:
        for stratum in self.result.strata:
            self.assertEqual(len(stratum.age_share), 5)
            if stratum.count:
                self.assertAlmostEqual(sum(stratum.age_share), 1.0, places=9)
            else:
                self.assertEqual(sum(stratum.age_share), 0.0)

    def test_factor_shares_are_sorted_and_normalised(self) -> None:
        factors = self.result.factors
        self.assertTrue(factors)
        self.assertLessEqual(len(factors), 6)
        shares = [abs(factor.share) for factor in factors]
        self.assertEqual(shares, sorted(shares, reverse=True))
        # Only the leading factors are kept, so the visible shares sum to less
        # than 100% of the decomposition.
        self.assertGreater(sum(shares), 50.0)
        self.assertLessEqual(sum(shares), 100.0)
        for factor in factors:
            self.assertTrue(factor.label)
            self.assertNotEqual(factor.share, 0.0)

    def test_profile_rates_are_percentages(self) -> None:
        self.assertEqual(len(self.result.profile), 6)
        for axis in self.result.profile:
            self.assertGreaterEqual(axis.high_rate, 0.0)
            self.assertLessEqual(axis.high_rate, 100.0)
            self.assertGreaterEqual(axis.overall_rate, 0.0)
            self.assertLessEqual(axis.overall_rate, 100.0)
            self.assertGreaterEqual(axis.high_count, 0)
            self.assertLessEqual(axis.high_count, axis.overall_count)

    def test_cases_belong_to_the_critical_band(self) -> None:
        cases = self.result.cases
        self.assertTrue(cases)
        self.assertLessEqual(len(cases), CASE_LIMIT)
        self.assertEqual([case.rank for case in cases], list(range(1, len(cases) + 1)))
        probabilities = [case.probability for case in cases]
        self.assertEqual(probabilities, sorted(probabilities, reverse=True))
        for probability in probabilities:
            self.assertGreaterEqual(probability, self.result.thresholds[-1])
            self.assertLessEqual(probability, 1.0)
        # The table can never show more patients than the band holds.
        self.assertLessEqual(len(cases), self.result.critical_count)

    def test_validation_metrics_are_sane(self) -> None:
        self.assertGreater(self.result.roc_auc, 0.85)
        self.assertLessEqual(self.result.roc_auc, 1.0)
        self.assertGreater(self.result.brier_score, 0.0)
        self.assertLess(self.result.brier_score, 0.25)
        self.assertEqual(self.result.dataset_fingerprint, self.dataset.fingerprint)

    def test_same_seed_reproduces_the_bands(self) -> None:
        repeated = run_risk_stratification(self.dataset, seed=self.result.seed)
        self.assertEqual(
            [stratum.count for stratum in repeated.strata],
            [stratum.count for stratum in self.result.strata],
        )
        self.assertEqual([flow.positive for flow in repeated.flows], [flow.positive for flow in self.result.flows])
        self.assertAlmostEqual(repeated.roc_auc, self.result.roc_auc, places=12)

    def test_custom_predictors_are_honoured(self) -> None:
        result = run_risk_stratification(self.dataset, predictors=("age", "sex", "oldpeak"))
        self.assertEqual(sum(stratum.count for stratum in result.strata), 303)
        self.assertTrue(result.factors)
        self.assertEqual({factor.key for factor in result.factors}, {"age", "sex", "oldpeak"})

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(RiskStratificationError):
            run_risk_stratification(self.dataset, folds=1)
        with self.assertRaises(RiskStratificationError):
            run_risk_stratification(self.dataset, predictors=("age", "target"))
        with self.assertRaises(RiskStratificationError):
            run_risk_stratification(self.dataset, predictors=("age", "age"))
        with self.assertRaises(RiskStratificationError):
            run_risk_stratification(self.dataset, predictors=("age", "not_a_column"))


if __name__ == "__main__":
    unittest.main()
