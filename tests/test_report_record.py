"""Integration test: the AI report record carries every analysis section.

The analyses are computed once for the class; the assertions focus on the
adapter (section names, JSON validity, highlight coverage), not on the model
numbers themselves.
"""

from __future__ import annotations

import json
import unittest

from algorithm.classification import run_classification_evaluation
from algorithm.clustering import run_clustering
from algorithm.dataset import load_dataset
from algorithm.feature_insight import run_feature_insight
from algorithm.random_forest import run_random_forest
from algorithm.regression import run_logistic_regression
from algorithm.risk_stratification import run_risk_stratification
from ui.ai_report_dialog import build_cardio_report_record
from ui.dashboard_data import load_dashboard


class ReportRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset = load_dataset()
        cls.record = build_cardio_report_record(
            load_dashboard(),
            run_logistic_regression(dataset, predictors=("cp", "oldpeak")),
            run_random_forest(dataset),
            run_risk_stratification(dataset),
            tuple(run_clustering(dataset, method=method) for method in ("kmeans", "single")),
            run_classification_evaluation(dataset),
            run_feature_insight(dataset),
        )
        cls.payload = json.dumps(cls.record.parameters, ensure_ascii=False)

    def test_every_analysis_section_is_present(self) -> None:
        self.assertEqual(
            set(self.record.parameters),
            {
                "阳性率",
                "年龄段",
                "风险因素",
                "组间比较",
                "连续特征分布",
                "回归分析",
                "随机森林分析",
                "风险分层",
                "无监督分群",
                "分类评估",
                "特征洞察",
            },
        )

    def test_payload_is_valid_json_without_nan(self) -> None:
        self.assertNotIn("NaN", self.payload)
        self.assertNotIn("Infinity", self.payload)
        parsed = json.loads(self.payload)
        self.assertEqual(len(parsed["特征洞察"]["特征"]), 13)
        self.assertEqual(len(parsed["特征洞察"]["相关矩阵"]["Spearman"]), 5)
        self.assertEqual(len(parsed["无监督分群"]), 2)
        self.assertEqual(
            [item["选定簇数"] for item in parsed["无监督分群"]], [2, 2]
        )

    def test_highlights_cover_the_models_and_pages(self) -> None:
        titles = [highlight.title for highlight in self.record.highlights]
        for expected in (
            "总体结果",
            "回归分析",
            "随机森林",
            "风险分档",
            "无监督分群",
            "分类评估",
            "特征洞察",
        ):
            self.assertIn(expected, titles)

    def test_optional_sections_disappear_when_missing(self) -> None:
        record = build_cardio_report_record(load_dashboard())
        self.assertEqual(set(record.parameters), {"阳性率", "年龄段", "风险因素", "组间比较", "连续特征分布"})
        self.assertEqual(len(record.highlights), 3)


if __name__ == "__main__":
    unittest.main()
