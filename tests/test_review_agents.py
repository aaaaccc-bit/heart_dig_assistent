"""Tests for the pre-write statistical and citation safety reviews."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ai.review_agents import citation_safety_review, statistical_review


class ReviewAgentsTest(unittest.TestCase):
    def test_statistical_review_checks_present_sections(self) -> None:
        record = SimpleNamespace(
            parameters={
                "回归分析": {"验证指标": {}, "拟合优度": {}},
                "分类评估": {"模型对比": [], "判定阈值": 0.5},
            },
            highlights=(),
        )

        artifact = statistical_review(record)

        self.assertEqual(artifact.agent, "统计审查 Agent")
        self.assertEqual(artifact.warnings, [])
        self.assertEqual(artifact.metadata["checked_sections"], ["回归分析", "分类评估"])

    def test_statistical_review_reports_incomplete_results(self) -> None:
        record = SimpleNamespace(
            parameters={"回归分析": {"验证指标": {}}},
            highlights=(),
        )

        artifact = statistical_review(record)

        self.assertTrue(any("拟合优度" in warning for warning in artifact.warnings))

    def test_citation_and_safety_review_checks_sources_and_risk(self) -> None:
        artifact = citation_safety_review(
            instruction="请判断患者是否应该接受治疗。",
            knowledge_context="[KB-1] 来源：本地知识库/limits.md · 主题：统计限制\n说明文本",
            evidence=[{"agent": "统计 Agent", "evidence_refs": ["parameters.回归分析"]}],
        )

        self.assertEqual(artifact.metadata["source_count"], 1)
        self.assertIn("医疗/治疗", artifact.metadata["risk_categories"])
        self.assertTrue(any("高风险" in warning for warning in artifact.warnings))


if __name__ == "__main__":
    unittest.main()
