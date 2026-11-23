from datetime import datetime
from io import BytesIO
import json
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from ai.providers import ProviderConfig
from ai.report_generator import AIReportCancelled, AIReportService


class ReportWorkflowTest(unittest.TestCase):
    provider = ProviderConfig(
        "Test", "openai", "https://example.test/v1", "base", ("base", "advanced"),
        analysis_model="advanced",
    )

    def setUp(self):
        self.service = AIReportService(retrieval_enabled=False)
        self.record = SimpleNamespace(
            title="统计报告",
            kind_label="描述性分析",
            dataset_name="测试数据",
            created_at=datetime(2026, 11, 19),
            trajectory_count=10,
            group_count=1,
            metrics=(),
            highlights=(),
            parameters={"阳性率": 30},
        )
        self.events = []
        self.requests = []

    def generate(self, *drafts, model=None, cancel_requested=None, progress_callback=None):
        responses = iter(drafts)

        def respond(request, *, timeout):
            self.requests.append(json.loads(request.data))
            draft = next(responses)
            if isinstance(draft, Exception):
                raise draft
            if self.provider.protocol == "anthropic":
                response = {"content": [{"type": "text", "text": draft}]}
            else:
                response = {"choices": [{"message": {"content": draft}}]}
            return BytesIO(json.dumps(response).encode())

        with (
            patch("ai.report_generator.urllib.request.urlopen", side_effect=respond),
            patch("ai.report_generator.cancellable_request_opener", return_value=respond),
        ):
            return self.service.generate(
                self.record, self.provider, "test-key", model,
                progress_callback=progress_callback or self.events.append,
                cancel_requested=cancel_requested,
                safety_mode="hard_block", safety_strength=66,
            )

    def statuses(self):
        return {
            event["agent"]: event["status"]
            for event in self.events
            if event["kind"] == "agent_finished"
        }

    def test_clean_descriptive_report_skips_model_analysis_and_revision(self):
        report = self.generate("## 结果\n样本量为 10。")

        self.assertIn("样本量为 10", report)
        self.assertEqual([request["model"] for request in self.requests], ["base"])
        self.assertEqual(self.statuses()["统计审查 Agent"], "skipped")
        self.assertEqual(self.statuses()["知识检索 Agent"], "skipped")
        self.assertEqual(self.statuses()["报告修订 Agent"], "skipped")
        self.assertNotIn("阳性率分析 Agent", self.statuses())

    def test_model_analysis_routes_to_advanced_and_checks_valid_citations(self):
        self.record.parameters["回归分析"] = {"验证指标": {}, "拟合优度": {}}
        self.service = AIReportService(knowledge_base=SimpleNamespace(
            format_context=lambda *_args, **_kwargs: "[KB-1] 来源：统计说明 · 主题：回归",
            retrieval_status={},
        ))

        report = self.generate("## 结果\n回归指标见表 [KB-1]。")

        self.assertEqual([request["model"] for request in self.requests], ["advanced"])
        self.assertEqual(self.statuses()["统计审查 Agent"], "completed")
        self.assertEqual(self.statuses()["修订规划 Agent"], "skipped")
        self.assertIn("- [KB-1] 统计说明", report)
        review_waves = [
            event
            for event in self.events
            if event.get("kind") == "wave_started"
            and "统计审查 Agent" in event.get("agents", [])
        ]
        self.assertEqual(
            review_waves[0]["agents"],
            ["统计审查 Agent", "引用与安全审查 Agent"],
        )

    def test_unknown_citation_with_no_sources_stays_in_human_review(self):
        report = self.generate("首稿 [KB-8]。", "修订后结果。")

        self.assertEqual([request["model"] for request in self.requests], ["base"])
        self.assertIn("[KB-8]", report)
        self.assertIn("引用校验提示", report)
        self.assertEqual(self.statuses()["修订规划 Agent"], "skipped")

    def test_manual_model_and_persistent_risk_preserve_hard_block(self):
        report = self.generate("患者应该用药。", "患者必须接受治疗。", model="base")

        self.assertEqual([request["model"] for request in self.requests], ["base", "base"])
        self.assertIn("## 安全硬阻断", report)
        self.assertIn("需要人工复核", report)

    def test_revision_failure_preserves_first_draft_with_citation_warning(self):
        report = self.generate("患者应该用药。", URLError("offline"))

        self.assertIn("患者应该用药", report)
        self.assertIn("安全硬阻断", report)
        self.assertIn("自动修订请求失败", report)
        self.assertEqual(self.statuses()["修订复审 Agent"], "skipped")

    def test_anthropic_revision_uses_the_default_when_no_advanced_model_is_configured(self):
        self.provider = ProviderConfig(
            "Test", "anthropic", "https://example.test/v1", "base", ("base",)
        )

        report = self.generate("患者应该用药。", "修订后结果。")

        self.assertEqual([request["model"] for request in self.requests], ["base", "base"])
        self.assertIn("修订后结果", report)

    def test_automatic_revision_downgrades_to_default_model(self):
        self.record.parameters["回归分析"] = {"验证指标": {}, "拟合优度": {}}

        self.generate("患者应该用药。", "修订后结果。")

        self.assertEqual([request["model"] for request in self.requests], ["advanced", "base"])

    def test_cancellation_after_planning_prevents_revision_request(self):
        cancel = Event()

        def progress(event):
            if event["kind"] == "agent_finished" and event["agent"] == "修订规划 Agent":
                cancel.set()

        with self.assertRaises(AIReportCancelled):
            self.generate("首稿 [KB-8]。", cancel_requested=cancel.is_set, progress_callback=progress)

        self.assertEqual(len(self.requests), 1)
