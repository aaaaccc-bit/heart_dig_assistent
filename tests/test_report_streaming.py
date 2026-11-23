from io import BytesIO
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from ai.orchestrator import ReportState
from ai.providers import ProviderConfig
from ai.report_generator import AIReportService


class StreamingResponse(BytesIO):
    headers = {"Content-Type": "text/event-stream"}

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)


class ReportStreamingTest(unittest.TestCase):
    provider = ProviderConfig(
        "Test", "openai", "https://example.test/v1", "base", ("base",)
    )

    def test_openai_sse_is_assembled_and_reports_chunks(self) -> None:
        stream = "".join(
            [
                'data: {"choices":[{"delta":{"content":"## "}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"结果"}}]}\n\n',
                "data: [DONE]\n\n",
            ]
        ).encode("utf-8")
        events = []
        state = ReportState(
            record=SimpleNamespace(), progress_callback=events.append
        )
        service = AIReportService(timeout=10)

        with patch(
            "ai.report_generator.urllib.request.urlopen",
            return_value=StreamingResponse(stream),
        ):
            result = service._request_report(
                self.provider,
                "test-key",
                "base",
                "生成报告",
                state,
                agent="报告写作 Agent",
            )

        self.assertEqual(result, "## 结果")
        self.assertEqual(
            [event["chars"] for event in events if event["kind"] == "model_chunk"],
            [3, 5],
        )

    def test_non_stream_json_response_remains_supported(self) -> None:
        body = {"choices": [{"message": {"content": "## 结果"}}]}
        response = BytesIO(json.dumps(body).encode())
        state = ReportState(record=SimpleNamespace())
        service = AIReportService(timeout=10)

        with patch(
            "ai.report_generator.urllib.request.urlopen", return_value=response
        ):
            result = service._request_report(
                self.provider,
                "test-key",
                "base",
                "生成报告",
                state,
            )

        self.assertEqual(result, "## 结果")

    def test_connect_timeout_is_separate_from_read_timeout(self) -> None:
        body = {"choices": [{"message": {"content": "## 结果"}}]}
        response = BytesIO(json.dumps(body).encode())
        state = ReportState(record=SimpleNamespace())
        service = AIReportService(timeout=40, connect_timeout=2, read_timeout=7)
        observed = {}

        def open_request(_request, *, timeout):
            observed["timeout"] = timeout
            return response

        with patch("ai.report_generator.urllib.request.urlopen", side_effect=open_request):
            service._request_report(
                self.provider,
                "test-key",
                "base",
                "生成报告",
                state,
            )

        self.assertEqual(observed["timeout"], 2)

    def test_prompt_value_limits_large_collections(self) -> None:
        value = AIReportService._compact_prompt_value(
            {"rows": list(range(25)), "text": "x" * 2000}
        )

        self.assertEqual(value["rows"][-1], {"_truncated_items": 5})
        self.assertTrue(value["text"].endswith("…[文本已截断]"))


if __name__ == "__main__":
    unittest.main()
