from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Event, Thread
from time import perf_counter
from urllib.error import HTTPError, URLError
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from ai.orchestrator import OrchestrationCancelled, ReportState
from ai.providers import ProviderConfig
from ai.report_generator import AIReportError, AIReportService


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

    def test_reasoning_updates_progress_without_becoming_report_text(self) -> None:
        events = []
        response = StreamingResponse(
            b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning":"still thinking"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"## Report"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        )
        state = ReportState(record=None, progress_callback=events.append)
        with patch("ai.report_generator.urllib.request.urlopen", return_value=response) as request:
            text = AIReportService()._request_report(self.provider, "test-key", "base", "test", state)
        self.assertEqual(text, "## Report")
        self.assertEqual([event["kind"] for event in events], ["model_waiting", "model_thinking", "model_chunk"])
        self.assertNotIn("reasoning_effort", json.loads(request.call_args.args[0].data))

    def test_streaming_output_limit_is_reported_without_returning_truncated_draft(self) -> None:
        cases = (
            ("openai", b'data: {"choices":[{"delta":{"reasoning":"thinking"},"finish_reason":"length"}]}\n\n'),
            ("anthropic", b'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"}}\n\n'),
        )
        for protocol, data in cases:
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(AIReportError, "生成长度上限"):
                    AIReportService._read_sse(StreamingResponse(data), ReportState(record=None), protocol, agent=None)

    def test_context_and_gateway_errors_have_distinct_safe_hints(self) -> None:
        provider = ProviderConfig.custom("openai", "https://example.test/v1", "base")
        cases = (
            (400, b'{"error":{"message":"request exceeds the available context size"}}', "上下文上限"),
            (502, b'{"error":{"message":"sensitive upstream diagnostic"}}', "上游模型"),
        )
        for status, body, hint in cases:
            with self.subTest(status=status):
                error = HTTPError(provider.endpoint, status, "error", {}, BytesIO(body))
                with patch("ai.report_generator.open_custom_request", side_effect=error):
                    with self.assertRaisesRegex(AIReportError, rf"HTTP {status}.*{hint}") as caught:
                        AIReportService()._request_report(provider, "test-key", "base", "test", ReportState(record=None))
                self.assertNotIn("sensitive upstream diagnostic", str(caught.exception))

    def test_timeout_hint_identifies_the_phase(self) -> None:
        state = ReportState(record=None)
        with patch("ai.report_generator.urllib.request.urlopen", side_effect=URLError(TimeoutError())):
            with self.assertRaisesRegex(AIReportError, "等待响应超时"):
                AIReportService()._request_report(self.provider, "test-key", "base", "test", state)
        with (
            patch("ai.report_generator.urllib.request.urlopen", return_value=StreamingResponse()),
            patch.object(AIReportService, "_read_response", side_effect=TimeoutError()),
        ):
            with self.assertRaisesRegex(AIReportError, "读取响应超时"):
                AIReportService()._request_report(self.provider, "test-key", "base", "test", state)

    def test_cancellation_interrupts_open_connection_at_headers_and_body(self) -> None:
        for block_headers in (True, False):
            with self.subTest(block_headers=block_headers):
                self._check_socket_cancellation(block_headers=block_headers)

    def _check_socket_cancellation(self, *, block_headers: bool) -> None:
        waiting = Event()
        unblock_server = Event()
        cancelled = Event()
        outcome = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                with suppress(OSError):
                    if not block_headers:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        self.wfile.write(b'data: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n')
                        self.wfile.flush()
                    waiting.set()
                    unblock_server.wait(5)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            server_thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            server_thread.start()
            provider = ProviderConfig.custom("openai", f"http://127.0.0.1:{server.server_port}/v1", "base")
            reading = Event()

            def progress(event):
                if event["kind"] == "model_thinking":
                    reading.set()

            state = ReportState(record=None, cancel_requested=cancelled.is_set, progress_callback=progress)

            def run():
                try:
                    AIReportService()._request_report(provider, "", "base", "test", state)
                except Exception as exc:
                    outcome.append(exc)

            request_thread = Thread(target=run, daemon=True)
            request_thread.start()
            try:
                self.assertTrue((waiting if block_headers else reading).wait(3))
                start = perf_counter()
                cancelled.set()
                request_thread.join(1)
                self.assertFalse(request_thread.is_alive())
                self.assertLess(perf_counter() - start, 1)
                self.assertEqual(len(outcome), 1)
                self.assertIsInstance(outcome[0], OrchestrationCancelled)
            finally:
                unblock_server.set()
                request_thread.join(2)
                server.shutdown()
                server_thread.join(2)

    def test_prompt_value_limits_large_collections(self) -> None:
        value = AIReportService._compact_prompt_value(
            {"rows": list(range(25)), "text": "x" * 2000}
        )

        self.assertEqual(value["rows"][-1], {"_truncated_items": 5})
        self.assertTrue(value["text"].endswith("…[文本已截断]"))


if __name__ == "__main__":
    unittest.main()
