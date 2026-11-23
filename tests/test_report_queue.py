from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json
from io import BytesIO

from ai.memory import LongTermMemory
from ai.providers import ProviderConfig
from ai.report_generator import AIReportService
from ai.task_queue import DistributedTaskQueue


class ReportQueueIntegrationTest(unittest.TestCase):
    provider = ProviderConfig(
        "Test", "openai", "https://example.test/v1", "base", ("base",)
    )

    def _record(self):
        return SimpleNamespace(
            title="报告",
            kind_label="描述性分析",
            dataset_name="测试集",
            created_at=datetime(2026, 11, 21),
            trajectory_count=3,
            group_count=1,
            metrics=(),
            highlights=(),
            parameters={"阳性率": 30},
        )

    def test_enqueue_and_process_without_persisting_api_key(self):
        with TemporaryDirectory() as directory:
            queue = DistributedTaskQueue(Path(directory) / "tasks.sqlite3")
            service = AIReportService(
                retrieval_enabled=False,
                memory_enabled=False,
                task_queue=queue,
            )
            task_id = service.enqueue_report(self._record(), self.provider)
            queued = queue.get(task_id)
            self.assertNotIn("api_key", queued.payload)

            with patch.object(service, "generate", return_value="生成的报告") as generate:
                report = service.process_next_queued_report(
                    lambda provider_name: self.assertEqual(provider_name, "Test") or "secret-key",
                    worker_id="worker-a",
                )
                generate.assert_called_once()

            completed = queue.get(task_id)

        self.assertEqual(report, "生成的报告")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.result["report"], "生成的报告")

    def test_cross_task_experience_is_injected_into_writer_prompt(self):
        with TemporaryDirectory() as directory:
            memory = LongTermMemory(Path(directory) / "memory.sqlite3")
            record = self._record()
            memory.record_experience(
                task_type=record.kind_label,
                signal="样本量偏小",
                recommendation="在结果中明确不确定性",
            )
            service = AIReportService(
                retrieval_enabled=False,
                memory_store=memory,
            )
            response = {"choices": [{"message": {"content": "## 结果\n完成。"}}]}

            with patch(
                "ai.report_generator.urllib.request.urlopen",
                return_value=BytesIO(json.dumps(response).encode()),
            ) as urlopen:
                service.generate(record, self.provider, "secret-key")

            request = urlopen.call_args.args[0]
            prompt = json.loads(request.data)["messages"][1]["content"]

        self.assertIn("跨任务经验", prompt)
        self.assertIn("明确不确定性", prompt)


if __name__ == "__main__":
    unittest.main()
