from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from ai.memory import LongTermMemory


class LongTermMemoryTest(unittest.TestCase):
    def setUp(self):
        self.record = SimpleNamespace(
            title="报告",
            kind_label="心脏风险分析",
            dataset_name="测试集",
            trajectory_count=12,
            group_count=2,
            parameters={"回归分析": {}, "分类评估": {}},
        )

    def test_task_and_experience_round_trip(self):
        with TemporaryDirectory() as directory:
            memory = LongTermMemory(Path(directory) / "memory.sqlite3")
            fingerprint = memory.fingerprint(self.record)
            memory.remember_task(
                task_fingerprint=fingerprint,
                task_type=memory.task_type(self.record),
                summary="历史任务摘要",
                lessons=("检查置信区间",),
                metadata={"model": "base"},
            )
            memory.record_experience(
                task_type=memory.task_type(self.record),
                signal="样本量较小",
                recommendation="优先报告不确定性",
                evidence={"n": 12},
                confidence=0.9,
            )

            context = memory.format_context(
                task_fingerprint=fingerprint,
                task_type=memory.task_type(self.record),
            )

        self.assertIn("历史任务摘要", context)
        self.assertIn("优先报告不确定性", context)

    def test_fingerprint_is_stable_and_changes_with_task_shape(self):
        first = LongTermMemory.fingerprint(self.record)
        second = LongTermMemory.fingerprint(SimpleNamespace(**self.record.__dict__))
        changed = SimpleNamespace(**{**self.record.__dict__, "group_count": 3})
        self.assertEqual(first, second)
        self.assertNotEqual(first, LongTermMemory.fingerprint(changed))

    def test_unavailable_database_does_not_raise(self):
        with TemporaryDirectory() as directory:
            blocker = Path(directory) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            memory = LongTermMemory(blocker / "memory.sqlite3")
        self.assertFalse(memory.enabled)
        self.assertIsNone(memory.remember_task(
            task_fingerprint="x", task_type="t", summary="summary"
        ))


if __name__ == "__main__":
    unittest.main()
