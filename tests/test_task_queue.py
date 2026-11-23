from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import time
import unittest

from ai.task_queue import (
    DistributedTaskQueue,
    QueueOwnershipError,
    QueueWorker,
)


class DistributedTaskQueueTest(unittest.TestCase):
    def test_claim_ack_and_ownership(self):
        with TemporaryDirectory() as directory:
            queue = DistributedTaskQueue(Path(directory) / "tasks.sqlite3")
            task_id = queue.enqueue("demo", {"value": 1})
            task = queue.claim("worker-a")
            self.assertEqual(task.id, task_id)
            with self.assertRaises(QueueOwnershipError):
                queue.ack(task_id, "worker-b")
            queue.ack(task_id, "worker-a", result={"ok": True})
            completed = queue.get(task_id)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.result, {"ok": True})

    def test_fail_retries_then_marks_failed(self):
        with TemporaryDirectory() as directory:
            queue = DistributedTaskQueue(Path(directory) / "tasks.sqlite3")
            task_id = queue.enqueue("demo", {})
            first = queue.claim("worker-a")
            self.assertEqual(queue.fail(task_id, "worker-a", "temporary", retry_delay_s=0, max_attempts=2), "pending")
            second = queue.claim("worker-b")
            self.assertEqual(second.attempts, 2)
            self.assertEqual(queue.fail(task_id, "worker-b", "permanent", max_attempts=2), "failed")
            self.assertEqual(queue.get(task_id).status, "failed")

    def test_expired_lease_can_be_reclaimed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite3"
            queue = DistributedTaskQueue(path)
            task_id = queue.enqueue("demo", {})
            queue.claim("worker-a", lease_s=30)
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    "UPDATE task_queue SET lease_until = ? WHERE id = ?",
                    (time.time() - 1, task_id),
                )
                connection.commit()
            finally:
                connection.close()
            reclaimed = queue.claim("worker-b", lease_s=30)

        self.assertEqual(reclaimed.lease_owner, "worker-b")
        self.assertEqual(reclaimed.attempts, 2)

    def test_worker_records_handler_failure(self):
        with TemporaryDirectory() as directory:
            queue = DistributedTaskQueue(Path(directory) / "tasks.sqlite3")
            task_id = queue.enqueue("demo", {})
            worker = QueueWorker(queue, worker_id="worker", max_attempts=1)
            self.assertTrue(worker.run_once(lambda _task: (_ for _ in ()).throw(ValueError("bad"))))
            task = queue.get(task_id)

        self.assertEqual(task.status, "failed")
        self.assertIn("bad", task.last_error)


if __name__ == "__main__":
    unittest.main()
