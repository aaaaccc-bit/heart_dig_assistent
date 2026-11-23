"""Tests for the dependency-graph report orchestrator."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ai.orchestrator import (
    AgentTimeoutError,
    GraphNode,
    OrchestrationCancelled,
    ReportOrchestrator,
    ReportState,
)


class OrchestratorTest(unittest.TestCase):
    def test_parallel_fan_out_and_fan_in_use_artifacts(self) -> None:
        events: list[dict] = []

        def source(_state: ReportState):
            return {"value": 3}

        def left(state: ReportState):
            return state.get_artifact("source")["value"] + 1

        def right(state: ReportState):
            return state.get_artifact("source")["value"] + 2

        def join(state: ReportState):
            return state.get_artifact("left") + state.get_artifact("right")

        graph = ReportOrchestrator(
            [
                GraphNode("source", source),
                GraphNode("left", left, depends_on=("source",)),
                GraphNode("right", right, depends_on=("source",)),
                GraphNode("join", join, depends_on=("left", "right")),
            ],
            max_workers=2,
        )
        state = graph.run(ReportState(record=None, progress_callback=events.append))

        self.assertEqual(state.get_artifact("join"), 9)
        self.assertEqual(state.results["join"].output, 9)
        finished = [event for event in events if event.get("kind") == "agent_finished"]
        self.assertEqual(len(finished), 4)
        self.assertEqual([event["completed"] for event in finished], [1, 2, 3, 4])

    def test_retry_backoff_and_attempt_metadata(self) -> None:
        calls = 0
        events: list[dict] = []

        def flaky(_state: ReportState):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("temporary failure")
            return "ok"

        graph = ReportOrchestrator(
            [
                GraphNode(
                    "flaky",
                    flaky,
                    max_retries=2,
                    retry_backoff_s=0,
                    retry_backoff_multiplier=2,
                )
            ]
        )
        state = graph.run(ReportState(record=None, progress_callback=events.append))

        self.assertEqual(calls, 3)
        self.assertEqual(state.results["flaky"].attempts, 3)
        self.assertEqual(state.get_artifact("flaky"), "ok")
        retries = [event for event in events if event.get("kind") == "agent_retry"]
        self.assertEqual([event["attempt"] for event in retries], [2, 3])
        self.assertEqual(retries[0]["delay_s"], 0)

    def test_wave_and_graph_timings_are_reported(self) -> None:
        events: list[dict] = []
        graph = ReportOrchestrator(
            [
                GraphNode("source", lambda _state: "ok"),
                GraphNode("left", lambda _state: "left", depends_on=("source",)),
                GraphNode("right", lambda _state: "right", depends_on=("source",)),
            ],
            max_workers=2,
        )

        state = graph.run(ReportState(record=None, progress_callback=events.append))

        self.assertEqual(len(state.metadata["waves"]), 2)
        self.assertEqual(set(state.metadata["agent_timings_ms"]), {"source", "left", "right"})
        self.assertGreaterEqual(state.metadata["elapsed_ms"], 0)
        self.assertEqual(
            [event["kind"] for event in events if event["kind"] == "graph_finished"],
            ["graph_finished"],
        )

    def test_optional_failure_is_degraded_and_dependencies_continue(self) -> None:
        def fail(_state: ReportState):
            raise RuntimeError("retrieval unavailable")

        def downstream(state: ReportState):
            self.assertIsNone(state.get_artifact("optional"))
            return "fallback"

        graph = ReportOrchestrator(
            [
                GraphNode("optional", fail, optional=True),
                GraphNode("downstream", downstream, depends_on=("optional",)),
            ]
        )
        state = graph.run(ReportState(record=None))

        self.assertEqual(state.results["optional"].status, "degraded")
        self.assertEqual(state.results["downstream"].status, "completed")
        self.assertEqual(state.get_artifact("downstream"), "fallback")

    def test_timeout_is_recorded_as_node_failure(self) -> None:
        now = 0.0

        def slow(state: ReportState):
            nonlocal now
            now += 2.0
            state.check_cancelled()

        graph = ReportOrchestrator(
            [GraphNode("slow", slow, timeout_s=1.0, optional=True)]
        )
        with patch("ai.orchestrator.monotonic", side_effect=lambda: now):
            state = graph.run(ReportState(record=None))

        result = state.results["slow"]
        self.assertEqual(result.status, "degraded")
        self.assertEqual(result.exception_type, AgentTimeoutError.__name__)

    def test_cancellation_propagates(self) -> None:
        graph = ReportOrchestrator([GraphNode("cancelled", lambda _state: None)])

        with self.assertRaises(OrchestrationCancelled):
            graph.run(ReportState(record=None, cancel_requested=lambda: True))

    def test_graph_configuration_errors_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "名称重复"):
            ReportOrchestrator(
                [
                    GraphNode("same", lambda _state: None),
                    GraphNode("same", lambda _state: None),
                ]
            )

        with self.assertRaisesRegex(ValueError, "循环依赖"):
            ReportOrchestrator(
                [
                    GraphNode("a", lambda _state: None, depends_on=("b",)),
                    GraphNode("b", lambda _state: None, depends_on=("a",)),
                ]
            )


if __name__ == "__main__":
    unittest.main()
