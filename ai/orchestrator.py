"""Graph-driven multi-agent orchestration for AI analysis reports.

The implementation intentionally keeps the graph small and dependency-light. It
supports supervisor planning, pipeline dependencies, parallel fan-out/fan-in,
per-node retry with backoff, cooperative deadlines, cancellation, structured
artifacts and graceful degradation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic, sleep
from typing import Any, Callable, Iterable
from uuid import uuid4

from .artifacts import AgentArtifact


_NODE_DEADLINE: ContextVar[float | None] = ContextVar(
    "report_agent_deadline", default=None
)


class OrchestrationCancelled(RuntimeError):
    """Raised when the report generation is cancelled by the caller."""


class AgentTimeoutError(TimeoutError):
    """Raised when a node exceeds its cooperative execution deadline."""


@dataclass
class AgentResult:
    agent: str
    status: str = "completed"
    output: Any = None
    error: str | None = None
    attempts: int = 1
    elapsed_ms: float = 0.0
    exception_type: str | None = None


@dataclass
class ReportState:
    """Shared state passed between graph nodes."""

    record: Any
    instruction: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    knowledge_context: str = ""
    memory_context: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    report_text: str = ""
    results: dict[str, AgentResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_requested: Callable[[], bool] | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def check_cancelled(self) -> None:
        if self.cancel_requested is not None and self.cancel_requested():
            raise OrchestrationCancelled("AI 报告生成已取消。")
        deadline = _NODE_DEADLINE.get()
        if deadline is not None and monotonic() >= deadline:
            raise AgentTimeoutError("Agent 节点执行超时。")

    def report_progress(self, **event: Any) -> None:
        if self.progress_callback is not None:
            run_id = self.metadata.get("run_id")
            if run_id is not None:
                event.setdefault("run_id", run_id)
            with self._lock:
                self.progress_callback(event)

    def store_artifact(self, name: str, value: Any) -> None:
        """Store a node output for downstream nodes in a thread-safe way."""

        with self._lock:
            self.artifacts[name] = value

    def get_artifact(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self.artifacts.get(name, default)


@dataclass(frozen=True)
class GraphNode:
    name: str
    handler: Callable[[ReportState], Any]
    depends_on: tuple[str, ...] = ()
    max_retries: int = 0
    optional: bool = False
    retry_backoff_s: float = 0.0
    retry_backoff_multiplier: float = 2.0
    timeout_s: float | None = None
    condition: Callable[[ReportState], bool] | None = None


class ReportOrchestrator:
    """Execute a dependency graph with supervisor-style scheduling."""

    def __init__(self, nodes: Iterable[GraphNode], *, max_workers: int = 4):
        node_list = list(nodes)
        names: set[str] = set()
        duplicates: set[str] = set()
        for node in node_list:
            if node.name in names:
                duplicates.add(node.name)
            names.add(node.name)
        if duplicates:
            raise ValueError(f"Agent 名称重复：{', '.join(sorted(duplicates))}")
        self.nodes = {node.name: node for node in node_list}
        self.max_workers = max(1, max_workers)
        self._validate_graph()

    def run(self, state: ReportState) -> ReportState:
        run_started = monotonic()
        completed: set[str] = set()
        pending = set(self.nodes)
        total = len(self.nodes)
        wave_index = 0
        state.metadata.setdefault("run_id", uuid4().hex)
        state.metadata["total_agents"] = total
        state.metadata["completed_agents"] = 0
        state.metadata.setdefault("agent_timings_ms", {})
        state.metadata.setdefault("waves", [])
        condition_overrides: dict[str, bool] = {}
        state.metadata["_condition_overrides"] = condition_overrides

        def record_result(node: GraphNode, result: AgentResult) -> None:
            state.results[node.name] = result
            state.store_artifact(node.name, result.output)
            state.metadata["agent_timings_ms"][node.name] = result.elapsed_ms
            completed.add(node.name)
            pending.remove(node.name)
            state.metadata["completed_agents"] = len(completed)
            state.report_progress(
                kind="agent_finished",
                agent=node.name,
                status=result.status,
                error=result.error,
                attempts=result.attempts,
                elapsed_ms=result.elapsed_ms,
                exception_type=result.exception_type,
                completed=len(completed),
                total=total,
            )
            if result.status == "failed" and not node.optional:
                raise RuntimeError(f"Agent「{node.name}」执行失败：{result.error}")

        def drain_condition_skips() -> None:
            """Collapse chains of condition-false nodes before opening a wave."""

            while pending:
                ready = [
                    self.nodes[name]
                    for name in self.nodes
                    if name in pending
                    and all(dep in completed for dep in self.nodes[name].depends_on)
                ]
                skipped = []
                for node in ready:
                    if node.condition is None:
                        continue
                    if node.name in condition_overrides:
                        continue
                    try:
                        should_run = node.condition(state)
                    except Exception:
                        continue
                    if not should_run:
                        skipped.append(node)
                    else:
                        condition_overrides[node.name] = True
                if not skipped:
                    return
                for node in skipped:
                    state.check_cancelled()
                    started = monotonic()
                    state.report_progress(
                        kind="agent_started",
                        agent=node.name,
                        completed=len(completed),
                        total=total,
                    )
                    record_result(
                        node,
                        AgentResult(
                            agent=node.name,
                            status="skipped",
                            attempts=0,
                            elapsed_ms=(monotonic() - started) * 1000,
                        ),
                    )

        state.report_progress(
            kind="graph_started",
            run_id=state.metadata["run_id"],
            completed=0,
            total=total,
        )
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="report-agent"
        ) as pool:
            while pending:
                state.check_cancelled()
                drain_condition_skips()
                if not pending:
                    break
                # Iterate over insertion order so scheduling is reproducible.
                ready = [
                    self.nodes[name]
                    for name in self.nodes
                    if name in pending
                    and all(dep in completed for dep in self.nodes[name].depends_on)
                ]
                if not ready:
                    unresolved = ", ".join(sorted(pending))
                    raise RuntimeError(f"报告 Agent 图存在未满足依赖：{unresolved}")

                # Supervisor fans out independent workers and joins their results.
                wave_index += 1
                wave_started = monotonic()
                ready_names = [node.name for node in ready]
                state.report_progress(
                    kind="wave_started",
                    wave=wave_index,
                    agents=ready_names,
                    completed=len(completed),
                    total=total,
                )
                futures = {
                    pool.submit(self._execute_node, node, state, len(completed)): node
                    for node in ready
                }
                for future in as_completed(futures):
                    node = futures[future]
                    try:
                        result = future.result()
                    except BaseException:
                        for other in futures:
                            if other is not future:
                                other.cancel()
                        raise
                    record_result(node, result)
                wave_elapsed_ms = (monotonic() - wave_started) * 1000
                state.metadata["waves"].append(
                    {
                        "wave": wave_index,
                        "agents": ready_names,
                        "elapsed_ms": wave_elapsed_ms,
                    }
                )
                state.report_progress(
                    kind="wave_finished",
                    wave=wave_index,
                    agents=ready_names,
                    elapsed_ms=wave_elapsed_ms,
                    completed=len(completed),
                    total=total,
                )
        state.metadata.pop("_condition_overrides", None)
        elapsed_ms = (monotonic() - run_started) * 1000
        state.metadata["elapsed_ms"] = elapsed_ms
        state.report_progress(
            kind="graph_finished",
            elapsed_ms=elapsed_ms,
            completed=total,
            total=total,
        )
        return state

    def _execute_node(
        self, node: GraphNode, state: ReportState, completed: int = 0
    ) -> AgentResult:
        started = monotonic()
        attempts = 0
        state.report_progress(
            kind="agent_started",
            agent=node.name,
            completed=completed,
            total=state.metadata.get("total_agents", len(self.nodes)),
        )
        while True:
            attempts += 1
            deadline_token = _NODE_DEADLINE.set(
                monotonic() + node.timeout_s if node.timeout_s is not None else None
            )
            retry_delay: float | None = None
            try:
                state.check_cancelled()
                if node.condition is not None:
                    overrides = state.metadata.get("_condition_overrides", {})
                    if node.name in overrides:
                        should_run = overrides.pop(node.name)
                    else:
                        should_run = node.condition(state)
                    if not should_run:
                        return AgentResult(
                            agent=node.name,
                            status="skipped",
                            attempts=0,
                            elapsed_ms=(monotonic() - started) * 1000,
                        )
                output = node.handler(state)
                # The final check also catches handlers that return after their
                # deadline without calling state.check_cancelled() internally.
                state.check_cancelled()
                result = AgentResult(
                    agent=node.name,
                    output=output,
                    attempts=attempts,
                    elapsed_ms=(monotonic() - started) * 1000,
                )
                return result
            except OrchestrationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - graph boundary records failures
                if attempts > node.max_retries:
                    result = AgentResult(
                        agent=node.name,
                        status="degraded" if node.optional else "failed",
                        error=str(exc),
                        attempts=attempts,
                        elapsed_ms=(monotonic() - started) * 1000,
                        exception_type=type(exc).__name__,
                    )
                    return result
                delay = node.retry_backoff_s * (
                    node.retry_backoff_multiplier ** (attempts - 1)
                )
                state.report_progress(
                    kind="agent_retry",
                    agent=node.name,
                    attempt=attempts + 1,
                    delay_s=delay,
                    error=str(exc),
                    completed=completed,
                    total=state.metadata.get("total_agents", len(self.nodes)),
                )
                retry_delay = delay
            finally:
                _NODE_DEADLINE.reset(deadline_token)
            if retry_delay is not None:
                self._sleep_with_cancellation(retry_delay, state)

    @staticmethod
    def _sleep_with_cancellation(delay: float, state: ReportState) -> None:
        """Sleep between retries while keeping cancellation responsive."""

        remaining = max(0.0, delay)
        while remaining > 0:
            state.check_cancelled()
            interval = min(0.05, remaining)
            sleep(interval)
            remaining -= interval

    def _validate_graph(self) -> None:
        for node in self.nodes.values():
            if not node.name.strip():
                raise ValueError("Agent 名称不能为空")
            if not callable(node.handler):
                raise ValueError(f"Agent「{node.name}」handler 必须可调用")
            if node.max_retries < 0:
                raise ValueError(f"Agent「{node.name}」max_retries 不能为负数")
            if node.retry_backoff_s < 0:
                raise ValueError(f"Agent「{node.name}」retry_backoff_s 不能为负数")
            if node.retry_backoff_multiplier < 1:
                raise ValueError(
                    f"Agent「{node.name}」retry_backoff_multiplier 不能小于 1"
                )
            if node.timeout_s is not None and node.timeout_s <= 0:
                raise ValueError(f"Agent「{node.name}」timeout_s 必须大于 0")
            missing = [dep for dep in node.depends_on if dep not in self.nodes]
            if missing:
                raise ValueError(f"Agent「{node.name}」依赖不存在：{', '.join(missing)}")
        # A lightweight cycle check keeps configuration errors readable.
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Agent 图存在循环依赖：{name}")
            if name in visited:
                return
            visiting.add(name)
            for dep in self.nodes[name].depends_on:
                visit(dep)
            visiting.remove(name)
            visited.add(name)

        for name in self.nodes:
            visit(name)


__all__ = [
    "AgentArtifact",
    "AgentResult",
    "AgentTimeoutError",
    "GraphNode",
    "OrchestrationCancelled",
    "ReportOrchestrator",
    "ReportState",
]
