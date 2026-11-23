"""API-compatible AI report generation from completed analysis records."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime
from http.client import HTTPException
from types import SimpleNamespace
from typing import Any, Callable
from uuid import uuid4

from .artifacts import AgentArtifact
from .memory import LongTermMemory, get_shared_memory
from .providers import ProviderConfig
from .knowledge_base import ChineseKnowledgeBase, get_shared_knowledge_base
from .report_guard import finalize_report
from .review_agents import (
    citation_safety_review,
    has_statistical_sections,
    needs_statistical_review,
    review_draft,
    statistical_review,
)
from .security_settings import load_security_settings
from .structured import StructuredModelClient, StructuredOutputCancelled
from .tools import ToolRegistry
from .task_queue import DistributedTaskQueue, get_shared_task_queue
from .entities import expand_query
from .orchestrator import (
    GraphNode,
    OrchestrationCancelled,
    ReportOrchestrator,
    ReportState,
)

_READ_CHUNK = 4096
_MAX_PROMPT_LIST_ITEMS = 20
_MAX_PROMPT_TEXT_CHARS = 1600


class AIReportError(RuntimeError):
    """Raised when a provider request cannot produce a report."""


class AIReportCancelled(AIReportError):
    """Raised when the caller interrupts a report request."""


class AIReportService:
    def __init__(
        self,
        timeout: float = 90,
        *,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        knowledge_base: ChineseKnowledgeBase | None = None,
        retrieval_enabled: bool = True,
        retrieval_top_k: int = 4,
        tool_registry: ToolRegistry | None = None,
        memory_enabled: bool = True,
        memory_store: LongTermMemory | None = None,
        task_queue: DistributedTaskQueue | None = None,
    ):
        self.timeout = timeout
        self.connect_timeout = (
            min(timeout, 15.0) if connect_timeout is None else connect_timeout
        )
        self.read_timeout = timeout if read_timeout is None else read_timeout
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("连接和读取超时必须大于 0")
        # 知识库（尤其是向量模型）可能触发较慢的本地加载或模型下载。
        # 不要在打开 AI 报告对话框的 UI 线程中初始化，首次生成时再由
        # AIReportWorker 所在线程懒加载。
        self.knowledge_base = knowledge_base
        self.retrieval_enabled = retrieval_enabled
        self.retrieval_top_k = retrieval_top_k
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_enabled = memory_enabled
        self.memory_store = memory_store
        self.task_queue = task_queue

    def generate_structured(
        self,
        provider: ProviderConfig,
        api_key: str,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: type = AgentArtifact,
        model: str | None = None,
        tool_names: list[str] | tuple[str, ...] | None = None,
        json_mode: bool = True,
        cancel_requested: Callable[[], bool] | None = None,
    ):
        """Generate a validated JSON artifact for a specialist agent."""

        tools = self.tool_registry.schemas(tool_names)
        client = StructuredModelClient(timeout=self.timeout)
        try:
            return client.generate(
                provider=provider,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_schema=output_schema,
                model=model,
                tools=tools,
                json_mode=json_mode,
                tool_registry=self.tool_registry,
                cancel_requested=cancel_requested,
            )
        except StructuredOutputCancelled as exc:
            raise AIReportCancelled(str(exc)) from exc

    def build_prompt(
        self,
        record,
        instruction: str = "",
        *,
        knowledge_context_override: str | None = None,
    ) -> str:
        metrics = [{"label": m.label, "value": m.value, "note": m.note} for m in record.metrics]
        highlights = [{"title": h.title, "detail": h.detail} for h in record.highlights]
        payload = {
            "title": record.title,
            "kind": record.kind_label,
            "dataset": record.dataset_name,
            "created_at": record.created_at.isoformat(),
            "trajectory_count": record.trajectory_count,
            "group_count": record.group_count,
            "metrics": metrics,
            "highlights": highlights,
            "parameters": self._compact_prompt_value(record.parameters),
        }
        extra = instruction.strip() or (
            "请生成一份中文心脏风险数据分析报告，包含摘要、方法、关键结果、解释、局限性和后续建议。"
            "若结构化结果中包含回归或随机森林分析，请对比模型表现；包含风险分档或聚类时，"
            "说明分档/分群依据并指出分群质量警示；包含分类评估时对比各模型在同一阈值下的表现；"
            "包含特征洞察时解释特征贡献、方向与共线性。不要虚构诊疗结论。"
        )
        format_instruction = (
            "\n\n输出格式要求：只输出 Markdown 正文，不要包裹 ```markdown 代码围栏。"
            "使用清晰的标题层级（#、##）、列表、表格和加粗强调来组织内容；"
            "数据和结论必须严格来自给定结果，无法确定的内容请明确标注。"
            "涉及参考知识时，在对应句末使用 [KB-1] 这类引用编号，并在文末增加‘参考知识’小节。"
        )
        knowledge_context = knowledge_context_override or ""
        if self.retrieval_enabled and knowledge_context_override is None:
            self.knowledge_base = self._get_knowledge_base()
            query = self._build_retrieval_query(record, instruction)
            try:
                knowledge_context = self.knowledge_base.format_context(
                    query, top_k=self.retrieval_top_k
                )
            except Exception:
                # 检索是增强能力，不能阻断原有报告生成链路。
                knowledge_context = ""
        knowledge_block = (
            "\n\n参考知识（仅用于概念解释，不是当前实验数据）：\n"
            f"{knowledge_context}"
            if knowledge_context
            else ""
        )
        return (
            f"{extra}{format_instruction}{knowledge_block}"
            "\n\n以下是结构化分析结果（当前实验的唯一数据来源）：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _get_knowledge_base(self) -> ChineseKnowledgeBase:
        if self.knowledge_base is None:
            self.knowledge_base = get_shared_knowledge_base()
        return self.knowledge_base

    def _get_memory_store(self) -> LongTermMemory | None:
        if not self.memory_enabled:
            return None
        if self.memory_store is None:
            self.memory_store = get_shared_memory()
        return self.memory_store

    @classmethod
    def _compact_prompt_value(cls, value: Any, *, depth: int = 0) -> Any:
        if depth >= 8:
            return "[嵌套内容已省略]"
        if isinstance(value, Mapping):
            return {
                str(key): cls._compact_prompt_value(item, depth=depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            items = [
                cls._compact_prompt_value(item, depth=depth + 1)
                for item in value[:_MAX_PROMPT_LIST_ITEMS]
            ]
            if len(value) > _MAX_PROMPT_LIST_ITEMS:
                items.append({"_truncated_items": len(value) - _MAX_PROMPT_LIST_ITEMS})
            return items
        if isinstance(value, str) and len(value) > _MAX_PROMPT_TEXT_CHARS:
            return value[:_MAX_PROMPT_TEXT_CHARS] + "…[文本已截断]"
        return value

    @classmethod
    def _build_retrieval_query(cls, record, instruction: str) -> str:
        parts = [record.title, record.kind_label, instruction]
        for metric in record.metrics:
            parts.extend((metric.label, metric.value, metric.note))
        for highlight in record.highlights:
            parts.extend((highlight.title, highlight.detail))
        for key, value in record.parameters.items():
            compact = cls._compact_prompt_value(value)
            parts.extend((str(key), str(compact)))
        # Group summaries and comparisons contain useful terms, but cap their
        # contribution to avoid turning a retrieval query into the full report.
        for frame_name in ("group_summary", "comparisons"):
            frame = getattr(record, frame_name, None)
            if frame is not None and not getattr(frame, "empty", True):
                parts.extend(str(value) for value in frame.head(6).to_numpy().ravel())
        return expand_query(" ".join(part for part in parts if part))

    def generate(
        self,
        record,
        provider: ProviderConfig,
        api_key: str,
        model: str | None = None,
        instruction: str = "",
        *,
        cancel_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        safety_mode: str | None = None,
        safety_strength: int | None = None,
        task_id: str | None = None,
    ) -> str:
        self._raise_if_cancelled(cancel_requested)
        security = load_security_settings()
        safety_mode = safety_mode or str(security["mode"])
        safety_strength = security["strength"] if safety_strength is None else safety_strength
        key = (api_key or "").strip()
        if not key:
            raise AIReportError("请先填写 API Key，或在环境变量中配置对应密钥。")
        state = None
        try:
            state = self._run_orchestration(
                record,
                instruction,
                provider=provider,
                api_key=key,
                selected_model=model,
                safety_mode=safety_mode,
                safety_strength=safety_strength,
                task_id=task_id,
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
            )
        except OrchestrationCancelled as exc:
            if state is not None:
                self._persist_memory(state, provider=provider, success=False, error=str(exc))
            raise AIReportCancelled(str(exc)) from exc
        except RuntimeError as exc:
            if state is not None:
                self._persist_memory(state, provider=provider, success=False, error=str(exc))
            raise AIReportError(str(exc)) from exc
        self._persist_memory(state, provider=provider, success=True)
        return state.report_text

    def _writer_prompt(self, state: ReportState) -> str:
        prompt = self.build_prompt(
            state.record,
            state.instruction,
            knowledge_context_override=state.knowledge_context,
        )
        if state.evidence:
            prompt += (
                "\n\n多 Agent 证据汇总（仅可用于组织表达，不能替代结构化结果）：\n"
                + json.dumps(
                    self._compact_evidence(state.evidence),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        if state.memory_context:
            prompt += "\n\n" + state.memory_context
        review_outputs = self._review_outputs(state)
        if review_outputs:
            prompt += (
                "\n\n审查 Agent 结果（必须遵守其中的警告和后续动作）：\n"
                + json.dumps(review_outputs, ensure_ascii=False, indent=2)
            )
        return prompt

    @staticmethod
    def _compact_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = ("agent", "claim", "value", "note", "warnings")
        compact = []
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            compact.append({key: item[key] for key in fields if key in item})
        return compact

    def _request_report(
        self,
        provider: ProviderConfig,
        key: str,
        model: str,
        prompt: str,
        state: ReportState,
        *,
        agent: str | None = None,
    ) -> str:
        state.check_cancelled()
        if provider.protocol == "anthropic":
            payload = {
                "model": model,
                "max_tokens": 4096,
                "temperature": 0.2,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            }
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            url = provider.base_url.rstrip("/") + "/messages"
        else:
            payload = {
                "model": model,
                "temperature": 0.2,
                "stream": True,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是严谨的心脏风险数据分析助手，只能根据提供的统计结果进行解释，不得编造诊疗结论。",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
            url = provider.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.connect_timeout) as response:
                self._set_read_timeout(response, self.read_timeout)
                streamed, body = self._read_response(
                    response,
                    state,
                    provider.protocol,
                    agent=agent,
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise AIReportError(f"{provider.name} API 请求失败（HTTP {exc.code}）：{detail}") from exc
        except TimeoutError as exc:
            raise AIReportError(f"{provider.name} API 请求超时：{exc}") from exc
        except (urllib.error.URLError, OSError, HTTPException) as exc:
            raise AIReportError(f"无法连接 {provider.name} API：{exc}") from exc
        state.check_cancelled()
        if streamed:
            return self._normalize_markdown(body)
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AIReportError("API 返回内容不是有效 JSON。") from exc
        return self._normalize_markdown(self._extract_text(data, provider.protocol))

    def _run_orchestration(
        self,
        record,
        instruction: str,
        *,
        provider: ProviderConfig,
        api_key: str,
        selected_model: str | None,
        safety_mode: str,
        safety_strength: int,
        task_id: str | None = None,
        cancel_requested=None,
        progress_callback=None,
    ):
        state = ReportState(
            record=record,
            instruction=instruction,
            cancel_requested=cancel_requested,
            progress_callback=progress_callback,
        )
        if task_id:
            state.metadata["task_id"] = task_id

        def validate(current):
            if not isinstance(record.parameters, Mapping):
                raise ValueError("报告参数必须是对象")
            current.payload = {
                "title": record.title,
                "kind": record.kind_label,
                "metrics": [m.label for m in record.metrics],
                "highlights": [h.title for h in record.highlights],
            }
            if not record.title:
                raise ValueError("报告标题为空")
            current.metadata["task_fingerprint"] = LongTermMemory.fingerprint(record)
            current.metadata["task_type"] = LongTermMemory.task_type(record)

        def retrieve(current):
            self.knowledge_base = self._get_knowledge_base()
            query = self._build_retrieval_query(record, instruction)
            current.knowledge_context = self.knowledge_base.format_context(
                query, top_k=self.retrieval_top_k
            )
            status = self.knowledge_base.retrieval_status
            degradation = []
            if status.get("vector_backend") == "hash_fallback":
                degradation.append("BGE 向量不可用，当前使用哈希向量回退")
            if status.get("reranker_backend") == "LocalReranker_fallback":
                degradation.append("BGE 重排器不可用，当前使用本地规则重排")
            if degradation:
                current.knowledge_context = (
                    "[检索降级提示] " + "；".join(degradation) + "；"
                    "请降低对语义召回结果的信任。\n\n" + current.knowledge_context
                )

        def recall_experience(current):
            store = self._get_memory_store()
            if store is None or not getattr(store, "enabled", True):
                return ""
            try:
                current.memory_context = store.format_context(
                    task_fingerprint=current.metadata["task_fingerprint"],
                    task_type=current.metadata["task_type"],
                )
            except Exception as exc:  # noqa: BLE001 - memory must not block reports
                current.metadata["memory_error"] = str(exc)
                current.memory_context = ""
            return current.memory_context

        def metric_worker(current, metric_name):
            current.check_cancelled()
            matches = [m for m in record.metrics if metric_name in m.label]
            if not matches:
                return
            return AgentArtifact(
                agent=f"{metric_name}分析 Agent",
                claims=[
                    {
                        "claim": matches[0].label,
                        "value": matches[0].value,
                        "note": matches[0].note,
                    }
                ],
                evidence_refs=[f"metrics[{record.metrics.index(matches[0])}]"],
            )

        def aggregate(current):
            evidence = []
            for node in metric_nodes:
                artifact = current.get_artifact(node.name)
                if isinstance(artifact, AgentArtifact):
                    item = artifact.to_dict()
                    if len(artifact.claims) == 1:
                        item.update(artifact.claims[0])
                    evidence.append(item)
                elif isinstance(artifact, dict):
                    evidence.append(artifact)
            current.evidence = sorted(evidence, key=lambda item: item["agent"])

        def statistical_review_worker(current):
            return statistical_review(current.record)

        def citation_safety_review_worker(current):
            return citation_safety_review(
                instruction=instruction,
                knowledge_context=current.knowledge_context,
                evidence=current.evidence,
            )

        advanced = (
            isinstance(record.parameters, Mapping)
            and has_statistical_sections(record.parameters)
        )
        metric_names = tuple(
            name
            for name in ("阳性率", "年龄", "血压", "心率", "胆固醇", "比较")
            if any(name in metric.label for metric in record.metrics)
        )
        metric_nodes = [
            GraphNode(
                f"{name}分析 Agent",
                lambda current, metric_name=name: metric_worker(current, metric_name),
                depends_on=("数据校验 Agent",),
            )
            for name in metric_names
        ]
        graph = ReportOrchestrator(
            [
                GraphNode("数据校验 Agent", validate, max_retries=0),
                GraphNode(
                    "知识检索 Agent",
                    retrieve,
                    depends_on=("数据校验 Agent",),
                    optional=True,
                    condition=lambda _state: self.retrieval_enabled,
                ),
                GraphNode(
                    "经验检索 Agent",
                    recall_experience,
                    depends_on=("数据校验 Agent",),
                    optional=True,
                    condition=lambda _state: self.memory_enabled,
                ),
                GraphNode(
                    "证据汇聚 Agent",
                    aggregate,
                    depends_on=("知识检索 Agent",) + tuple(node.name for node in metric_nodes),
                ),
                GraphNode(
                    "统计审查 Agent",
                    statistical_review_worker,
                    depends_on=("证据汇聚 Agent",),
                    condition=lambda current: needs_statistical_review(current.record.parameters),
                ),
                GraphNode(
                    "引用与安全审查 Agent",
                    citation_safety_review_worker,
                    depends_on=("证据汇聚 Agent", "知识检索 Agent"),
                ),
                *metric_nodes,
                *self._writing_nodes(
                    provider, api_key, selected_model, advanced, safety_mode, safety_strength
                ),
            ]
        )
        return graph.run(state)

    def _writing_nodes(
        self,
        provider: ProviderConfig,
        api_key: str,
        selected_model: str | None,
        advanced: bool,
        safety_mode: str,
        safety_strength: int,
    ) -> list[GraphNode]:
        draft = "报告写作 Agent"
        draft_review = "初稿审查 Agent"
        plan = "修订规划 Agent"
        rewrite = "报告修订 Agent"
        rewrite_review = "修订复审 Agent"

        def write(current, *, revision=False):
            name = rewrite if revision else draft
            model = provider.report_model(selected_model, advanced=advanced and not revision)
            current.metadata.setdefault("models_used", []).append(model)
            current.report_progress(
                kind="model_selected", agent=name, model=model,
                completed=current.metadata["completed_agents"],
                total=current.metadata["total_agents"],
            )
            prompt = self._writer_prompt(current)
            if revision:
                actions = "\n".join(f"- {action}" for action in current.get_artifact(plan))
                prompt += (
                    "\n\n请根据以下审查意见重写完整报告，保持结构化数据原值。"
                    f"\n{actions}\n\n待修订草稿\n{current.get_artifact(draft)}"
                )
            current.report_progress(
                kind="prompt_built",
                agent=name,
                prompt_chars=len(prompt),
            )
            return self._request_report(
                provider,
                api_key,
                model,
                prompt,
                current,
                agent=name,
            )

        def audit(current, name):
            return review_draft(
                current.get_artifact(name),
                current.knowledge_context,
                safety_mode=safety_mode,
                safety_strength=safety_strength,
            )

        def finalize(current):
            name = draft
            review = current.get_artifact(draft_review)
            revised_review = current.get_artifact(rewrite_review)
            notes = self._audit_notes(current)
            if revised_review is not None and revised_review.severity < review.severity:
                name, review = rewrite, revised_review
            if current.results[rewrite].status == "degraded":
                notes.append("自动修订请求失败，已保留初稿的审查结果。")
            if any(review.severity):
                notes.append("最终稿仍有引用或安全审查问题，需要人工复核。")
            current.report_text = finalize_report(
                current.get_artifact(name),
                current.knowledge_context,
                audit_notes=notes,
                risk_categories=review.risk_categories,
                safety_mode=safety_mode,
                safety_strength=safety_strength,
            )

        return [
            GraphNode(
                draft,
                write,
                depends_on=(
                    "统计审查 Agent",
                    "引用与安全审查 Agent",
                    "经验检索 Agent",
                ),
            ),
            GraphNode(draft_review, lambda state: audit(state, draft), depends_on=(draft,)),
            GraphNode(
                plan,
                lambda state: state.get_artifact(draft_review).revision_actions(),
                depends_on=(draft_review,),
                condition=lambda state: state.get_artifact(draft_review).requires_revision,
            ),
            GraphNode(
                rewrite,
                lambda state: write(state, revision=True),
                depends_on=(plan,),
                condition=lambda state: state.get_artifact(plan) is not None,
                optional=True,
            ),
            GraphNode(
                rewrite_review,
                lambda state: audit(state, rewrite),
                depends_on=(rewrite,),
                condition=lambda state: state.results[rewrite].status == "completed",
            ),
            GraphNode("报告定稿", finalize, depends_on=(rewrite_review,)),
        ]

    @staticmethod
    def _review_outputs(state: ReportState) -> list[dict]:
        outputs = []
        for name in ("统计审查 Agent", "引用与安全审查 Agent"):
            artifact = state.get_artifact(name)
            if isinstance(artifact, AgentArtifact):
                outputs.append(
                    {
                        "agent": artifact.agent,
                        "claims": artifact.claims[:8],
                        "warnings": artifact.warnings[:8],
                        "next_actions": artifact.next_actions[:8],
                    }
                )
        return outputs

    @staticmethod
    def _audit_notes(state: ReportState) -> list[str]:
        notes = []
        for name in ("统计审查 Agent", "引用与安全审查 Agent"):
            artifact = state.get_artifact(name)
            if isinstance(artifact, AgentArtifact):
                notes.extend(f"{name}：{warning}" for warning in artifact.warnings)
        return notes

    def _persist_memory(
        self,
        state: ReportState | None,
        *,
        provider: ProviderConfig,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Persist a bounded task summary without storing report contents or keys."""

        if state is None:
            return
        store = self._get_memory_store()
        if store is None or not getattr(store, "enabled", True):
            return
        record = state.record
        lessons: list[str] = []
        for name in ("统计审查 Agent", "引用与安全审查 Agent"):
            artifact = state.get_artifact(name)
            if not isinstance(artifact, AgentArtifact):
                continue
            lessons.extend(str(item) for item in artifact.warnings[:4])
            lessons.extend(str(item) for item in artifact.next_actions[:4])
        if error:
            lessons.append(f"本次任务未完成：{error[:300]}")
        if success and not lessons:
            lessons.append("报告生成与审查流程完成，未记录新增审查警告。")
        summary = self._task_summary(record)
        metadata = {
            "provider": provider.name,
            "models": list(dict.fromkeys(state.metadata.get("models_used", []))),
            "elapsed_ms": round(float(state.metadata.get("elapsed_ms", 0.0)), 2),
            "wave_count": len(state.metadata.get("waves", [])),
            "agent_statuses": {
                name: result.status for name, result in state.results.items()
            },
        }
        if state.metadata.get("task_id"):
            metadata["task_id"] = state.metadata["task_id"]
        if state.metadata.get("memory_error"):
            metadata["memory_error"] = str(state.metadata["memory_error"])
        try:
            store.remember_task(
                task_fingerprint=state.metadata.get(
                    "task_fingerprint", LongTermMemory.fingerprint(record)
                ),
                task_type=state.metadata.get("task_type", LongTermMemory.task_type(record)),
                summary=summary,
                lessons=tuple(dict.fromkeys(lessons))[:12],
                metadata=metadata,
                success=success,
            )
            for lesson in tuple(dict.fromkeys(lessons))[:6]:
                store.record_experience(
                    task_type=state.metadata.get(
                        "task_type", LongTermMemory.task_type(record)
                    ),
                    signal="历史任务审查反馈",
                    recommendation=lesson,
                    evidence={"provider": provider.name, "success": success},
                    confidence=0.75 if success else 0.55,
                )
        except Exception as exc:  # noqa: BLE001 - memory is an optional side effect
            state.metadata["memory_error"] = str(exc)

    @staticmethod
    def _task_summary(record: Any) -> str:
        metrics = [str(getattr(item, "label", "")) for item in getattr(record, "metrics", ())]
        sections = list(getattr(record, "parameters", {}) or {})
        summary = (
            f"{getattr(record, 'title', '')}；任务类型：{getattr(record, 'kind_label', '')}；"
            f"数据集：{getattr(record, 'dataset_name', '')}；"
            f"样本量：{getattr(record, 'trajectory_count', '')}；"
            f"分析节：{', '.join(str(item) for item in sections[:12])}"
        )
        if metrics:
            summary += f"；关键指标：{', '.join(metrics[:8])}"
        return summary[:2000]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Mapping):
            return {str(key): AIReportService._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [AIReportService._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _serialize_record(cls, record: Any) -> dict[str, Any]:
        return {
            "title": str(getattr(record, "title", "")),
            "kind_label": str(getattr(record, "kind_label", "")),
            "dataset_name": str(getattr(record, "dataset_name", "")),
            "created_at": cls._json_safe(getattr(record, "created_at", "")),
            "trajectory_count": int(getattr(record, "trajectory_count", 0) or 0),
            "group_count": int(getattr(record, "group_count", 0) or 0),
            "metrics": [
                {
                    "label": str(getattr(item, "label", "")),
                    "value": str(getattr(item, "value", "")),
                    "note": str(getattr(item, "note", "")),
                }
                for item in getattr(record, "metrics", ())
            ],
            "highlights": [
                {
                    "title": str(getattr(item, "title", "")),
                    "detail": str(getattr(item, "detail", "")),
                }
                for item in getattr(record, "highlights", ())
            ],
            "parameters": cls._json_safe(getattr(record, "parameters", {})),
        }

    @staticmethod
    def _deserialize_record(payload: Mapping[str, Any]) -> Any:
        created_at = payload.get("created_at", "")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        return SimpleNamespace(
            title=str(payload.get("title", "")),
            kind_label=str(payload.get("kind_label", "")),
            dataset_name=str(payload.get("dataset_name", "")),
            created_at=created_at,
            trajectory_count=int(payload.get("trajectory_count", 0) or 0),
            group_count=int(payload.get("group_count", 0) or 0),
            metrics=tuple(SimpleNamespace(**item) for item in payload.get("metrics", [])),
            highlights=tuple(SimpleNamespace(**item) for item in payload.get("highlights", [])),
            parameters=dict(payload.get("parameters", {})),
        )

    @staticmethod
    def _serialize_provider(provider: ProviderConfig) -> dict[str, Any]:
        return {
            "name": provider.name,
            "protocol": provider.protocol,
            "base_url": provider.base_url,
            "default_model": provider.default_model,
            "models": list(provider.models),
            "analysis_model": provider.analysis_model,
        }

    @staticmethod
    def _deserialize_provider(payload: Mapping[str, Any]) -> ProviderConfig:
        return ProviderConfig(
            str(payload.get("name", "")),
            str(payload.get("protocol", "openai")),
            str(payload.get("base_url", "")),
            str(payload.get("default_model", "")),
            tuple(str(item) for item in payload.get("models", [])),
            payload.get("analysis_model"),
        )

    def enqueue_report(
        self,
        record: Any,
        provider: ProviderConfig,
        *,
        model: str | None = None,
        instruction: str = "",
        safety_mode: str | None = None,
        safety_strength: int | None = None,
        priority: int = 0,
        delay_s: float = 0.0,
        task_id: str | None = None,
    ) -> str:
        """Enqueue a report request; the API key is deliberately excluded."""

        queue = self.task_queue or get_shared_task_queue()
        security = load_security_settings()
        payload = {
            "record": self._serialize_record(record),
            "provider": self._serialize_provider(provider),
            "model": model,
            "instruction": instruction,
            "safety_mode": safety_mode or str(security["mode"]),
            "safety_strength": security["strength"] if safety_strength is None else safety_strength,
        }
        return queue.enqueue(
            "ai_report",
            payload,
            priority=priority,
            delay_s=delay_s,
            task_id=task_id,
        )

    def process_next_queued_report(
        self,
        api_key_resolver: Callable[[str], str] | None = None,
        *,
        worker_id: str | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        lease_s: float = 300.0,
        max_attempts: int = 3,
    ) -> str | None:
        """Claim and execute one queued report, returning its generated text."""

        queue = self.task_queue or get_shared_task_queue()
        owner = worker_id or f"report-worker-{os.getpid()}-{uuid4().hex}"
        task = queue.claim(owner, lease_s=lease_s)
        if task is None:
            return None
        try:
            if task.kind != "ai_report":
                raise AIReportError(f"未知的报告队列任务类型：{task.kind}")
            payload = task.payload
            provider = self._deserialize_provider(payload["provider"])
            resolver = api_key_resolver or self.api_key_from_env
            api_key = resolver(provider.name)
            report = self.generate(
                self._deserialize_record(payload["record"]),
                provider,
                api_key,
                payload.get("model"),
                payload.get("instruction", ""),
                cancel_requested=cancel_requested,
                progress_callback=progress_callback,
                safety_mode=payload.get("safety_mode"),
                safety_strength=payload.get("safety_strength"),
                task_id=task.id,
            )
            queue.ack(
                task.id,
                owner,
                result={"report": report, "provider": provider.name},
            )
            return report
        except Exception as exc:  # noqa: BLE001 - queue boundary records failures
            queue.fail(task.id, owner, str(exc), max_attempts=max_attempts)
            if isinstance(exc, AIReportError):
                raise
            raise AIReportError(f"队列报告任务执行失败：{exc}") from exc

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        """Remove accidental outer Markdown code fences from provider output."""
        value = text.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if len(lines) >= 2:
                value = "\n".join(lines[1:-1]).strip()
        return value

    @staticmethod
    def _set_read_timeout(response, timeout: float) -> None:
        try:
            socket = response.fp.raw._sock
        except AttributeError:
            return
        if socket is None:
            return
        try:
            socket.settimeout(timeout)
        except OSError:
            return

    @staticmethod
    def _read_response(
        response,
        state: ReportState,
        protocol: str,
        *,
        agent: str | None,
    ) -> tuple[bool, bytes | str]:
        headers = getattr(response, "headers", None)
        content_type = str(headers.get("Content-Type", "")) if headers else ""
        if "text/event-stream" not in content_type.lower():
            return False, AIReportService._read_body(response, state)
        return True, AIReportService._read_sse(response, state, protocol, agent=agent)

    @staticmethod
    def _read_sse(
        response,
        state: ReportState,
        protocol: str,
        *,
        agent: str | None,
    ) -> str:
        read = getattr(response, "read1", None) or response.read
        buffer = bytearray()
        parts: list[str] = []
        received_chars = 0

        def consume_line(line: bytes) -> None:
            nonlocal received_chars
            if not line.startswith(b"data:"):
                return
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                return
            try:
                event = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            fragment = AIReportService._stream_fragment(event, protocol)
            if not fragment:
                return
            parts.append(fragment)
            received_chars += len(fragment)
            state.report_progress(
                kind="model_chunk",
                agent=agent,
                chars=received_chars,
            )

        while True:
            state.check_cancelled()
            chunk = read(_READ_CHUNK)
            if not chunk:
                break
            buffer.extend(chunk)
            while b"\n" in buffer:
                line, _, remainder = buffer.partition(b"\n")
                buffer = bytearray(remainder)
                consume_line(line.rstrip(b"\r"))
        if buffer:
            consume_line(bytes(buffer).rstrip(b"\r"))
        text = "".join(parts).strip()
        if not text:
            raise AIReportError("API 流式响应未包含报告文本。")
        return text

    @staticmethod
    def _stream_fragment(data: dict[str, Any], protocol: str) -> str:
        if protocol == "anthropic":
            if data.get("type") != "content_block_delta":
                return ""
            delta = data.get("delta") or {}
            return str(delta.get("text") or "")
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta") or {}
        content = delta.get("content", choice.get("text", ""))
        if isinstance(content, list):
            return "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, Mapping)
            )
        return str(content or "")

    @staticmethod
    def _read_body(response, state: ReportState) -> bytes:
        """Read the response body, checking for cancellation between chunks."""
        if state.cancel_requested is None:
            return response.read()
        read = getattr(response, "read1", None) or response.read
        buffer = bytearray()
        while True:
            state.check_cancelled()
            chunk = read(_READ_CHUNK)
            if not chunk:
                return bytes(buffer)
            buffer.extend(chunk)

    @staticmethod
    def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None):
        if cancel_requested is not None and cancel_requested():
            raise AIReportCancelled("AI 报告生成已取消。")

    @staticmethod
    def _extract_text(data: dict, protocol: str) -> str:
        if protocol == "anthropic":
            blocks = data.get("content") or []
            text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        else:
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        if not text:
            raise AIReportError("API 已返回，但未找到报告文本。")
        return text.strip()

    @staticmethod
    def api_key_from_env(provider_name: str) -> str:
        aliases = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY", "Kimi": "MOONSHOT_API_KEY", "GLM": "GLM_API_KEY", "豆包": "DOUBAO_API_KEY"}
        return os.getenv(aliases.get(provider_name, ""), "")
