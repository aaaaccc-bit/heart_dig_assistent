"""Deterministic reviews for report inputs and generated drafts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .artifacts import AgentArtifact
from .report_guard import (
    citation_sources,
    has_risky_wording,
    requires_hard_block,
    scan_risk,
    unknown_citations,
)


_SECTION_REQUIREMENTS = {
    "回归分析": ("验证指标", "拟合优度"),
    "随机森林分析": ("验证指标", "模型参数"),
    "风险分层": ("验证指标", "分档"),
    "分类评估": ("模型对比", "判定阈值"),
    "特征洞察": ("特征", "相关矩阵"),
}


def has_statistical_sections(parameters: Mapping[str, Any]) -> bool:
    return "无监督分群" in parameters or any(
        section in parameters for section in _SECTION_REQUIREMENTS
    )


def _non_finite_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_non_finite_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_non_finite_paths(item, f"{path}[{index}]"))
        return paths
    return []


def needs_statistical_review(parameters: Mapping[str, Any]) -> bool:
    return has_statistical_sections(parameters) or bool(
        _non_finite_paths(parameters, "parameters")
    )


def statistical_review(record: Any) -> AgentArtifact:
    """Check the shape and numerical hygiene of available analysis results."""

    parameters = getattr(record, "parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("报告参数必须是对象")

    claims: list[dict[str, Any]] = []
    warnings: list[str] = []
    evidence_refs: list[str] = []
    checked_sections: list[str] = []

    for section, required_fields in _SECTION_REQUIREMENTS.items():
        if section not in parameters:
            continue
        section_value = parameters[section]
        checked_sections.append(section)
        if not isinstance(section_value, Mapping):
            warnings.append(f"{section} 结果不是对象，无法完成字段审查")
            continue
        missing = [field for field in required_fields if field not in section_value]
        if missing:
            warnings.append(f"{section} 缺少字段：{'、'.join(missing)}")
        claims.append(
            {
                "section": section,
                "status": "checked" if not missing else "incomplete",
                "available_fields": sorted(section_value),
            }
        )
        evidence_refs.append(f"parameters.{section}")

    clusters = parameters.get("无监督分群")
    if clusters is not None:
        if not isinstance(clusters, list):
            warnings.append("无监督分群结果不是数组，无法检查每个聚类方法")
        else:
            for index, item in enumerate(clusters):
                if not isinstance(item, Mapping):
                    warnings.append(f"无监督分群[{index}] 不是对象")
                    continue
                missing = [
                    field
                    for field in ("算法", "选定簇数", "轮廓系数")
                    if field not in item
                ]
                if missing:
                    warnings.append(
                        f"无监督分群[{index}] 缺少字段：{'、'.join(missing)}"
                    )
                evidence_refs.append(f"parameters.无监督分群[{index}]")

    non_finite = _non_finite_paths(dict(parameters), "parameters")
    if non_finite:
        warnings.append("检测到非有限数值：" + "、".join(non_finite[:8]))

    highlights = getattr(record, "highlights", ())
    singleton_warning = any(
        "单点簇" in str(getattr(item, "detail", "")) for item in highlights
    )
    if singleton_warning:
        warnings.append("无监督分群存在单点簇，不能直接据此解释分群质量")

    if not checked_sections and clusters is None:
        warnings.append("当前报告没有可审查的模型分析模块")

    next_actions = ["写作 Agent 只能使用已通过字段审查的结构化结果"]
    if warnings:
        next_actions.append("在报告中明确标注统计限制并交由人工复核")

    return AgentArtifact(
        agent="统计审查 Agent",
        claims=claims,
        evidence_refs=evidence_refs,
        warnings=warnings,
        confidence=1.0,
        next_actions=next_actions,
        metadata={"checked_sections": checked_sections},
    )


def citation_safety_review(
    *,
    instruction: str,
    knowledge_context: str,
    evidence: list[dict[str, Any]],
    statistical_artifact: AgentArtifact | None = None,
) -> AgentArtifact:
    """Check citation availability and high-risk wording before writing."""

    sources = citation_sources(knowledge_context)
    warnings: list[str] = []
    evidence_refs = [
        ref
        for item in evidence
        if isinstance(item, Mapping)
        for ref in item.get("evidence_refs", [])
    ]

    if not knowledge_context.strip():
        warnings.append("当前没有知识库上下文，写作 Agent 不得生成 [KB-n] 引用")
    elif not sources:
        warnings.append("知识库上下文缺少可解析的来源信息")
    unknown = unknown_citations(knowledge_context, sources)
    if unknown:
        warnings.append("知识库上下文存在未知引用编号：" + ", ".join(f"[KB-{item}]" for item in unknown))

    risk_categories = scan_risk(
        "\n".join(
            [instruction or "", json.dumps(evidence, ensure_ascii=False, default=str)]
        )
    )
    if risk_categories:
        warnings.append("输入或证据命中高风险类别：" + "、".join(risk_categories))

    if statistical_artifact is not None and statistical_artifact.warnings:
        warnings.append("统计审查 Agent 已发现结构或统计限制")

    claims = [
        {"claim": "知识库来源数量", "value": len(sources)},
        {"claim": "高风险类别", "value": list(risk_categories)},
        {"claim": "证据引用数量", "value": len(evidence_refs)},
    ]
    next_actions = ["最终报告继续执行引用编号校验和安全扫描"]
    if warnings:
        next_actions.append("报告生成后必须进入人工复核")

    return AgentArtifact(
        agent="引用与安全审查 Agent",
        claims=claims,
        evidence_refs=evidence_refs,
        warnings=warnings,
        confidence=1.0,
        next_actions=next_actions,
        metadata={"source_count": len(sources), "risk_categories": list(risk_categories)},
    )


@dataclass(frozen=True)
class DraftReview:
    unknown_refs: tuple[str, ...]
    risk_categories: tuple[str, ...]
    risky_wording: bool
    hard_blocked: bool

    @property
    def severity(self) -> tuple[bool, bool, int]:
        return self.hard_blocked, self.risky_wording, len(self.unknown_refs)

    @property
    def requires_revision(self) -> bool:
        """Only high-risk wording warrants another model generation pass."""

        return self.hard_blocked or self.risky_wording

    def revision_actions(self) -> tuple[str, ...]:
        actions = []
        if self.unknown_refs:
            refs = "、".join(f"[KB-{number}]" for number in self.unknown_refs)
            actions.append(f"移除无来源引用 {refs}，仅在来源支持论断时使用知识库已有编号。")
        if self.risky_wording or self.hard_blocked:
            categories = "、".join(self.risk_categories) or "过度确定的表述"
            actions.append(
                f"修订命中「{categories}」的内容，保留汇总统计和研究局限，"
                "删除个体诊疗建议、隐私信息及超出证据范围的确定性结论。"
            )
        return tuple(actions)


def review_draft(
    text: str, knowledge_context: str, *, safety_mode: str, safety_strength: int
) -> DraftReview:
    risks = scan_risk(text)
    return DraftReview(
        unknown_citations(text, citation_sources(knowledge_context)),
        risks,
        has_risky_wording(text),
        requires_hard_block(risks, safety_mode, safety_strength),
    )


__all__ = ["citation_safety_review", "statistical_review"]
