"""Typed artifacts exchanged by analysis agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence


class ArtifactValidationError(ValueError):
    """Raised when an agent artifact does not satisfy its output contract."""


def _list_of_mappings(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactValidationError(f"{field_name} 必须是对象数组")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ArtifactValidationError(
                f"{field_name}[{index}] 必须是对象"
            )
        result.append(dict(item))
    return result


def _list_of_strings(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactValidationError(f"{field_name} 必须是字符串数组")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ArtifactValidationError(
                f"{field_name}[{index}] 必须是非空字符串"
            )
        result.append(item.strip())
    return result


@dataclass
class AgentArtifact:
    """Stable output contract for a specialist or review agent."""

    agent: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None
    next_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str) or not self.agent.strip():
            raise ArtifactValidationError("agent 必须是非空字符串")
        self.agent = self.agent.strip()
        self.claims = _list_of_mappings(self.claims, "claims")
        self.evidence_refs = _list_of_strings(self.evidence_refs, "evidence_refs")
        self.warnings = _list_of_strings(self.warnings, "warnings")
        self.next_actions = _list_of_strings(self.next_actions, "next_actions")
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise ArtifactValidationError("confidence 必须是 0 到 1 之间的数字")
            self.confidence = float(self.confidence)
            if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
                raise ArtifactValidationError("confidence 必须是 0 到 1 之间的数字")
        if not isinstance(self.metadata, Mapping):
            raise ArtifactValidationError("metadata 必须是对象")
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AgentArtifact":
        if not isinstance(payload, Mapping):
            raise ArtifactValidationError("结构化 Agent 输出必须是对象")
        evidence_refs = payload.get("evidence_refs", payload.get("evidence", []))
        return cls(
            agent=payload.get("agent", ""),
            claims=payload.get("claims", []),
            evidence_refs=evidence_refs,
            warnings=payload.get("warnings", []),
            confidence=payload.get("confidence"),
            next_actions=payload.get("next_actions", []),
            metadata=payload.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return a provider-neutral schema description for model prompts."""

        return {
            "type": "object",
            "required": ["agent", "claims", "evidence_refs", "warnings"],
            "properties": {
                "agent": {"type": "string"},
                "claims": {"type": "array", "items": {"type": "object"}},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "next_actions": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            },
            "additionalProperties": False,
        }


__all__ = ["AgentArtifact", "ArtifactValidationError"]
