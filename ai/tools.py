"""Allow-listed, auditable tools available to analysis agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from threading import RLock
from typing import Any, Callable, Mapping


class ToolRegistryError(RuntimeError):
    """Base class for tool registry errors."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when an agent requests an unregistered tool."""


class ToolValidationError(ToolRegistryError):
    """Raised when tool arguments fail validation."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a registered tool fails during execution."""


@dataclass(frozen=True)
class ToolSpec:
    """Tool metadata and the callable used by the registry."""

    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    })
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    status: str
    elapsed_ms: float
    error: str | None = None


class ToolRegistry:
    """Registry for typed, allow-listed and observable tool calls."""

    def __init__(self, tools: list[ToolSpec] | tuple[ToolSpec, ...] = ()):
        self._tools: dict[str, ToolSpec] = {}
        self._calls: list[ToolCallRecord] = []
        self._lock = RLock()
        for spec in tools:
            self.register(spec)

    def register(
        self,
        spec: ToolSpec | None = None,
        *,
        name: str | None = None,
        description: str = "",
        handler: Callable[..., Any] | None = None,
        parameters: dict[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        replace: bool = False,
    ) -> ToolSpec:
        if spec is None:
            if name is None or handler is None:
                raise ToolValidationError("注册工具需要 name 和 handler")
            spec = ToolSpec(
                name=name,
                description=description,
                handler=handler,
                parameters=parameters or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                validator=validator,
            )
        if not spec.name.strip():
            raise ToolValidationError("工具名称不能为空")
        if not callable(spec.handler):
            raise ToolValidationError(f"工具「{spec.name}」handler 必须可调用")
        with self._lock:
            if spec.name in self._tools and not replace:
                raise ToolValidationError(f"工具「{spec.name}」已注册")
            self._tools[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec:
        with self._lock:
            try:
                return self._tools[name]
            except KeyError as exc:
                raise ToolNotFoundError(f"工具未注册：{name}") from exc

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._tools)

    def schemas(self, names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
        selected = self.names() if names is None else tuple(names)
        return [self.get(name).schema() for name in selected]

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        spec = self.get(name)
        raw_arguments = dict(arguments or {})
        try:
            checked_arguments = (
                spec.validator(raw_arguments) if spec.validator else raw_arguments
            )
        except Exception as exc:  # noqa: BLE001 - validation boundary
            self._record_call(name, raw_arguments, "invalid", 0.0, str(exc))
            raise ToolValidationError(f"工具「{name}」参数无效：{exc}") from exc
        if not isinstance(checked_arguments, Mapping):
            raise ToolValidationError(f"工具「{name}」参数校验器必须返回对象")
        checked_arguments = dict(checked_arguments)
        started = monotonic()
        try:
            output = spec.handler(**checked_arguments)
        except Exception as exc:  # noqa: BLE001 - tool boundary
            elapsed_ms = (monotonic() - started) * 1000
            self._record_call(name, checked_arguments, "failed", elapsed_ms, str(exc))
            raise ToolExecutionError(f"工具「{name}」执行失败：{exc}") from exc
        elapsed_ms = (monotonic() - started) * 1000
        self._record_call(name, checked_arguments, "completed", elapsed_ms, None)
        return output

    def call_records(self) -> tuple[ToolCallRecord, ...]:
        with self._lock:
            return tuple(self._calls)

    def clear_call_records(self) -> None:
        with self._lock:
            self._calls.clear()

    def _record_call(
        self,
        name: str,
        arguments: dict[str, Any],
        status: str,
        elapsed_ms: float,
        error: str | None,
    ) -> None:
        with self._lock:
            self._calls.append(
                ToolCallRecord(name, dict(arguments), status, elapsed_ms, error)
            )


__all__ = [
    "ToolCallRecord",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "ToolValidationError",
]
