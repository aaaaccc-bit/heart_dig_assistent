"""Provider-compatible structured model output for specialist agents."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPException
from typing import Any, Callable, Mapping

from .providers import ProviderConfig, open_custom_request
from .tools import ToolRegistry


class StructuredOutputError(RuntimeError):
    """Raised when a provider response cannot satisfy the requested schema."""


class StructuredOutputCancelled(StructuredOutputError):
    """Raised when a structured model request is cancelled by its caller."""


@dataclass(frozen=True)
class _ModelResponse:
    text: str
    tool_calls: tuple[dict[str, Any], ...]
    raw_message: Any = None


class StructuredModelClient:
    """Call an OpenAI-compatible or Anthropic provider and parse JSON output."""

    def __init__(
        self,
        timeout: float = 90,
        *,
        request_executor: Callable[[urllib.request.Request, float], bytes] | None = None,
    ):
        self.timeout = timeout
        self.request_executor = request_executor

    def generate(
        self,
        *,
        provider: ProviderConfig,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        output_schema: type | Callable[[Mapping[str, Any]], Any] = dict,
        model: str | None = None,
        tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = True,
        tool_registry: ToolRegistry | None = None,
        max_tool_rounds: int = 4,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Any:
        try:
            headers = provider.request_headers(api_key or "")
            url = provider.endpoint
        except ValueError as exc:
            raise StructuredOutputError(str(exc)) from exc
        model = model or provider.default_model
        schema = getattr(output_schema, "json_schema", None)
        schema_hint = (
            "\n输出字段结构："
            + json.dumps(schema(), ensure_ascii=False)
            if callable(schema)
            else ""
        )
        prompt = (
            f"{user_prompt}{schema_hint}\n\n"
            "请只输出一个合法 JSON 对象，不要使用 Markdown 代码围栏。"
        )
        if max_tool_rounds < 0:
            raise StructuredOutputError("max_tool_rounds 不能为负数")
        if provider.protocol == "anthropic":
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        formatted_tools = self._format_tools(tools, provider.protocol) if tools else []
        for round_index in range(max_tool_rounds + 1):
            payload = self._build_payload(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                tools=formatted_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode and not provider.is_custom,
            )
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            response = self._parse_response(
                self._decode_json(self._request(request, cancel_requested, user_endpoint=provider.uses_user_endpoint)),
                provider.protocol,
            )
            if response.tool_calls:
                if tool_registry is None:
                    raise StructuredOutputError("模型请求了工具，但当前 Agent 未配置工具注册表")
                if round_index >= max_tool_rounds:
                    raise StructuredOutputError("模型工具调用超过最大轮数")
                self._append_tool_messages(
                    messages,
                    response,
                    provider.protocol,
                    tool_registry,
                )
                continue
            text = response.text
            break
        else:  # pragma: no cover - loop always returns or breaks
            raise StructuredOutputError("结构化模型调用未产生最终输出")
        try:
            parsed = json.loads(self._strip_code_fence(text))
        except (json.JSONDecodeError, TypeError) as exc:
            raise StructuredOutputError("模型返回内容不是合法 JSON") from exc
        return self._coerce_schema(parsed, output_schema)

    def _request(
        self,
        request: urllib.request.Request,
        cancel_requested: Callable[[], bool] | None,
        *,
        user_endpoint: bool = False,
    ) -> bytes:
        self._raise_if_cancelled(cancel_requested)
        try:
            if self.request_executor is not None:
                raw = self.request_executor(request, self.timeout)
            else:
                open_request = open_custom_request if user_endpoint else urllib.request.urlopen
                with open_request(request, timeout=self.timeout) as response:
                    raw = self._read_body(response, cancel_requested)
        except StructuredOutputError:
            raise
        except urllib.error.HTTPError as exc:
            with exc:
                detail = "" if user_endpoint else exc.read(2048).decode("utf-8", errors="replace")[:500]
            suffix = f"：{detail}" if detail else ""
            raise StructuredOutputError(f"{request.full_url} 请求失败（HTTP {exc.code}）{suffix}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, HTTPException) as exc:
            raise StructuredOutputError(f"无法连接结构化模型服务：{exc}") from exc
        self._raise_if_cancelled(cancel_requested)
        if not isinstance(raw, bytes):
            raise StructuredOutputError("请求执行器必须返回 bytes")
        return raw

    @staticmethod
    def _read_body(response, cancel_requested: Callable[[], bool] | None) -> bytes:
        if cancel_requested is None:
            return response.read()
        read = getattr(response, "read1", None) or response.read
        buffer = bytearray()
        while True:
            StructuredModelClient._raise_if_cancelled(cancel_requested)
            chunk = read(4096)
            if not chunk:
                return bytes(buffer)
            buffer.extend(chunk)

    @staticmethod
    def _decode_json(raw: bytes) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StructuredOutputError("API 返回内容不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("API 返回根对象必须是 JSON 对象")
        return value

    @staticmethod
    def _extract_text(data: Mapping[str, Any], protocol: str) -> str:
        if protocol == "anthropic":
            blocks = data.get("content") or []
            values = [
                block.get("text", "")
                for block in blocks
                if isinstance(block, Mapping) and block.get("type") == "text"
            ]
            text = "\n".join(values)
        else:
            choices = data.get("choices") or []
            message = choices[0].get("message") if choices else None
            content = message.get("content") if isinstance(message, Mapping) else ""
            if isinstance(content, list):
                text = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, Mapping)
                )
            else:
                text = content or ""
        if not isinstance(text, str) or not text.strip():
            raise StructuredOutputError("API 已返回，但未找到结构化文本")
        return text.strip()

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        value = text.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if len(lines) >= 2:
                value = "\n".join(lines[1:-1]).strip()
        return value

    @staticmethod
    def _coerce_schema(value: Any, output_schema: Any) -> Any:
        if output_schema is dict:
            return value
        if not isinstance(value, Mapping):
            raise StructuredOutputError("结构化输出必须是 JSON 对象")
        try:
            from_dict = getattr(output_schema, "from_dict", None)
            if callable(from_dict):
                return from_dict(value)
            return output_schema(**dict(value))
        except Exception as exc:  # noqa: BLE001 - schema boundary
            raise StructuredOutputError(f"结构化输出校验失败：{exc}") from exc

    @staticmethod
    def _build_payload(
        *,
        provider: ProviderConfig,
        model: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        if provider.protocol == "anthropic":
            payload: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": messages,
            }
        else:
            payload = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
        return payload

    @classmethod
    def _parse_response(
        cls, data: Mapping[str, Any], protocol: str
    ) -> _ModelResponse:
        if protocol == "anthropic":
            blocks = data.get("content") or []
            text_parts = []
            tool_calls = []
            for block in blocks:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                            "block": dict(block),
                        }
                    )
            return _ModelResponse("\n".join(text_parts).strip(), tuple(tool_calls), blocks)

        choices = data.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not isinstance(message, Mapping):
            raise StructuredOutputError("API 已返回，但未找到模型消息")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, Mapping)
            )
        tool_calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError as exc:
                raise StructuredOutputError("模型工具参数不是合法 JSON") from exc
            if not isinstance(arguments, Mapping):
                raise StructuredOutputError("模型工具参数必须是 JSON 对象")
            tool_calls.append(
                {
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": dict(arguments),
                }
            )
        return _ModelResponse(str(content).strip(), tuple(tool_calls), dict(message))

    @staticmethod
    def _append_tool_messages(
        messages: list[dict[str, Any]],
        response: _ModelResponse,
        protocol: str,
        tool_registry: ToolRegistry,
    ) -> None:
        if protocol == "anthropic":
            messages.append({"role": "assistant", "content": response.raw_message})
            results = []
            for call in response.tool_calls:
                output = tool_registry.execute(call["name"], call["arguments"])
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": json.dumps(output, ensure_ascii=False, default=str),
                    }
                )
            messages.append({"role": "user", "content": results})
            return

        messages.append({"role": "assistant", "content": response.raw_message.get("content", ""), "tool_calls": response.raw_message.get("tool_calls", [])})
        for call in response.tool_calls:
            output = tool_registry.execute(call["name"], call["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(output, ensure_ascii=False, default=str),
                }
            )

    @staticmethod
    def _format_tools(
        tools: list[dict[str, Any]] | tuple[dict[str, Any], ...], protocol: str
    ) -> list[dict[str, Any]]:
        if protocol != "anthropic":
            return list(tools)
        formatted = []
        for tool in tools:
            function = tool.get("function", {})
            formatted.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return formatted

    @staticmethod
    def _raise_if_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
        if cancel_requested is not None and cancel_requested():
            raise StructuredOutputCancelled("结构化模型调用已取消")


__all__ = [
    "StructuredModelClient",
    "StructuredOutputCancelled",
    "StructuredOutputError",
]
