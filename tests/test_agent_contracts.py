"""Tests for typed agent artifacts, tools and structured model output."""

from __future__ import annotations

import json
import unittest

from ai.artifacts import AgentArtifact, ArtifactValidationError
from ai.providers import ProviderConfig
from ai.structured import StructuredModelClient
from ai.tools import ToolRegistry, ToolValidationError


class AgentContractsTest(unittest.TestCase):
    def test_artifact_round_trip_and_validation(self) -> None:
        artifact = AgentArtifact.from_dict(
            {
                "agent": "统计审查 Agent",
                "claims": [{"claim": "AUC", "value": 0.91}],
                "evidence": ["classification.models[0]"],
                "warnings": ["样本量有限"],
                "confidence": 0.8,
            }
        )

        self.assertEqual(artifact.agent, "统计审查 Agent")
        self.assertEqual(artifact.to_dict()["evidence_refs"], ["classification.models[0]"])
        with self.assertRaises(ArtifactValidationError):
            AgentArtifact(agent="", confidence=0.5)
        with self.assertRaises(ArtifactValidationError):
            AgentArtifact(agent="x", confidence=1.1)

    def test_tool_registry_validates_executes_and_records(self) -> None:
        registry = ToolRegistry()

        def validate(arguments: dict[str, object]) -> dict[str, object]:
            if not isinstance(arguments.get("left"), int):
                raise ValueError("left 必须是整数")
            if not isinstance(arguments.get("right"), int):
                raise ValueError("right 必须是整数")
            return arguments

        registry.register(
            name="add",
            description="加法工具",
            handler=lambda left, right: left + right,
            parameters={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            validator=validate,
        )

        self.assertEqual(registry.execute("add", {"left": 2, "right": 3}), 5)
        self.assertEqual(registry.schemas()[0]["function"]["name"], "add")
        self.assertEqual(registry.call_records()[0].status, "completed")
        with self.assertRaises(ToolValidationError):
            registry.execute("add", {"left": "2", "right": 3})

    def test_structured_client_parses_schema_and_sends_tools(self) -> None:
        captured: dict[str, object] = {}

        def execute(request, _timeout: float) -> bytes:
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "```json\n"
                                    '{"agent":"模型评估 Agent", "claims": []}'
                                    "\n```"
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

        client = StructuredModelClient(request_executor=execute)
        provider = ProviderConfig(
            "Test", "openai", "https://example.test/v1", "test-model", ("test-model",)
        )
        artifact = client.generate(
            provider=provider,
            api_key="test-key",
            system_prompt="返回 AgentArtifact",
            user_prompt="分析当前结果",
            output_schema=AgentArtifact,
            tools=[{"type": "function", "function": {"name": "add"}}],
        )

        self.assertIsInstance(artifact, AgentArtifact)
        self.assertEqual(artifact.agent, "模型评估 Agent")
        self.assertEqual(captured["payload"]["tools"][0]["function"]["name"], "add")
        self.assertEqual(captured["payload"]["response_format"]["type"], "json_object")

    def test_structured_client_executes_model_tool_calls(self) -> None:
        registry = ToolRegistry()
        registry.register(
            name="add",
            description="加法工具",
            handler=lambda left, right: left + right,
        )
        rounds = 0

        def execute(_request, _timeout: float) -> bytes:
            nonlocal rounds
            rounds += 1
            if rounds == 1:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "add",
                                                "arguments": '{"left": 2, "right": 3}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"agent":"工具 Agent", "claims": []}'
                            }
                        }
                    ]
                }
            ).encode("utf-8")

        client = StructuredModelClient(request_executor=execute)
        provider = ProviderConfig(
            "Test", "openai", "https://example.test/v1", "test-model", ("test-model",)
        )
        artifact = client.generate(
            provider=provider,
            api_key="test-key",
            system_prompt="调用工具",
            user_prompt="计算结果",
            output_schema=AgentArtifact,
            tools=registry.schemas(),
            tool_registry=registry,
        )

        self.assertEqual(artifact.agent, "工具 Agent")
        self.assertEqual(rounds, 2)
        self.assertEqual(registry.call_records()[0].status, "completed")


if __name__ == "__main__":
    unittest.main()
