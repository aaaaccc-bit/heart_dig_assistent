"""AI report generation integrations.

The report service imports the optional retrieval stack lazily so the graph
orchestrator remains independently testable in lightweight environments.
"""

from .providers import PROVIDER_PRESETS, ProviderConfig
from .artifacts import AgentArtifact, ArtifactValidationError
from .structured import (
    StructuredModelClient,
    StructuredOutputCancelled,
    StructuredOutputError,
)
from .review_agents import citation_safety_review, statistical_review
from .memory import ExperienceEntry, LongTermMemory, MemoryEntry, get_shared_memory
from .task_queue import (
    DistributedTaskQueue,
    QueueError,
    QueueOwnershipError,
    QueueTask,
    QueueWorker,
    get_shared_task_queue,
)
from .tools import (
    ToolCallRecord,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    ToolValidationError,
)

__all__ = [
    "AIReportCancelled",
    "AIReportService",
    "AgentArtifact",
    "ArtifactValidationError",
    "ProviderConfig",
    "PROVIDER_PRESETS",
    "StructuredModelClient",
    "StructuredOutputCancelled",
    "StructuredOutputError",
    "citation_safety_review",
    "statistical_review",
    "ToolCallRecord",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolSpec",
    "ToolValidationError",
    "LongTermMemory",
    "MemoryEntry",
    "ExperienceEntry",
    "get_shared_memory",
    "DistributedTaskQueue",
    "QueueError",
    "QueueOwnershipError",
    "QueueTask",
    "QueueWorker",
    "get_shared_task_queue",
]


def __getattr__(name: str):
    if name in {"AIReportCancelled", "AIReportService"}:
        from .report_generator import AIReportCancelled, AIReportService

        return {
            "AIReportCancelled": AIReportCancelled,
            "AIReportService": AIReportService,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
