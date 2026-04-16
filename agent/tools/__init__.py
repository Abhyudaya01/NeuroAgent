"""Shared tool contracts for the NeuroAgent tool layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ToolResult:
    """Normalized result returned by every external tool adapter.

    Attributes:
        content: Human-readable tool output.
        sources: Source URLs or document identifiers used to produce the output.
        raw_confidence: Tool-native confidence normalized to the range 0.0 to 1.0.
        tool_name: Stable tool identifier used by the executor.
    """

    content: str
    sources: list[str] = field(default_factory=list)
    raw_confidence: float = 0.0
    tool_name: str = ""


__all__ = ["ToolResult"]
