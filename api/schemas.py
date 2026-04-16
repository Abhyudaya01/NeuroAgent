"""Pydantic request/response schemas for the NeuroAgent API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    """Request payload for executing a NeuroAgent research run."""

    question: str = Field(
        min_length=5,
        description="Research question to answer with the NeuroAgent workflow.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional client-provided session identifier for logging/tracing.",
    )


class TaskSummary(BaseModel):
    """Compact summary of one task from the execution plan."""

    id: int
    description: str
    tool: str
    status: str
    confidence: float
    retries: int
    sources: list[str]


class RunAgentResponse(BaseModel):
    """API response for a completed NeuroAgent run."""

    session_id: str
    question: str
    final_report: str
    final_confidence: float
    total_retries: int
    num_tasks: int
    tasks: list[TaskSummary]


class HealthResponse(BaseModel):
    """Basic health response for deployments and uptime checks."""

    status: str
    service: str
