"""Shared state definitions for the NeuroAgent LangGraph workflow.

This module defines the graph state container and related execution
constants used to coordinate planning, tool use, critique, and final
report synthesis across agent nodes.
"""

from typing import NotRequired, TypedDict


CRITIC_THRESHOLD = 0.7
"""Minimum acceptable critic score before the workflow can progress."""

MAX_RETRIES = 3
"""Maximum number of retries allowed for an individual sub-task."""

MAX_ITERATIONS = 50
"""Maximum number of node transitions allowed before stopping execution."""


class PlanTask(TypedDict):
    """A single research sub-task tracked in the agent execution plan.

    Each task records what should be done, which tool should be used,
    which earlier tasks it depends on, its current status, the latest
    result, an associated confidence estimate, and how many retries
    have been attempted so far.
    """

    id: int
    description: str
    tool: str
    dependencies: list[int]
    status: str
    result: str
    sources: list[str]
    confidence: float
    critic_score: NotRequired[float]
    resolved_answer: NotRequired[str]
    evidence_gap: NotRequired[str]
    evidence_sources: NotRequired[dict[str, str]]
    supplemental_evidence: NotRequired[str]
    dependency_rechecks: NotRequired[int]
    retries: int


class AgentState(TypedDict):
    """Complete LangGraph state for the autonomous research agent.

    The state captures the original research question, task planning and
    execution progress, memory retrieved for context, critic outputs,
    the final synthesized report, and safety fields used for error
    handling and circuit breaking.
    """

    research_question: str
    session_id: str
    plan: list[PlanTask]
    current_task_index: int
    tool_history: list[str]
    memory_context: str
    critic_score: float
    critic_feedback: str
    should_retry: bool
    final_report: str
    wandb_task_rows: list[dict[str, object]]
    error: str | None
    dependency_to_retry: NotRequired[int | None]
    iteration_count: int
