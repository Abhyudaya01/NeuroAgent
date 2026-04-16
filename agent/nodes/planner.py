"""Planner node for decomposing a research question into atomic sub-tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import wandb
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.llm_retry import invoke_with_rate_limit_retry
from agent.state import AgentState, PlanTask

DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
"""Fallback chat model for the planner when no env override is provided."""


class PlannerTask(BaseModel):
    """Structured representation of a planner-generated sub-task."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="Unique integer identifier for the sub-task.")
    description: str = Field(
        min_length=1,
        description="Specific atomic action the agent should perform.",
    )
    tool: Literal["web_search", "code_interpreter", "document_retrieval", "synthesis"] = Field(
        description="Best tool to execute this sub-task.",
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="IDs of prerequisite sub-tasks that must complete first.",
    )


class PlannerResponse(BaseModel):
    """Structured plan returned by the planner model."""

    model_config = ConfigDict(extra="forbid")

    tasks: list[PlannerTask] = Field(
        min_length=3,
        max_length=7,
        description="Ordered list of 3 to 7 atomic sub-tasks.",
    )


DEFAULT_INDEX_PATH = "faiss_index"
"""Default local retrieval index path used for planning-time tool selection."""


def _retrieval_index_is_populated(index_path: str = DEFAULT_INDEX_PATH) -> bool:
    """Return whether the local retrieval index appears usable.

    The planner uses this to avoid assigning `document_retrieval` when the
    local FAISS store has not been built yet or contains no chunks.
    """

    metadata_path = Path(index_path)
    if metadata_path.suffix:
        metadata_path = metadata_path.with_suffix(".json")
    else:
        metadata_path = metadata_path.with_suffix(".json")

    if not metadata_path.exists():
        return False

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    chunks = payload.get("chunks", [])
    return bool(chunks)


def _build_planner_prompt(retrieval_available: bool, *, strict: bool) -> str:
    """Create the planner prompt with environment-aware tool guidance."""

    retrieval_guidance = (
        "A populated local retrieval index is available. Use document_retrieval only "
        "for tasks that review or synthesize already indexed documents."
        if retrieval_available
        else "No populated local retrieval index is available. Do not assign "
        "document_retrieval; prefer web_search instead."
    )

    base_prompt = f"""You are the planning node for an autonomous research agent.

Decompose the user's research question into 3 to 7 ordered sub-tasks.
Requirements:
- Be specific and concrete.
- Keep each task atomic and executable.
- Assign exactly one most relevant tool to each task.
- Valid tools are: web_search, code_interpreter, document_retrieval, synthesis.
- Include dependencies as a list of prior task IDs that must complete first.
- Use integer task IDs.
- Prefer web_search for discovering external papers, articles, and evidence.
- Use synthesis for analytical tasks such as comparing, ranking, summarizing findings, or identifying the best approaches after evidence has been gathered.
- Use code_interpreter only when the task explicitly requires executable Python, calculations, or programmatic data manipulation.
- {retrieval_guidance}
- Do not include commentary outside the structured response.
"""

    if not strict:
        return base_prompt

    return f"""{base_prompt}

Return ONLY a valid structured plan.
Every task must include:
- id: integer
- description: non-empty string
- tool: one of web_search, code_interpreter, document_retrieval, synthesis
- dependencies: list of integer task IDs referencing earlier tasks only

Additional constraints:
- Maximum 7 tasks.
- No prose, markdown, or explanation outside the structured response.
"""


def _build_llm(api_key: str) -> ChatOpenAI:
    """Create the chat model used by the planner node."""

    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )


def _invoke_planner(
    llm: ChatOpenAI,
    research_question: str,
    system_prompt: str,
) -> PlannerResponse:
    """Invoke the planner LLM and validate the structured response."""

    structured_llm = llm.with_structured_output(
        PlannerResponse,
        method="json_schema",
    )
    return invoke_with_rate_limit_retry(
        lambda: structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=research_question),
            ]
        )
    )


def _normalize_plan(tasks: list[PlannerTask], *, retrieval_available: bool) -> list[PlanTask]:
    """Convert planner output into the shared LangGraph plan representation.

    The planner may emit `document_retrieval` tasks even when no retrieval
    store is populated yet. In that case, remap those tasks to `web_search`
    so the execution plan remains actionable.
    """

    return [
        PlanTask(
            id=task.id,
            description=task.description,
            tool=(
                "web_search"
                if task.tool == "document_retrieval" and not retrieval_available
                else "synthesis"
                if task.tool == "code_interpreter" and _is_synthesis_task(task.description)
                else task.tool
            ),
            dependencies=task.dependencies,
            status="pending",
            result="",
            sources=[],
            confidence=0.0,
            retries=0,
        )
        for task in tasks
    ]


def _is_synthesis_task(description: str) -> bool:
    """Return whether a task description is better handled by synthesis."""

    lowered = description.lower()
    synthesis_markers = (
        "analyze",
        "compare",
        "synthesize",
        "identify the best",
        "summarize findings",
    )
    return any(marker in lowered for marker in synthesis_markers)


def _log_plan_to_wandb(plan: list[PlanTask], session_id: str) -> None:
    """Log the generated plan to Weights & Biases as a task table."""

    if wandb.run is None:
        wandb.init(
            project="NeuroAgent",
            job_type="planner",
            config={"session_id": session_id},
            mode="online" if os.getenv("WANDB_API_KEY") else "disabled",
        )

    table = wandb.Table(columns=["task_id", "description", "tool"])
    for task in plan:
        table.add_data(task["id"], task["description"], task["tool"])

    wandb.log({"planner_plan": table})


def planner_node(state: AgentState) -> AgentState:
    """Create an execution plan for the current research question.

    The planner calls GPT-4o with structured output, validates the
    returned task list, retries once with a stricter prompt if parsing
    fails, logs the resulting plan to Weights & Biases, and returns an
    updated graph state with the plan initialized.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    print("\n[planner] Starting planning step")
    print(f"[planner] Research question: {state['research_question']}")
    retrieval_available = _retrieval_index_is_populated()
    print(f"[planner] Retrieval index available: {retrieval_available}")
    print(f"[planner] Chat model: {os.getenv('OPENAI_CHAT_MODEL', DEFAULT_CHAT_MODEL)}")

    llm = _build_llm(api_key)

    try:
        response = _invoke_planner(
            llm,
            state["research_question"],
            _build_planner_prompt(retrieval_available, strict=False),
        )
    except (OutputParserException, ValidationError, ValueError, TypeError):
        try:
            response = _invoke_planner(
                llm,
                state["research_question"],
                _build_planner_prompt(retrieval_available, strict=True),
            )
        except (OutputParserException, ValidationError, ValueError, TypeError) as second_error:
            raise ValueError("Planner failed to return a valid structured plan.") from second_error

    plan = _normalize_plan(response.tasks, retrieval_available=retrieval_available)
    _log_plan_to_wandb(plan, state["session_id"])

    print(f"[planner] Generated {len(plan)} tasks")
    for task in plan:
        print(
            f"[planner] Task {task['id']}: tool={task['tool']} | "
            f"description={task['description']}"
        )

    updated_state: AgentState = {
        **state,
        "plan": plan,
        "current_task_index": 0,
    }
    return updated_state


__all__ = [
    "PlannerResponse",
    "PlannerTask",
    "planner_node",
]
