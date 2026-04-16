"""Critic node implementing a Reflexion-style self-evaluation loop."""

from __future__ import annotations

import os

import wandb
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.llm_retry import invoke_with_rate_limit_retry
from agent.state import AgentState, CRITIC_THRESHOLD, MAX_ITERATIONS, MAX_RETRIES

DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"
"""Fallback chat model for the critic when no env override is provided."""


class CriticDimensionScores(BaseModel):
    """Explicit per-dimension critic scores for structured output parsing."""

    model_config = ConfigDict(extra="forbid")

    factual_grounding: float = Field(ge=0.0, le=1.0, default=0.0)
    source_reliability: float = Field(ge=0.0, le=1.0, default=0.0)
    completeness: float = Field(ge=0.0, le=1.0, default=0.0)
    specificity: float = Field(ge=0.0, le=1.0, default=0.0)


class CriticOutput(BaseModel):
    """Structured self-evaluation produced by the critic model."""

    model_config = ConfigDict(extra="forbid")

    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall judgment of answer quality from 0.0 to 1.0.",
    )
    dimension_scores: CriticDimensionScores = Field(
        default_factory=CriticDimensionScores,
        description=(
            "Per-dimension scores for factual_grounding, source_reliability, "
            "completeness, and specificity."
        )
    )
    verbal_feedback: str = Field(
        min_length=1,
        description="Concrete explanation of the answer's weaknesses.",
    )
    should_retry: bool = Field(
        description="Whether the current sub-task should be retried.",
    )

    @model_validator(mode="after")
    def ensure_dimension_scores(self) -> "CriticOutput":
        """Backfill missing per-dimension scores for weaker model outputs.

        Some lower-capability chat models may omit `dimension_scores` even when
        structured output is requested. To keep the critic node running during
        development, populate any missing dimensions with the overall score.
        """

        expected_dimensions = (
            "factual_grounding",
            "source_reliability",
            "completeness",
            "specificity",
        )
        normalized_scores = self.dimension_scores.model_dump()
        for dimension in expected_dimensions:
            normalized_scores.setdefault(dimension, self.overall_score)
        self.dimension_scores = CriticDimensionScores(**normalized_scores)
        return self


CRITIC_SYSTEM_PROMPT = """You are the critic node for an autonomous research agent inspired by Reflexion.

Evaluate the current sub-task result on these four dimensions:
- factual_grounding: is the answer supported by the retrieved content?
- source_reliability: are the sources credible and relevant?
- completeness: does the answer fully address the sub-task?
- specificity: is the answer specific enough, or is it vague and generic?

Be strict. A score above 0.7 means you are confident this answer is factually grounded and complete. Do not give high scores to vague or unsourced answers.

Return structured output only.
Use scores between 0.0 and 1.0.
Write verbal_feedback as concise but actionable criticism focused on what is missing or untrustworthy.
"""


def _build_llm(api_key: str) -> ChatOpenAI:
    """Create the chat model used by the critic node."""

    model_name = os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )


def _log_critic_evaluation(session_id: str, task_id: int, output: CriticOutput) -> None:
    """Log each critic evaluation to Weights & Biases."""

    if wandb.run is None:
        wandb.init(
            project="NeuroAgent",
            job_type="critic",
            config={"session_id": session_id},
            mode="online" if os.getenv("WANDB_API_KEY") else "disabled",
        )

    wandb.log(
        {
            "critic/task_id": task_id,
            "critic/overall_score": output.overall_score,
            "critic/verbal_feedback": output.verbal_feedback,
            "critic/should_retry": output.should_retry,
        }
    )


def _invoke_critic(
    llm: ChatOpenAI,
    task_description: str,
    task_result: str,
    sources: list[str],
) -> CriticOutput:
    """Call the critic model with the current sub-task evidence."""

    structured_llm = llm.with_structured_output(
        CriticOutput,
        method="function_calling",
    )
    prompt = (
        f"Sub-task description:\n{task_description}\n\n"
        f"Sub-task result:\n{task_result or '[empty result]'}\n\n"
        f"Sources:\n{sources or ['[no sources provided]']}\n"
    )
    return invoke_with_rate_limit_retry(
        lambda: structured_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
    )


def critic_node(state: AgentState) -> AgentState:
    """Evaluate the current sub-task result and decide whether to retry.

    The critic scores the current result, updates the shared state with
    the latest numeric and verbal feedback, logs the evaluation to
    Weights & Biases, and either retries the same task or advances to
    the next task. If the graph exceeds the configured iteration limit,
    retries are forcibly disabled as a circuit breaker.
    """

    current_index = state["current_task_index"]
    plan = [task.copy() for task in state["plan"]]

    if current_index < 0 or current_index >= len(plan):
        return {
            **state,
            "error": f"Invalid current_task_index for critic: {current_index}",
            "iteration_count": state["iteration_count"] + 1,
        }

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    task = plan[current_index]
    print("\n[critic] Starting critic step")
    print(f"[critic] Evaluating task {task['id']}: {task['description']}")
    print(f"[critic] Chat model: {os.getenv('OPENAI_CHAT_MODEL', DEFAULT_CHAT_MODEL)}")
    llm = _build_llm(api_key)
    model_output = _invoke_critic(
        llm=llm,
        task_description=task["description"],
        task_result=task["result"],
        sources=task["sources"],
    )

    should_retry = (
        model_output.overall_score < CRITIC_THRESHOLD
        and task["retries"] < MAX_RETRIES
        and state["iteration_count"] <= MAX_ITERATIONS
    )
    if state["iteration_count"] > MAX_ITERATIONS:
        should_retry = False

    critic_output = CriticOutput(
        overall_score=model_output.overall_score,
        dimension_scores=model_output.dimension_scores,
        verbal_feedback=model_output.verbal_feedback,
        should_retry=should_retry,
    )

    _log_critic_evaluation(state["session_id"], task["id"], critic_output)

    print(
        f"[critic] Score={critic_output.overall_score:.2f} | "
        f"should_retry={should_retry} | retries={task['retries']}"
    )
    print(f"[critic] Feedback: {critic_output.verbal_feedback}")

    if should_retry:
        task["retries"] += 1
        task["status"] = "retrying"
        next_index = current_index
        next_tool_history: list[str] = []
        print(f"[critic] Retrying task {task['id']} and resetting tool history")
    else:
        task["status"] = "completed"
        next_index = min(current_index + 1, len(plan))
        next_tool_history = state["tool_history"]
        if next_index < len(plan):
            print(f"[critic] Moving to next task index {next_index}")
        else:
            print("[critic] All tasks complete, routing to report")

    return {
        **state,
        "plan": plan,
        "current_task_index": next_index,
        "tool_history": next_tool_history,
        "critic_score": critic_output.overall_score,
        "critic_feedback": critic_output.verbal_feedback,
        "should_retry": should_retry,
        "error": state["error"],
        "iteration_count": state["iteration_count"] + 1,
    }


__all__ = ["CriticOutput", "critic_node"]
