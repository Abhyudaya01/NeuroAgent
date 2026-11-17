"""LangGraph wiring for the NeuroAgent research workflow."""

from __future__ import annotations

import os
from statistics import mean
from typing import Literal

import wandb
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.nodes.critic import critic_node
from agent.nodes.executor import executor_node
from agent.nodes.planner import planner_node
from agent.nodes.report import report_node
from agent.state import AgentState, MAX_ITERATIONS


def route_after_critic(state: AgentState) -> Literal["executor", "report"]:
    """Route execution after the critic evaluates the current sub-task."""

    if state["iteration_count"] >= MAX_ITERATIONS - 2:
        return "report"
    if state["should_retry"]:
        return "executor"
    if state["current_task_index"] < len(state["plan"]):
        return "executor"
    return "report"


def _executor_with_logging(state: AgentState) -> AgentState:
    """Run the executor node and accumulate task-level metrics in graph state."""

    updated_state = executor_node(state)
    current_index = state["current_task_index"]
    task_rows = [row.copy() for row in state["wandb_task_rows"]]
    if 0 <= current_index < len(updated_state["plan"]):
        task = updated_state["plan"][current_index]
        task_rows.append(
            {
                "task_id": task["id"],
                "description": task["description"],
                "tool": task["tool"],
                "confidence": task["confidence"],
                "retries": task["retries"],
                "status": task["status"],
                "critic_score": None,
            }
        )

    return {
        **updated_state,
        "wandb_task_rows": task_rows,
    }


def _critic_with_logging(state: AgentState) -> AgentState:
    """Run the critic node and log critic scores to Weights & Biases."""

    updated_state = critic_node(state)
    task_rows = [row.copy() for row in state["wandb_task_rows"]]
    if task_rows:
        task_rows[-1]["critic_score"] = updated_state["critic_score"]
    if wandb.run is not None:
        wandb.log(
            {
                "critic_score": updated_state["critic_score"],
                "task_index": state["current_task_index"],
            }
        )
    return {
        **updated_state,
        "wandb_task_rows": task_rows,
    }


def build_graph():
    """Build and compile the NeuroAgent LangGraph workflow."""

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", _executor_with_logging)
    workflow.add_node("critic", _critic_with_logging)
    workflow.add_node("report", report_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "critic")
    workflow.add_conditional_edges("critic", route_after_critic)
    workflow.add_edge("report", END)

    return workflow.compile()


def run_agent(question: str, session_id: str) -> AgentState:
    """Run the compiled graph from an initial question and session ID."""

    load_dotenv()

    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="neuro-agent",
            name=session_id,
            config={"question": question},
        )

    graph = build_graph()
    initial_state: AgentState = {
        "research_question": question,
        "session_id": session_id,
        "plan": [],
        "current_task_index": -1,
        "tool_history": [],
        "memory_context": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "should_retry": False,
        "final_report": "",
        "wandb_task_rows": [],
        "error": None,
        "iteration_count": 0,
    }
    final_state = graph.invoke(initial_state, {"recursion_limit": MAX_ITERATIONS + 5})

    if wandb.run is not None:
        plan = final_state["plan"]
        task_table = wandb.Table(
            columns=[
                "task_id",
                "description",
                "tool",
                "confidence",
                "retries",
                "status",
                "critic_score",
            ]
        )
        for row in final_state["wandb_task_rows"]:
            task_table.add_data(
                row["task_id"],
                row["description"],
                row["tool"],
                row["confidence"],
                row["retries"],
                row["status"],
                row["critic_score"],
            )
        confidences = [task["confidence"] for task in plan]
        retries = [task["retries"] for task in plan]
        final_confidence = mean(confidences) if confidences else 0.0
        total_retries = sum(retries)
        hallucination_retries = sum(1 for retry_count in retries if retry_count > 0)

        wandb.log(
            {
                "task_metrics": task_table,
                "final_confidence": final_confidence,
                "total_retries": total_retries,
                "num_tasks": len(plan),
                "hallucination_retries": hallucination_retries,
            }
        )
        wandb.log({"report": wandb.Html(f"<pre>{final_state['final_report']}</pre>")})
        wandb.finish()

    return final_state


if __name__ == "__main__":
    SAMPLE_QUESTION = "What are the best recent approaches for long-context memory in LLM agents?"
    final_state = run_agent(
        question=SAMPLE_QUESTION,
        session_id="sample-session",
    )
    print(final_state["final_report"])
