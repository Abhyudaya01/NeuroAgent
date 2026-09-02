"""Ablation runner for comparing NeuroAgent configurations."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import time
from typing import Any
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.graph import run_agent
from agent.nodes.executor import executor_node
from agent.nodes.planner import planner_node
from agent.nodes.report import report_node
from agent.state import AgentState


ABALATION_CONFIGS = {
    "react_baseline": {
        "use_critic": False,
        "use_memory": False,
        "description": "Planner + executor + report, no critic retries, no retrieval memory.",
    },
    "neuro_no_memory": {
        "use_critic": True,
        "use_memory": False,
        "description": "Full NeuroAgent loop with critic, but retrieval memory disabled.",
    },
    "neuro_full": {
        "use_critic": True,
        "use_memory": True,
        "description": "Full NeuroAgent loop with critic and retrieval memory enabled.",
    },
}

DEFAULT_RESULTS_DIR = ROOT_DIR / "eval" / "results"
"""Default directory used to persist ablation outputs."""


def _initial_state(question: str, session_id: str) -> AgentState:
    """Create the shared graph state used by evaluation runners."""

    return {
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


def _load_questions(path: str) -> list[dict[str, Any]]:
    """Load questions from JSON and normalize into id/question objects."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("questions", [])

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if isinstance(item, str):
            normalized.append({"id": f"q{index}", "question": item})
        elif isinstance(item, dict) and "question" in item:
            normalized_item: dict[str, Any] = {
                "id": str(item.get("id", f"q{index}")),
                "question": str(item["question"]),
            }
            for optional_key in ("answer", "reasoning_types", "wiki_links"):
                if optional_key in item:
                    normalized_item[optional_key] = item[optional_key]
            normalized.append(normalized_item)
        else:
            raise ValueError("Questions file must contain strings or objects with a 'question' field.")

    if not normalized:
        raise ValueError("No questions found in the supplied questions file.")
    return normalized


def _run_react_baseline(question: str, session_id: str) -> AgentState:
    """Run a ReAct-style baseline without critic retries or retrieval memory."""

    state = _initial_state(question, session_id)
    state = planner_node(state)

    while state["current_task_index"] < len(state["plan"]):
        current_index = state["current_task_index"]
        state = executor_node(state)
        state = {
            **state,
            "current_task_index": current_index + 1,
            "tool_history": [],
            "should_retry": False,
        }

    return report_node(state)


def _run_config(question: str, session_id: str, config_name: str) -> AgentState:
    """Dispatch a question to the requested ablation configuration."""

    config = ABALATION_CONFIGS[config_name]
    patch_context = (
        patch("agent.nodes.planner._retrieval_index_is_populated", return_value=False)
        if not config["use_memory"]
        else nullcontext()
    )

    with patch_context:
        if not config["use_critic"]:
            return _run_react_baseline(question, session_id)
        return run_agent(question=question, session_id=session_id)


def _normalize_for_match(text: str) -> str:
    """Normalize free text for simple answer-presence checks."""

    return " ".join(text.lower().split())


def _answer_present(final_report: str, expected_answer: object | None) -> bool | None:
    """Return whether the expected answer appears in the report, when available."""

    if expected_answer in (None, ""):
        return None
    normalized_report = _normalize_for_match(final_report)
    if isinstance(expected_answer, list):
        answer_parts = [str(item) for item in expected_answer if str(item).strip()]
    else:
        answer_parts = [str(expected_answer)]
    normalized_answers = [
        _normalize_for_match(answer)
        for answer in answer_parts
        if _normalize_for_match(answer)
    ]
    if not normalized_answers:
        return None
    return any(answer in normalized_report for answer in normalized_answers)


def _question_summary(result: AgentState, question_row: dict[str, Any]) -> dict[str, Any]:
    """Summarize a completed run into a compact JSON-serializable record."""

    plan = result["plan"]
    confidences = [task["confidence"] for task in plan]
    retries = [task["retries"] for task in plan]
    summary = {
        "id": question_row["id"],
        "question": question_row["question"],
        "num_tasks": len(plan),
        "completed_tasks": sum(1 for task in plan if task["result"].strip()),
        "final_confidence": mean(confidences) if confidences else 0.0,
        "total_retries": sum(retries),
        "hallucination_retries": sum(1 for retry_count in retries if retry_count > 0),
        "final_report": result["final_report"],
    }
    if "answer" in question_row:
        summary["expected_answer"] = question_row["answer"]
        summary["answer_present"] = _answer_present(result["final_report"], question_row["answer"])
    if "reasoning_types" in question_row:
        summary["reasoning_types"] = question_row["reasoning_types"]
    if "wiki_links" in question_row:
        summary["wiki_links"] = question_row["wiki_links"]
    return summary


def _summary_paths(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Compute the output paths for a run-specific and stable-latest summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_name = str(summary["config"])
    questions_stem = Path(str(summary["questions_path"])).stem
    run_path = output_dir / f"{config_name}_{questions_stem}_{timestamp}.json"
    latest_path = output_dir / f"{config_name}_latest.json"
    return run_path, latest_path


def _write_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Persist a run summary to a timestamped file plus a stable latest file."""

    run_path, latest_path = _summary_paths(summary, output_dir)
    payload = json.dumps(summary, indent=2)
    run_path.write_text(payload + "\n", encoding="utf-8")
    latest_path.write_text(payload + "\n", encoding="utf-8")
    return run_path, latest_path


def main() -> None:
    """Parse CLI arguments and run the requested ablation config."""

    parser = argparse.ArgumentParser(description="Run NeuroAgent ablation experiments.")
    parser.add_argument("--config", choices=sorted(ABALATION_CONFIGS), required=True)
    parser.add_argument("--questions", required=True, help="Path to a JSON file of evaluation questions.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory where ablation summaries should be saved.",
    )
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    started_at = time()
    question_results: list[dict[str, Any]] = []

    for index, question_row in enumerate(questions, start=1):
        session_id = f"{args.config}-{question_row['id']}-{index}"
        print(f"[ablation] Running {args.config} on {question_row['id']}: {question_row['question']}")
        result = _run_config(question_row["question"], session_id, args.config)
        question_results.append(_question_summary(result, question_row))

    aggregate_confidences = [item["final_confidence"] for item in question_results]
    aggregate_retries = [item["total_retries"] for item in question_results]
    summary = {
        "config": args.config,
        "description": ABALATION_CONFIGS[args.config]["description"],
        "questions_path": str(Path(args.questions).resolve()),
        "num_questions": len(question_results),
        "avg_final_confidence": mean(aggregate_confidences) if aggregate_confidences else 0.0,
        "avg_total_retries": mean(aggregate_retries) if aggregate_retries else 0.0,
        "runtime_seconds": round(time() - started_at, 2),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": question_results,
    }
    run_path, latest_path = _summary_paths(summary, Path(args.output_dir))
    summary["saved_run_path"] = str(run_path)
    summary["saved_latest_path"] = str(latest_path)
    _write_summary(summary, Path(args.output_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
