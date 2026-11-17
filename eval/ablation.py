"""Ablation runner for comparing NeuroAgent configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from openai import AuthenticationError, PermissionDeniedError
from langchain_core.callbacks.usage import get_usage_metadata_callback
import wandb

from agent.graph import run_agent
from agent.nodes.executor import executor_node
from agent.nodes.planner import planner_node
from agent.nodes.report import report_node
from agent.state import AgentState
from eval.costs import estimate_chat_cost, GPT4O_PRICES


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

    if len({row["id"] for row in normalized}) != len(normalized):
        raise ValueError("Question IDs must be unique for checkpointing.")
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
        "completed_tasks": sum(1 for task in plan if task["status"] == "completed" and task["result"].strip()),
        "output_tasks": sum(1 for task in plan if task["result"].strip()),
        "completion_metric": "status_completed_with_nonempty_output",
        "final_confidence": mean(confidences) if confidences else 0.0,
        "total_retries": sum(retries),
        "hallucination_retries": sum(1 for retry_count in retries if retry_count > 0),
        "final_report": result["final_report"],
        "tasks": plan,
        "current_task_index": result["current_task_index"],
        "error": result.get("error"),
        "status": "completed",
    }
    critic_scores = [task["critic_score"] for task in plan if "critic_score" in task]
    summary["critic_confidence"] = mean(critic_scores) if critic_scores else None
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_name = str(summary["config"])
    questions_stem = Path(str(summary["questions_path"])).stem
    run_path = output_dir / f"{config_name}_{questions_stem}_{timestamp}.json"
    latest_path = output_dir / f"{config_name}_latest.json"
    return run_path, latest_path


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Replace a checkpoint only after its full JSON payload is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Persist a stable run checkpoint and update latest only on completion."""

    run_path, latest_path = _summary_paths(summary, output_dir)
    run_path = Path(summary.setdefault("saved_run_path", str(run_path)))
    summary["saved_latest_path"] = str(latest_path)
    _atomic_write(run_path, summary)
    if summary.get("status", "completed") == "completed":
        _atomic_write(latest_path, summary)
    return run_path, latest_path


def _experiment_metadata(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint inputs and code so resumed runs keep the same experiment."""

    source = hashlib.sha256()
    for directory in ("agent", "eval"):
        for path in sorted((ROOT_DIR / directory).rglob("*.py")):
            source.update(str(path.relative_to(ROOT_DIR)).encode())
            source.update(path.read_bytes())
    memory = hashlib.sha256()
    for name in ("faiss_index.index", "faiss_index.json"):
        path = ROOT_DIR / name
        memory.update(path.read_bytes() if path.exists() else b"missing")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "questions_sha256": hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest(),
        "source_sha256": source.hexdigest(),
        "memory_sha256": memory.hexdigest(),
        "git_revision": revision,
        "package_versions": {
            package: version(package)
            for package in ("openai", "langchain-core", "langchain-openai", "langgraph", "tavily-python")
        },
        "planner_critic_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
        "synthesis_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
        "usage_scope": "Chat model tokens reported by LangChain, excluding embeddings and external tools.",
    }


def run_benchmark(
    config_name: str,
    questions_path: str,
    output_dir: Path,
    *,
    limit: int | None = None,
    resume: Path | None = None,
    max_chat_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Run sequential questions with per-question usage, failures, and checkpoints."""

    load_dotenv()
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive.")
    if max_chat_cost_usd is not None and max_chat_cost_usd <= 0:
        raise ValueError("Chat cost budget must be positive.")
    questions = _load_questions(questions_path)[:limit]
    metadata = _experiment_metadata(questions)
    summary: dict[str, Any] = {
        "config": config_name,
        "description": ABALATION_CONFIGS[config_name]["description"],
        "questions_path": str(Path(questions_path).resolve()),
        "target_num_questions": len(questions),
        "metadata": metadata,
        "results": [],
        "runtime_seconds": 0.0,
    }
    if resume:
        summary = json.loads(resume.read_text(encoding="utf-8"))
        if summary["config"] != config_name or summary.get("metadata") != metadata:
            raise ValueError("Resume requires identical configuration, model, code, memory, and questions.")
        summary["saved_run_path"] = str(resume.resolve())
    rows = summary["results"]
    completed_ids = {row["id"] for row in rows}
    summary["status"] = "running"
    _write_summary(summary, output_dir)

    for index, question in enumerate(questions, start=1):
        if question["id"] in completed_ids:
            continue
        if max_chat_cost_usd is not None:
            spent = estimate_chat_cost(summary.get("chat_usage_by_model", {}))
            reserve = max(0.50, 2 * max(
                (estimate_chat_cost(row.get("chat_usage_by_model", {})) for row in rows), default=0,
            ))
            if spent + reserve > max_chat_cost_usd:
                summary["status"] = "budget_limited"
                _write_summary(summary, output_dir)
                return summary
        started = perf_counter()
        session_id = f"{config_name}-{question['id']}-{index}"
        print(f"[ablation] Running {session_id}", flush=True)
        fatal_error = False
        with get_usage_metadata_callback() as usage:
            try:
                result = _run_config(question["question"], session_id, config_name)
                row = _question_summary(result, question)
            except Exception as error:
                fatal_error = (
                    isinstance(error, (AuthenticationError, PermissionDeniedError))
                    or getattr(error, "code", None) == "insufficient_quota"
                )
                row = {
                    "id": question["id"], "question": question["question"],
                    "status": "error", "error": f"{type(error).__name__}: {error}",
                    "final_report": "", "final_confidence": 0.0, "total_retries": 0,
                }
                if "answer" in question:
                    row.update(expected_answer=question["answer"], answer_present=False)
            finally:
                # The baseline also starts W&B in planner_node. Close it per question.
                if wandb.run is not None:
                    wandb.finish()
        row["runtime_seconds"] = round(perf_counter() - started, 3)
        row["chat_usage_by_model"] = usage.usage_metadata
        rows.append(row)
        summary["runtime_seconds"] += row["runtime_seconds"]
        _update_aggregates(summary)
        if fatal_error:
            summary["status"] = "blocked"
            _write_summary(summary, output_dir)
            return summary
        _write_summary(summary, output_dir)

    summary["status"] = "completed"
    _update_aggregates(summary)
    _write_summary(summary, output_dir)
    return summary


def _update_aggregates(summary: dict[str, Any]) -> None:
    """Summarize attempted questions, keeping operational failures in the denominator."""

    rows = summary["results"]
    summary["num_questions"] = len(rows)
    summary["num_errors"] = sum(row.get("status") == "error" for row in rows)
    summary["avg_final_confidence"] = mean(row["final_confidence"] for row in rows) if rows else 0.0
    summary["avg_total_retries"] = mean(row["total_retries"] for row in rows) if rows else 0.0
    summary["avg_runtime_seconds"] = mean(row["runtime_seconds"] for row in rows) if rows else 0.0
    gold_rows = [row for row in rows if row.get("answer_present") is not None]
    summary["answer_presence_rate"] = mean(row["answer_present"] for row in gold_rows) if gold_rows else None
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        for model, usage in row.get("chat_usage_by_model", {}).items():
            total = totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            for key in total:
                total[key] += usage.get(key, 0)
    summary["chat_usage_by_model"] = totals
    try:
        summary["estimated_chat_cost_usd"] = estimate_chat_cost(totals)
        summary["chat_cost_rates"] = GPT4O_PRICES
    except ValueError:
        summary["estimated_chat_cost_usd"] = None
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()


def main() -> None:
    """Parse CLI arguments and run or resume an ablation."""

    parser = argparse.ArgumentParser(description="Run NeuroAgent ablation experiments.")
    parser.add_argument("--config", choices=sorted(ABALATION_CONFIGS), required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, help="Run the first N questions for a smoke test.")
    parser.add_argument("--max-chat-cost-usd", type=float, help="Stop between questions with a conservative GPT-4o chat cost reserve.")
    parser.add_argument("--resume", type=Path, help="Resume an interrupted run checkpoint.")
    args = parser.parse_args()
    summary = run_benchmark(
        args.config, args.questions, args.output_dir, limit=args.limit, resume=args.resume,
        max_chat_cost_usd=args.max_chat_cost_usd,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if summary["status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
