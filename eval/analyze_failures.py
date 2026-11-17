"""Analyze NeuroAgent ablation outputs for resume-relevant failure modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    """Load one saved ablation summary."""

    return json.loads(path.read_text(encoding="utf-8"))


def _premature_terminations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find runs with explicit evidence that execution stopped before the plan ended."""

    return [
        result
        for result in results
        if result.get("current_task_index") is not None
        and int(result["current_task_index"]) < int(result.get("num_tasks", 0))
    ]


def _critic_miscalibrations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find likely critic-confidence mismatches using available gold-answer metadata."""

    flagged: list[dict[str, Any]] = []
    for result in results:
        answer_present = result.get("answer_present")
        confidence = result.get("critic_confidence")
        if confidence is None:
            continue
        confidence = float(confidence)
        if answer_present is False and confidence >= 0.7:
            flagged.append({**result, "miscalibration_type": "high_confidence_missing_gold_answer"})
        elif answer_present is True and confidence < 0.5:
            flagged.append({**result, "miscalibration_type": "low_confidence_answer_present"})
    return flagged


def _num_gold_answer_cases(results: list[dict[str, Any]]) -> int:
    """Count results with enough metadata for answer-based calibration checks."""

    return sum(
        1 for result in results
        if result.get("answer_present") is not None and result.get("critic_confidence") is not None
    )


def analyze(paths: list[Path]) -> dict[str, Any]:
    """Build a compact cross-run failure analysis report."""

    runs = [_load_summary(path) for path in paths]
    run_summaries: list[dict[str, Any]] = []
    all_premature: list[dict[str, Any]] = []
    all_miscalibrated: list[dict[str, Any]] = []

    for path, run in zip(paths, runs):
        results = list(run.get("results", []))
        premature = _premature_terminations(results)
        miscalibrated = _critic_miscalibrations(results)
        gold_answer_cases = _num_gold_answer_cases(results)
        termination_cases = sum(row.get("current_task_index") is not None for row in results)
        incomplete_cases = sum(
            row.get("num_tasks") is not None
            and int(row.get("output_tasks", row.get("completed_tasks", 0))) < int(row["num_tasks"])
            for row in results
        )
        all_premature.extend({**item, "config": run.get("config"), "run_file": str(path)} for item in premature)
        all_miscalibrated.extend(
            {**item, "config": run.get("config"), "run_file": str(path)}
            for item in miscalibrated
        )

        run_summaries.append(
            {
                "file": str(path),
                "config": run.get("config"),
                "questions_path": run.get("questions_path"),
                "num_questions": run.get("num_questions", len(results)),
                "avg_final_confidence": run.get("avg_final_confidence"),
                "avg_total_retries": run.get("avg_total_retries"),
                "premature_termination_rate": len(premature) / termination_cases if termination_cases else None,
                "termination_observed_cases": termination_cases,
                "incomplete_output_rate": incomplete_cases / len(results) if results else None,
                "unfinished_task_rate": sum(
                    row.get("num_tasks") is not None and int(row.get("completed_tasks", 0)) < int(row["num_tasks"])
                    for row in results
                ) / len(results) if results else None,
                "operational_error_rate": sum(row.get("status") == "error" for row in results) / len(results) if results else None,
                "critic_miscalibration_rate": (
                    len(miscalibrated) / gold_answer_cases if gold_answer_cases else None
                ),
                "gold_answer_cases": gold_answer_cases,
            }
        )

    retry_rates = [
        float(summary["avg_total_retries"])
        for summary in run_summaries
        if summary.get("avg_total_retries") is not None
    ]
    return {
        "num_runs": len(runs),
        "runs": run_summaries,
        "dominant_failure_modes": {
            "premature_termination_cases": len(all_premature),
            "critic_miscalibration_cases": len(all_miscalibrated),
            "avg_retries_across_runs": mean(retry_rates) if retry_rates else 0.0,
        },
        "premature_termination_examples": [
            {
                "config": item.get("config"),
                "id": item.get("id"),
                "completed_tasks": item.get("completed_tasks"),
                "num_tasks": item.get("num_tasks"),
            }
            for item in all_premature[:10]
        ],
        "critic_miscalibration_examples": [
            {
                "config": item.get("config"),
                "id": item.get("id"),
                "critic_confidence": item.get("critic_confidence"),
                "answer_present": item.get("answer_present"),
                "miscalibration_type": item.get("miscalibration_type"),
            }
            for item in all_miscalibrated[:10]
        ],
    }


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Analyze NeuroAgent ablation failure modes.")
    parser.add_argument("summaries", nargs="+", help="Saved ablation summary JSON files.")
    parser.add_argument("--output", help="Optional path for the failure analysis JSON.")
    args = parser.parse_args()

    analysis = analyze([Path(path) for path in args.summaries])
    payload = json.dumps(analysis, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
