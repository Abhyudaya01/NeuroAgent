"""Tests for the ablation runner scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from eval.ablation import _load_questions, _write_summary


def test_load_questions_supports_strings_and_objects(tmp_path: Path) -> None:
    """Question loading should normalize both string and object formats."""

    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [
                "What is long-context memory?",
                {"id": "q2", "question": "How does Reflexion improve reliability?"},
            ]
        ),
        encoding="utf-8",
    )

    loaded = _load_questions(str(questions_path))

    assert loaded == [
        {"id": "q1", "question": "What is long-context memory?"},
        {"id": "q2", "question": "How does Reflexion improve reliability?"},
    ]


def test_write_summary_persists_timestamped_and_latest_files(tmp_path: Path) -> None:
    """Ablation summaries should be saved to both run-specific and latest files."""

    summary = {
        "config": "react_baseline",
        "questions_path": "/tmp/questions.json",
        "results": [{"id": "q1", "question": "Example", "final_report": "Done"}],
    }

    run_path, latest_path = _write_summary(summary, tmp_path)

    assert run_path.exists()
    assert latest_path.exists()
    assert latest_path.name == "react_baseline_latest.json"
    assert json.loads(run_path.read_text(encoding="utf-8"))["config"] == "react_baseline"
    assert json.loads(latest_path.read_text(encoding="utf-8"))["results"][0]["id"] == "q1"
