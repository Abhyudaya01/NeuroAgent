"""Tests for the ablation runner scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from eval.ablation import _answer_present, _load_questions, _write_summary


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


def test_load_questions_preserves_frames_metadata(tmp_path: Path) -> None:
    """FRAMES gold-answer metadata should survive benchmark normalization."""

    questions_path = tmp_path / "frames_questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {
                    "id": "frames_001",
                    "question": "Which city hosted the event?",
                    "answer": "Paris",
                    "reasoning_types": ["temporal"],
                    "wiki_links": ["https://en.wikipedia.org/wiki/Paris"],
                }
            ]
        ),
        encoding="utf-8",
    )

    loaded = _load_questions(str(questions_path))

    assert loaded == [
        {
            "id": "frames_001",
            "question": "Which city hosted the event?",
            "answer": "Paris",
            "reasoning_types": ["temporal"],
            "wiki_links": ["https://en.wikipedia.org/wiki/Paris"],
        }
    ]


def test_answer_present_handles_string_and_list_answers() -> None:
    """Simple answer-presence checks support common FRAMES answer shapes."""

    assert _answer_present("The answer is Paris, France.", "Paris") is True
    assert _answer_present("The answer is 1997.", ["1996", "1997"]) is True
    assert _answer_present("The answer is Berlin.", "Paris") is False
    assert _answer_present("The answer is Berlin.", "") is None


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
