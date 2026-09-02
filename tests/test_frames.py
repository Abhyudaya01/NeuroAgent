"""Tests for FRAMES benchmark preparation utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval.frames import load_frames_from_file, normalize_frames_row, write_questions


def test_normalize_frames_row_supports_common_huggingface_fields() -> None:
    """FRAMES rows should convert into NeuroAgent question records."""

    row = {
        "Prompt ID": "abc123",
        "Prompt": "What is the answer?",
        "Answer": "42",
        "reasoning_types": ["numerical"],
        "wiki_links": ["https://en.wikipedia.org/wiki/42"],
    }

    assert normalize_frames_row(row, 1) == {
        "id": "abc123",
        "question": "What is the answer?",
        "answer": "42",
        "reasoning_types": ["numerical"],
        "wiki_links": ["https://en.wikipedia.org/wiki/42"],
    }


def test_load_frames_from_file_supports_csv(tmp_path: Path) -> None:
    """Local FRAMES CSV exports should be accepted for offline prep."""

    csv_path = tmp_path / "frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_id", "question", "answer", "reasoning_types", "wiki_links"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": "frames_1",
                "question": "Which scientist won?",
                "answer": "Marie Curie",
                "reasoning_types": json.dumps(["temporal", "multiple_constraints"]),
                "wiki_links": json.dumps(["https://en.wikipedia.org/wiki/Marie_Curie"]),
            }
        )

    rows = load_frames_from_file(csv_path)

    assert rows[0]["id"] == "frames_1"
    assert rows[0]["answer"] == "Marie Curie"
    assert rows[0]["reasoning_types"] == ["temporal", "multiple_constraints"]


def test_write_questions_respects_limit(tmp_path: Path) -> None:
    """FRAMES question files can be capped for controlled ablation subsets."""

    rows = [
        {"id": "q1", "question": "First?", "answer": "A", "reasoning_types": [], "wiki_links": []},
        {"id": "q2", "question": "Second?", "answer": "B", "reasoning_types": [], "wiki_links": []},
    ]
    output_path = write_questions(rows, tmp_path / "frames_questions_1.json", limit=1)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(payload) == 1
    assert payload[0]["id"] == "q1"
