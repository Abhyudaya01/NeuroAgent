"""Utilities for preparing Google FRAMES questions for NeuroAgent ablations."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "google/frames-benchmark"
DEFAULT_SPLIT = "test"
DEFAULT_CACHE_DIR = "eval/cache/huggingface"


def _coerce_list(value: Any) -> list[str]:
    """Normalize FRAMES list-like fields across CSV, JSON, and datasets rows."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                if "|" in stripped:
                    return [part.strip() for part in stripped.split("|") if part.strip()]
                return [stripped]
            return _coerce_list(parsed)
        if isinstance(parsed, str) and "|" in parsed:
            return [part.strip() for part in parsed.split("|") if part.strip()]
        if isinstance(parsed, str):
            return [parsed]
        return _coerce_list(parsed)
    return [str(value)]


def _first_present(row: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """Return the first non-empty field from a row using known FRAMES aliases."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_frames_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert one FRAMES record into NeuroAgent's question-row schema."""

    prompt_id = _first_present(
        row,
        ("Prompt ID", "prompt_id", "id", "ID"),
        f"frames_{index:04d}",
    )
    question = _first_present(row, ("Prompt", "prompt", "question", "Question"))
    answer = _first_present(row, ("Answer", "answer", "gold_answer", "Gold Answer"))
    reasoning_types = _coerce_list(
        _first_present(row, ("reasoning_types", "Reasoning Types", "reasoning_type"), [])
    )
    wiki_links = _coerce_list(
        _first_present(row, ("wiki_links", "Wikipedia Links", "wikipedia_links", "source_urls"), [])
    )

    if not question:
        raise ValueError(f"FRAMES row {index} has no question/prompt field.")

    return {
        "id": str(prompt_id),
        "question": str(question),
        "answer": str(answer),
        "reasoning_types": reasoning_types,
        "wiki_links": wiki_links,
    }


def load_frames_from_file(path: Path) -> list[dict[str, Any]]:
    """Load FRAMES rows from a local JSON, JSONL, or CSV file."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("questions") or payload.get("data") or payload.get("rows") or []
        else:
            rows = payload
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("FRAMES input must be .json, .jsonl, or .csv.")

    if not isinstance(rows, list):
        raise ValueError("FRAMES input must contain a list of rows.")
    return [normalize_frames_row(dict(row), index) for index, row in enumerate(rows, start=1)]


def load_frames_from_huggingface(
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    """Load FRAMES from Hugging Face datasets when the optional package is installed."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
        raise RuntimeError(
            "Install the optional 'datasets' package or pass --input-file with a local FRAMES export."
        ) from exc

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    return [
        normalize_frames_row(dict(row), index)
        for index, row in enumerate(dataset, start=1)
    ]


def write_questions(rows: list[dict[str, Any]], output_path: Path, limit: int | None) -> Path:
    """Persist a NeuroAgent-compatible FRAMES question file."""

    selected_rows = rows[:limit] if limit is not None else rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """CLI entry point for preparing FRAMES question subsets."""

    parser = argparse.ArgumentParser(description="Prepare FRAMES questions for NeuroAgent ablations.")
    parser.add_argument("--input-file", help="Optional local FRAMES .json, .jsonl, or .csv export.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Hugging Face split.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Repo-local Hugging Face cache.")
    parser.add_argument("--limit", type=int, default=50, help="Number of questions to write.")
    parser.add_argument(
        "--output",
        default="eval/frames_questions_50.json",
        help="Output path for NeuroAgent-compatible questions.",
    )
    args = parser.parse_args()

    if args.input_file:
        rows = load_frames_from_file(Path(args.input_file))
    else:
        rows = load_frames_from_huggingface(args.dataset, args.split, args.cache_dir)

    output_path = write_questions(rows, Path(args.output), args.limit)
    print(
        json.dumps(
            {
                "dataset": args.input_file or args.dataset,
                "split": args.split if not args.input_file else None,
                "num_loaded": len(rows),
                "num_written": len(rows[: args.limit]) if args.limit is not None else len(rows),
                "output": str(output_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FRAMES preparation failed: {error}", file=sys.stderr)
        raise
