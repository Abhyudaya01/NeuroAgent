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


def test_benchmark_checkpoints_failures_and_resumes_without_repeating(tmp_path, monkeypatch):
    """An interrupted run preserves successes and rejects changed experiment inputs."""
    import pytest
    import eval.ablation as ablation

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "One?"}, {"id": "q2", "question": "Two?"}]))
    monkeypatch.setattr(ablation, "load_dotenv", lambda: None)
    monkeypatch.setattr(ablation, "_experiment_metadata", lambda rows: {"questions": rows})
    calls = []

    def run(question, session, config):
        calls.append(question)
        if question == "Two?":
            raise KeyboardInterrupt()
        state = ablation._initial_state(question, session)
        state["final_report"] = "Answer"
        return state

    monkeypatch.setattr(ablation, "_run_config", run)
    with pytest.raises(KeyboardInterrupt):
        ablation.run_benchmark("react_baseline", str(questions), tmp_path)
    checkpoint = next(tmp_path.glob("react_baseline_questions_*.json"))
    saved = json.loads(checkpoint.read_text())
    assert len(saved["results"]) == 1
    assert saved["status"] == "running"
    assert not (tmp_path / "react_baseline_latest.json").exists()

    monkeypatch.setattr(ablation, "_run_config", lambda q, session, config: ablation._initial_state(q, session))
    result = ablation.run_benchmark("react_baseline", str(questions), tmp_path, resume=checkpoint)
    assert [row["id"] for row in result["results"]] == ["q1", "q2"]
    assert result["status"] == "completed"
    assert result["saved_run_path"] == str(checkpoint.resolve())
    assert calls == ["One?", "Two?"]
    questions.write_text(json.dumps([{"id": "q3", "question": "Changed?"}]))
    with pytest.raises(ValueError, match="Resume requires"):
        ablation.run_benchmark("react_baseline", str(questions), tmp_path, resume=checkpoint)


def test_benchmark_keeps_errors_in_metrics(tmp_path, monkeypatch):
    import eval.ablation as ablation

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "One?", "answer": "Paris"}]))
    monkeypatch.setattr(ablation, "load_dotenv", lambda: None)
    monkeypatch.setattr(ablation, "_experiment_metadata", lambda rows: {})

    def fail(*args):
        raise RuntimeError("Tool failed")

    monkeypatch.setattr(ablation, "_run_config", fail)
    result = ablation.run_benchmark("react_baseline", str(questions), tmp_path)
    assert result["num_errors"] == 1
    assert result["answer_presence_rate"] == 0
    assert result["results"][0]["chat_usage_by_model"] == {}
    assert result["results"][0]["runtime_seconds"] >= 0


def test_auth_failure_stops_after_first_question(tmp_path, monkeypatch):
    import httpx
    from openai import AuthenticationError
    import eval.ablation as ablation

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["One?", "Two?"]))
    monkeypatch.setattr(ablation, "load_dotenv", lambda: None)
    monkeypatch.setattr(ablation, "_experiment_metadata", lambda rows: {})
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.com"))

    def fail(*args):
        raise AuthenticationError("Invalid key", response=response, body=None)

    monkeypatch.setattr(ablation, "_run_config", fail)
    result = ablation.run_benchmark("react_baseline", str(questions), tmp_path)
    assert result["status"] == "blocked"
    assert result["num_questions"] == 1
    assert result["target_num_questions"] == 2
    assert not (tmp_path / "react_baseline_latest.json").exists()


def test_chat_usage_is_captured_even_when_question_later_fails(tmp_path, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    import eval.ablation as ablation

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["One?"]))
    monkeypatch.setattr(ablation, "load_dotenv", lambda: None)
    monkeypatch.setattr(ablation, "_experiment_metadata", lambda rows: {})
    model = FakeMessagesListChatModel(responses=[AIMessage(
        content="Plan", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "test-model"},
    )])

    def fail_after_planning(*args):
        model.invoke("Plan the question")
        raise RuntimeError("Search failed")

    monkeypatch.setattr(ablation, "_run_config", fail_after_planning)
    result = ablation.run_benchmark("react_baseline", str(questions), tmp_path)
    assert result["num_errors"] == 1
    assert result["chat_usage_by_model"]["test-model"]["total_tokens"] == 15


def test_budget_stop_saves_partial_run_and_preserves_latest(tmp_path, monkeypatch):
    import eval.ablation as ablation

    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["One?"]))
    monkeypatch.setattr(ablation, "load_dotenv", lambda: None)
    monkeypatch.setattr(ablation, "_experiment_metadata", lambda rows: {})
    def unexpected_call(*args):
        raise AssertionError("Budget should stop the run before any API call")
    monkeypatch.setattr(ablation, "_run_config", unexpected_call)
    result = ablation.run_benchmark("react_baseline", str(questions), tmp_path, max_chat_cost_usd=0.1)
    assert result["status"] == "budget_limited"
    assert result["results"] == []
    assert Path(result["saved_run_path"]).exists()
    assert not (tmp_path / "react_baseline_latest.json").exists()


def test_cost_estimation_rejects_unknown_models():
    import pytest
    from eval.costs import estimate_chat_cost

    assert estimate_chat_cost({"gpt-4o-2024-08-06": {"input_tokens": 1000, "output_tokens": 100}}) == pytest.approx(0.0035)
    with pytest.raises(ValueError, match="No verified price"):
        estimate_chat_cost({"unknown-model": {"input_tokens": 1000}})
