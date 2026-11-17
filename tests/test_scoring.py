"""Regression checks for honest answer grading and failure classification."""

import json
from unittest.mock import Mock

import pytest

from eval import scoring
from eval.analyze_failures import analyze


def test_judge_checkpoint_resumes_and_includes_operational_failures(tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"results": [
        {"id": "a", "question": "City?", "expected_answer": "Paris", "final_report": "Paris."},
        {"id": "b", "question": "Year?", "expected_answer": "1997", "status": "error"},
    ]}))
    output = tmp_path / "grades.json"
    judge = Mock()
    judge.invoke.return_value = scoring.AnswerGrade(correct=True, predicted_answer="Paris", answer_quote="Paris.", explanation="Matches.")
    model = Mock()
    model.with_structured_output.return_value = judge
    monkeypatch.setattr(scoring, "ChatOpenAI", Mock(return_value=model))
    monkeypatch.setattr(scoring, "load_dotenv", lambda: None)
    result = scoring.score_run(source, output)
    assert result["judge_accuracy"] == 0.5
    assert result["num_scored"] == 2
    assert result["status"] == "completed"
    scoring.score_run(source, output)
    judge.invoke.assert_called_once()
    source.write_text(json.dumps({"results": []}))
    with pytest.raises(ValueError, match="different run"):
        scoring.score_run(source, output)


def test_missing_output_is_distinct_from_early_stop_and_tool_confidence(tmp_path):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"results": [
        {"id": "old", "num_tasks": 4, "completed_tasks": 3, "answer_present": False, "final_confidence": 0.9},
        {"id": "failed_tool", "num_tasks": 4, "completed_tasks": 3, "current_task_index": 4},
        {"id": "early", "num_tasks": 4, "completed_tasks": 2, "current_task_index": 2},
    ]}))
    result = analyze([source])["runs"][0]
    assert result["incomplete_output_rate"] == 1
    assert result["premature_termination_rate"] == 0.5
    assert result["termination_observed_cases"] == 2
    assert result["critic_miscalibration_rate"] is None


def test_judge_cannot_fill_missing_report_answer_from_gold(tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"results": [{"id": "a", "question": "Letter count?", "expected_answer": "17",
                                               "final_report": "Blocked. The film title is missing."}]}))
    judge = Mock()
    judge.invoke.return_value = scoring.AnswerGrade(correct=True, predicted_answer="17",
        answer_quote="The answer is 17 letters.", explanation="Matches gold")
    model = Mock()
    model.with_structured_output.return_value = judge
    monkeypatch.setattr(scoring, "ChatOpenAI", Mock(return_value=model))
    monkeypatch.setattr(scoring, "load_dotenv", lambda: None)
    result = scoring.score_run(source, tmp_path / "grades.json")
    assert result["judge_accuracy"] == 0
    assert "rejected" in result["results"][0]["explanation"]


def test_quote_format_retry_recovers_supported_answer(tmp_path, monkeypatch):
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"results": [{"id": "a", "question": "Who?", "expected_answer": "Paris",
                                               "final_report": "The answer is Paris."}]}))
    judge = Mock()
    judge.invoke.side_effect = [
        scoring.AnswerGrade(correct=True, predicted_answer="Paris", answer_quote="The answer was Paris.", explanation="Equivalent"),
        scoring.AnswerGrade(correct=True, predicted_answer="Paris", answer_quote="The answer is Paris.", explanation="Direct assertion"),
    ]
    model = Mock()
    model.with_structured_output.return_value = judge
    monkeypatch.setattr(scoring, "ChatOpenAI", Mock(return_value=model))
    monkeypatch.setattr(scoring, "load_dotenv", lambda: None)
    result = scoring.score_run(source, tmp_path / "grades.json")
    assert result["judge_accuracy"] == 1
    assert result["results"][0]["quote_validation_retries"] == 1
    assert judge.invoke.call_count == 2


def test_report_witness_recovers_typographic_apostrophe():
    report = "Lawton, as Chelsea’s leading striker, took the Number 9 shirt."
    grade = scoring.AnswerGrade(correct=True, predicted_answer="9",
        answer_quote="Lawton, as Chelsea\x19s leading striker, took the Number 9 shirt.", explanation="Direct assertion")
    witness = scoring._source_witness(grade, report)
    assert witness and witness in report
    assert "Chelsea’s" in witness


def test_report_witness_accepts_explicit_multi_part_answers():
    report = "1. **Bridgeton Railway Station**: nearby.\n2. **Dalmarnock Railway Station**: nearby."
    grade = scoring.AnswerGrade(correct=True, predicted_answer="Bridgeton Railway Station and Dalmarnock Railway Station",
        answer_quote=report, explanation="Both stations are listed")
    assert scoring._has_answer_witness(grade, report)
    assert not scoring._has_answer_witness(grade.model_copy(update={"predicted_answer": "Bridgeton Railway Station and Missing Station"}),report)


def test_report_witness_cannot_invent_rounded_result():
    grade = scoring.AnswerGrade(correct=True, predicted_answer="506000",
        answer_quote="Oklahoma City | 506,132", explanation="Rounded")
    assert not scoring._has_answer_witness(grade, "Oklahoma City | 506,132")


def test_ordered_answer_parts_match_formatted_list():
    report = "Chronological order:\n1. **World War I**: 1914\n2. **Great Depression**: 1929\n3. **Berlin Wall falls**: 1989"
    grade = scoring.AnswerGrade(correct=True, predicted_answer="World War I", answer_quote=report,
        answer_parts=["World War I", "Great Depression", "Berlin Wall falls"], explanation="Correct order")
    assert scoring._has_answer_witness(grade,report)
    assert not scoring._has_answer_witness(grade.model_copy(update={"answer_parts":["Berlin Wall falls","World War I"]}),report)


def test_answer_parts_cannot_hide_missing_computed_result():
    report = "Inputs are 1978 and 1880."
    grade = scoring.AnswerGrade(correct=True, predicted_answer="98", answer_quote=report,
        answer_parts=["1978","1880"], explanation="Computed by judge")
    assert not scoring._has_answer_witness(grade,report)
