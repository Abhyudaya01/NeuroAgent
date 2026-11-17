"""Regression tests for evidence gates and bounded prerequisite repair."""

from unittest.mock import Mock, patch

import pytest

from agent.evidence import EvidenceQuote, ResearchAnswer, resolve_evidence
from agent.graph import build_graph, route_after_critic
from agent.nodes.critic import CriticOutput, critic_node
from agent.nodes.executor import GeneratedCode
from agent.state import MAX_ITERATIONS, MAX_RETRIES
from agent.tools import ToolResult
from eval.ablation import _initial_state


def task(identifier, tool="web_search", dependencies=None, **updates):
    return {"id": identifier, "description": "Find the first film composed by David Arnold.",
            "tool": tool, "dependencies": dependencies or [], "status": "pending", "result": "",
            "sources": [], "confidence": 0.0, "retries": 0, **updates}


def evidence_model(answer):
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = answer
    return model


@pytest.mark.parametrize("source,quote,supported", [
    ("https://example.com/a", "His first film was The Young Americans.", True),
    ("https://example.com/b", "His first film was The Young Americans.", False),
    ("https://example.com/a", "He won an Oscar for the film.", False),
    ("https://unlisted.example/", "His first film was The Young Americans.", False),
])
def test_quotes_must_belong_to_the_cited_source(source, quote, supported):
    raw = ToolResult(content="two pages", sources=["https://example.com/a", "https://example.com/b"],
                     source_content={"https://example.com/a": "His first film was The Young Americans.",
                                     "https://example.com/b": "His later film was Stargate."},
                     raw_confidence=0.8, tool_name="web_search")
    answer = ResearchAnswer(answer="The Young Americans", evidence=[EvidenceQuote(source=source, quote=quote)])
    with patch("agent.evidence.ChatOpenAI", return_value=evidence_model(answer)):
        result = resolve_evidence(raw, "Find his first film", "How many letters in the first film title?")
    assert bool(result.resolved_answer) is supported
    assert bool(result.evidence_gap) is not supported
    assert result.raw_confidence == (0.8 if supported else 0)


def test_missing_fact_overrides_apparently_supported_answer():
    raw = ToolResult(content="Stargate was scored by David Arnold.", sources=["local.md"],
                     raw_confidence=0.8, tool_name="document_retrieval")
    answer = ResearchAnswer(answer="Stargate", missing_fact="The earliest film is not established.",
                            evidence=[EvidenceQuote(source="local.md", quote=raw.content)])
    with patch("agent.evidence.ChatOpenAI", return_value=evidence_model(answer)):
        result = resolve_evidence(raw, "Find the first film", "Count its title letters")
    assert result.resolved_answer == ""
    assert result.raw_confidence == 0
    assert "earliest" in result.evidence_gap


def test_empty_search_does_not_call_resolver_model():
    with patch("agent.evidence.ChatOpenAI") as model:
        result = resolve_evidence(ToolResult(content="", tool_name="web_search"), "Find a fact", "Question")
    model.assert_not_called()
    assert result.evidence_gap


def test_graph_repairs_prerequisite_and_skips_unrelated_completed_work(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = _initial_state("How many letters in the title of his first film?", "test")
    state["plan"] = [
        task(1, status="failed", resolved_answer="", evidence_gap="Establish the first film."),
        task(2, status="completed", result="Independent work", confidence=0.8),
        task(3, "code_interpreter", [1], description="Count the letters in the first film title."),
    ]
    state["current_task_index"] = 2
    raw = ToolResult(content="His first film was The Young Americans.", sources=["local.md"],
                     source_content={"local.md": "His first film was The Young Americans."},
                     raw_confidence=0.8, tool_name="web_search")
    answer = ResearchAnswer(answer="The Young Americans", evidence=[
        EvidenceQuote(source="local.md", quote=raw.content)])
    with patch("agent.graph.planner_node", side_effect=lambda ignored: state), patch(
        "agent.nodes.executor.run_web_search", return_value=raw
    ) as search, patch("agent.nodes.executor._reformulate_retrieval_query", return_value="first film"), patch(
        "agent.evidence.ChatOpenAI", return_value=evidence_model(answer)
    ), patch("agent.nodes.executor._prepare_code_input", return_value=(GeneratedCode(code="print(17)"), ["local.md"], True)), patch(
        "agent.nodes.executor.run_code", return_value=ToolResult(content="17", raw_confidence=1, tool_name="code_interpreter")
    ) as code, patch("agent.nodes.critic._build_llm"), patch(
        "agent.nodes.critic._invoke_critic", return_value=CriticOutput(overall_score=0.9, verbal_feedback="Supported")
    ), patch("agent.nodes.critic._log_critic_evaluation"), patch("agent.graph.wandb.run", None):
        result = build_graph().invoke(state)
    assert result["plan"][0]["dependency_rechecks"] == 1
    assert result["plan"][0]["resolved_answer"] == "The Young Americans"
    assert result["plan"][1]["result"] == "Independent work"
    assert result["plan"][2]["resolved_answer"] == "17"
    assert result["plan"][2]["status"] == "completed"
    assert result["current_task_index"] == 3
    search.assert_called_once()
    code.assert_called_once()
    # State snapshots passed into the graph remain intact.
    assert state["plan"][0]["status"] == "failed"


def test_exhausted_prerequisite_does_not_retry_dependent_calculation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = _initial_state("Question", "test")
    state.update(plan=[task(1, dependency_rechecks=1, status="failed", resolved_answer=""),
                       task(2, "code_interpreter", [1], evidence_gap="Missing film title", resolved_answer="")],
                 current_task_index=1, dependency_to_retry=0)
    with patch("agent.nodes.critic._build_llm") as model, patch("agent.nodes.critic._log_critic_evaluation"):
        result = critic_node(state)
    model.assert_not_called()
    assert result["should_retry"] is False
    assert result["current_task_index"] == 2
    assert result["plan"][1]["status"] == "failed"


def test_rejected_answer_stays_unresolved_after_retries(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = _initial_state("Question", "test")
    state.update(plan=[task(1, retries=MAX_RETRIES, result="Related fact", resolved_answer="Related fact", confidence=0.8)],
                 current_task_index=0)
    with patch("agent.nodes.critic._build_llm"), patch(
        "agent.nodes.critic._invoke_critic", return_value=CriticOutput(overall_score=0.2, verbal_feedback="Wrong entity")
    ), patch("agent.nodes.critic._log_critic_evaluation"):
        result = critic_node(state)
    assert result["plan"][0]["status"] == "failed"
    assert result["plan"][0]["resolved_answer"] == ""
    assert result["plan"][0]["confidence"] == 0


def test_graph_budget_routes_to_partial_report():
    state = _initial_state("Question", "test")
    state.update(iteration_count=MAX_ITERATIONS - 2, should_retry=True)
    assert route_after_critic(state) == "report"


def test_blocked_output_is_reported_as_unfinished():
    from agent.nodes.report import report_node
    from eval.ablation import _question_summary

    state = _initial_state("Question", "test")
    state.update(plan=[task(1, status="failed", result="Blocked: missing title", confidence=0)], current_task_index=1)
    result = report_node(state)
    assert "Completed 0 of 1 tasks" in result["final_report"]
    summary = _question_summary(result, {"id": "q1", "question": "Question"})
    assert summary["completed_tasks"] == 0
    assert summary["output_tasks"] == 1


def test_research_retry_extracts_source_passages():
    from agent.tools.search import run_web_search
    url = "https://en.wikipedia.org/wiki/Example"
    snippet = "A list of film credits. " * 30
    passage = "His first film was The Young Americans in 1993. " * 30
    client = Mock()
    client.search.return_value = {"results": [{"url": url, "title": "Filmography", "content": snippet}]}
    client.extract.return_value = {"results": [{"url": url, "raw_content": passage}]}
    with patch("agent.tools.search.TavilyClient", return_value=client), patch(
        "agent.tools.search.os.getenv", return_value="test-key"
    ):
        result = run_web_search("first movie David Arnold", fetch_sources=True)
    assert passage.strip() in result.content
    assert passage.strip() in result.source_content[url]
    client.extract.assert_called_once()
    assert client.extract.call_args.kwargs["urls"] == [url]
    assert client.extract.call_args.kwargs["chunks_per_source"] == 3


def test_page_extraction_failure_preserves_search_evidence():
    from agent.tools.search import run_web_search
    url = "https://en.wikipedia.org/wiki/Example"
    snippet = "A useful original search passage. " * 30
    client = Mock()
    client.search.return_value = {"results": [{"url": url, "title": "Example", "content": snippet}]}
    client.extract.side_effect = RuntimeError("Page unavailable")
    with patch("agent.tools.search.TavilyClient", return_value=client), patch(
        "agent.tools.search.os.getenv", return_value="test-key"
    ):
        result = run_web_search("a question", fetch_sources=True)
    assert snippet.strip() in result.content
    assert result.raw_confidence > 0


def test_fact_retry_retains_short_relevant_passage():
    from agent.tools.search import run_web_search
    url = "https://en.wikipedia.org/wiki/Example"
    passage = "His first film was The Young Americans."
    client = Mock()
    client.search.return_value = {"results": [{"url": url, "title": "Biography", "content": passage}]}
    client.extract.return_value = {"results": []}
    with patch("agent.tools.search.TavilyClient", return_value=client), patch(
        "agent.tools.search.os.getenv", return_value="test-key"
    ):
        result = run_web_search("first movie", fetch_sources=True)
    assert passage in result.content
    assert result.sources == [url]


def test_quote_matching_recovers_source_punctuation_without_changing_numbers():
    from agent.evidence import _matched_quote
    original = 'They made their debuts with The Young Americans "The Young Americans (film)").'
    proposed = 'They made their debuts with The Young Americans "The Young Americans (film)".'
    matched = _matched_quote(proposed, original)
    assert matched and matched in original
    assert _matched_quote("The difference is 17", "The difference is 117") == ""
    assert _matched_quote("The difference is -5", "The difference is 5") == ""
    assert _matched_quote("The ratio is 2.5", "The ratio is 25") == ""


def test_calculation_fetches_missing_fact_on_own_retry(monkeypatch):
    from agent.nodes.executor import executor_node
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state = _initial_state("How many letters in the film title?", "test")
    state.update(plan=[task(1, "code_interpreter", retries=1, evidence_gap="Find the title", resolved_answer="")],
                 current_task_index=0, critic_feedback="Find the title")
    additional = ToolResult(content="Answer: The Young Americans", sources=["source.md"], resolved_answer="The Young Americans", raw_confidence=0.8)
    def prepare(current, *args):
        assert "The Young Americans" in current["supplemental_evidence"]
        return GeneratedCode(code="print(17)"), [], True
    with patch("agent.nodes.executor._reformulate_retrieval_query", return_value="first film"), patch(
        "agent.nodes.executor.run_web_search", return_value=ToolResult(content="page")
    ) as search, patch("agent.nodes.executor.resolve_evidence", return_value=additional), patch(
        "agent.nodes.executor._prepare_code_input", side_effect=prepare
    ), patch("agent.nodes.executor.run_code", return_value=ToolResult(content="17", raw_confidence=1, tool_name="code_interpreter")):
        result = executor_node(state)
    search.assert_called_once_with("first film", fetch_sources=True)
    assert result["plan"][0]["resolved_answer"] == "17"
    assert result["plan"][0]["sources"] == ["source.md"]


def test_resolver_keeps_intermediate_lookup_separate_from_final_calculation():
    raw = ToolResult(content="His first film was The Young Americans.", sources=["source.md"],
                     tool_name="document_retrieval", raw_confidence=0.8)
    model = evidence_model(ResearchAnswer(answer="The Young Americans", evidence=[
        EvidenceQuote(source="source.md", quote=raw.content)]))
    with patch("agent.evidence.ChatOpenAI", return_value=model):
        result = resolve_evidence(raw, "Determine the title of his first film", "Count the letters in that title")
    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "Answer ONLY the subtask" in messages[0].content
    assert result.resolved_answer == "The Young Americans"
    assert result.evidence_gap == ""
