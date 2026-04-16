"""Unit tests for planner, executor, and critic nodes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.exceptions import OutputParserException

import agent.graph as graph_module
import agent.nodes.critic as critic_module
import agent.nodes.planner as planner_module
import agent.tools.search as search_module
from agent.nodes.critic import CriticOutput, critic_node
from agent.nodes.executor import executor_node
from agent.nodes.planner import PlannerResponse, planner_node
from agent.nodes.report import report_node
from agent.state import AgentState, MAX_ITERATIONS
from agent.tools import ToolResult
from agent.tools.search import run_web_search


def make_state(
    *,
    plan: list[dict] | None = None,
    current_task_index: int = 0,
    tool_history: list[str] | None = None,
    iteration_count: int = 0,
    error: str | None = None,
) -> AgentState:
    """Build a complete AgentState fixture for node unit tests."""

    return {
        "research_question": "What are the latest advances in LLM memory architectures?",
        "session_id": "test-session",
        "plan": plan or [],
        "current_task_index": current_task_index,
        "tool_history": tool_history or [],
        "memory_context": "",
        "critic_score": 0.0,
        "critic_feedback": "",
        "should_retry": False,
        "final_report": "",
        "wandb_task_rows": [],
        "error": error,
        "iteration_count": iteration_count,
    }


def make_plan_task(
    *,
    task_id: int = 1,
    description: str = "Search for recent work on memory architectures.",
    tool: str = "web_search",
    status: str = "pending",
    result: str = "",
    sources: list[str] | None = None,
    confidence: float = 0.0,
    retries: int = 0,
) -> dict:
    """Build a plan task matching the shared state contract."""

    return {
        "id": task_id,
        "description": description,
        "tool": tool,
        "dependencies": [],
        "status": status,
        "result": result,
        "sources": sources or [],
        "confidence": confidence,
        "retries": retries,
    }


def make_planner_response() -> PlannerResponse:
    """Return a valid planner response fixture."""

    return PlannerResponse(
        tasks=[
            {
                "id": 1,
                "description": "Search for recent papers and announcements on LLM memory architectures.",
                "tool": "web_search",
                "dependencies": [],
            },
            {
                "id": 2,
                "description": "Retrieve indexed notes or papers about memory modules and benchmarks.",
                "tool": "document_retrieval",
                "dependencies": [1],
            },
            {
                "id": 3,
                "description": "Compare approaches, datasets, and tradeoffs using structured analysis.",
                "tool": "synthesis",
                "dependencies": [1, 2],
            },
        ]
    )


def make_critic_output(score: float, *, should_retry: bool = False) -> CriticOutput:
    """Return a structured critic response fixture."""

    return CriticOutput(
        overall_score=score,
        dimension_scores={
            "factual_grounding": score,
            "source_reliability": score,
            "completeness": score,
            "specificity": score,
        },
        verbal_feedback=(
            "The answer is too vague, under-sourced, and does not clearly connect the result "
            "to the sub-task requirements."
        ),
        should_retry=should_retry,
    )


def test_planner_node_populates_plan_and_sets_current_task_index() -> None:
    """Planner should normalize a valid structured response into plan state."""

    state = make_state(current_task_index=-1)
    mocked_response = make_planner_response()

    with patch("agent.nodes.planner.load_dotenv"), patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch(
        "agent.nodes.planner._retrieval_index_is_populated",
        return_value=True,
    ), patch("agent.nodes.planner._build_llm", return_value=object()), patch(
        "agent.nodes.planner._invoke_planner",
        return_value=mocked_response,
    ) as invoke_mock, patch("agent.nodes.planner._log_plan_to_wandb") as wandb_mock:
        updated_state = planner_node(state)

    assert updated_state["current_task_index"] == 0
    assert len(updated_state["plan"]) == 3
    assert updated_state["plan"][0]["status"] == "pending"
    assert updated_state["plan"][0]["sources"] == []
    assert updated_state["plan"][1]["dependencies"] == [1]
    assert updated_state["plan"][2]["tool"] == "synthesis"
    invoke_mock.assert_called_once()
    wandb_mock.assert_called_once_with(updated_state["plan"], state["session_id"])


def test_planner_node_retries_with_stricter_prompt_after_parse_failure() -> None:
    """Planner should retry once if the first structured parse fails."""

    state = make_state(current_task_index=-1)
    mocked_response = make_planner_response()

    with patch("agent.nodes.planner.load_dotenv"), patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch(
        "agent.nodes.planner._retrieval_index_is_populated",
        return_value=True,
    ), patch("agent.nodes.planner._build_llm", return_value=object()), patch(
        "agent.nodes.planner._invoke_planner",
        side_effect=[OutputParserException("bad json"), mocked_response],
    ) as invoke_mock, patch("agent.nodes.planner._log_plan_to_wandb"):
        updated_state = planner_node(state)

    assert len(updated_state["plan"]) == 3
    assert invoke_mock.call_count == 2


def test_planner_node_raises_value_error_after_second_parse_failure() -> None:
    """Planner should raise when both structured-output attempts fail."""

    state = make_state(current_task_index=-1)

    with patch("agent.nodes.planner.load_dotenv"), patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch(
        "agent.nodes.planner._retrieval_index_is_populated",
        return_value=True,
    ), patch("agent.nodes.planner._build_llm", return_value=object()), patch(
        "agent.nodes.planner._invoke_planner",
        side_effect=[OutputParserException("first"), OutputParserException("second")],
    ), patch("agent.nodes.planner._log_plan_to_wandb"):
        with pytest.raises(ValueError, match="Planner failed to return a valid structured plan."):
            planner_node(state)


def test_planner_node_remaps_document_retrieval_when_index_is_unavailable() -> None:
    """Planner should avoid retrieval tasks when no local index is populated."""

    state = make_state(current_task_index=-1)
    mocked_response = PlannerResponse(
        tasks=[
            {
                "id": 1,
                "description": "Search for recent papers on long-context memory.",
                "tool": "web_search",
                "dependencies": [],
            },
            {
                "id": 2,
                "description": "Review indexed notes about long-context memory methods.",
                "tool": "document_retrieval",
                "dependencies": [1],
            },
            {
                "id": 3,
                "description": "Summarize the strongest recent approaches.",
                "tool": "document_retrieval",
                "dependencies": [1, 2],
            },
        ]
    )

    with patch("agent.nodes.planner.load_dotenv"), patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch(
        "agent.nodes.planner._retrieval_index_is_populated",
        return_value=False,
    ), patch("agent.nodes.planner._build_llm", return_value=object()), patch(
        "agent.nodes.planner._invoke_planner",
        return_value=mocked_response,
    ), patch("agent.nodes.planner._log_plan_to_wandb"):
        updated_state = planner_node(state)

    assert updated_state["plan"][1]["tool"] == "web_search"
    assert updated_state["plan"][2]["tool"] == "web_search"


def test_planner_node_remaps_legacy_analysis_code_tasks_to_synthesis() -> None:
    """Planner should normalize compare/analyze tasks onto the synthesis tool."""

    state = make_state(current_task_index=-1)
    mocked_response = PlannerResponse(
        tasks=[
            {
                "id": 1,
                "description": "Search for recent papers on long-context memory.",
                "tool": "web_search",
                "dependencies": [],
            },
            {
                "id": 2,
                "description": "Analyze and compare the strongest approaches from the gathered evidence.",
                "tool": "code_interpreter",
                "dependencies": [1],
            },
            {
                "id": 3,
                "description": "Summarize findings and identify the best overall method.",
                "tool": "code_interpreter",
                "dependencies": [1, 2],
            },
        ]
    )

    with patch("agent.nodes.planner.load_dotenv"), patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch(
        "agent.nodes.planner._retrieval_index_is_populated",
        return_value=True,
    ), patch("agent.nodes.planner._build_llm", return_value=object()), patch(
        "agent.nodes.planner._invoke_planner",
        return_value=mocked_response,
    ), patch("agent.nodes.planner._log_plan_to_wandb"):
        updated_state = planner_node(state)

    assert updated_state["plan"][1]["tool"] == "synthesis"
    assert updated_state["plan"][2]["tool"] == "synthesis"


def test_planner_build_llm_uses_env_configured_model() -> None:
    """Planner should read the chat model name from the environment."""

    with patch("agent.nodes.planner.ChatOpenAI") as chat_model, patch(
        "agent.nodes.planner.os.getenv",
        side_effect=lambda key, default=None: "test-model" if key == "OPENAI_CHAT_MODEL" else default,
    ):
        planner_module._build_llm("test-key")

    chat_model.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        temperature=0,
    )


def test_executor_node_dispatches_to_task_tool_and_updates_plan() -> None:
    """Executor should call the task tool and persist the result into the plan."""

    plan = [make_plan_task()]
    state = make_state(plan=plan, current_task_index=0, tool_history=[], iteration_count=4)
    tool_result = ToolResult(
        content="Retrieved summary of recent memory architecture work.",
        sources=["https://example.com/paper"],
        raw_confidence=0.82,
        tool_name="web_search",
    )

    with patch("agent.nodes.executor.run_web_search", return_value=tool_result) as search_mock:
        updated_state = executor_node(state)

    assert updated_state["plan"][0]["result"] == tool_result.content
    assert updated_state["plan"][0]["sources"] == tool_result.sources
    assert updated_state["plan"][0]["confidence"] == 0.82
    assert updated_state["plan"][0]["status"] == "completed"
    assert updated_state["tool_history"] == ["web_search"]
    assert updated_state["iteration_count"] == 5
    assert updated_state["error"] is None
    search_mock.assert_called_once_with(plan[0]["description"])


def test_executor_node_switches_tools_after_repeated_use() -> None:
    """Executor should avoid calling the same tool more than twice for one task."""

    plan = [make_plan_task(tool="web_search")]
    state = make_state(
        plan=plan,
        current_task_index=0,
        tool_history=["web_search", "web_search"],
    )
    retrieval_result = ToolResult(
        content="Retrieved internal benchmark notes.",
        sources=["local://chunk/0"],
        raw_confidence=0.73,
        tool_name="document_retrieval",
    )

    with patch("agent.nodes.executor.run_retrieval", return_value=retrieval_result) as retrieval_mock, patch(
        "agent.nodes.executor.run_web_search"
    ) as search_mock:
        updated_state = executor_node(state)

    assert updated_state["tool_history"][-1] == "document_retrieval"
    assert updated_state["plan"][0]["result"] == retrieval_result.content
    retrieval_mock.assert_called_once_with(plan[0]["description"], top_k=3)
    search_mock.assert_not_called()


def test_executor_node_marks_task_failed_when_tool_returns_zero_confidence() -> None:
    """Executor should mark the task as failed and set an error message on tool failure."""

    plan = [make_plan_task(tool="document_retrieval")]
    state = make_state(plan=plan, current_task_index=0, tool_history=[], iteration_count=1)
    tool_result = ToolResult(
        content="",
        sources=[],
        raw_confidence=0.0,
        tool_name="document_retrieval",
    )

    with patch("agent.nodes.executor.run_retrieval", return_value=tool_result):
        updated_state = executor_node(state)

    assert updated_state["plan"][0]["status"] == "failed"
    assert updated_state["plan"][0]["confidence"] == 0.0
    assert updated_state["error"] == "Tool execution failed for task 1 with document_retrieval."
    assert updated_state["iteration_count"] == 2


def test_executor_node_rewrites_retry_retrieval_query_with_critic_feedback() -> None:
    """Executor should reformulate retrieval queries on retry and widen top-k."""

    plan = [
        make_plan_task(
            tool="document_retrieval",
            description="Retrieve and review indexed documents about hallucination reduction.",
            retries=1,
        )
    ]
    state = make_state(plan=plan, current_task_index=0)
    state["critic_feedback"] = "The prior retrieval was too broad and missed concrete evaluation metrics."
    tool_result = ToolResult(
        content="Retrieved targeted evidence on faithfulness and RAGAS.",
        sources=["data/seed_docs/agent_reliability_evaluation.md"],
        raw_confidence=0.74,
        tool_name="document_retrieval",
    )

    with patch(
        "agent.nodes.executor._reformulate_retrieval_query",
        return_value="hallucination reduction faithfulness metrics RAGAS",
    ) as reformulate_mock, patch(
        "agent.nodes.executor.run_retrieval",
        return_value=tool_result,
    ) as retrieval_mock:
        updated_state = executor_node(state)

    reformulate_mock.assert_called_once_with(
        "Retrieve and review indexed documents about hallucination reduction.",
        "The prior retrieval was too broad and missed concrete evaluation metrics.",
    )
    retrieval_mock.assert_called_once_with(
        "hallucination reduction faithfulness metrics RAGAS",
        top_k=5,
    )
    assert updated_state["plan"][0]["confidence"] == 0.74
    assert updated_state["plan"][0]["status"] == "completed"


def test_executor_node_uses_original_query_for_first_pass_retrieval() -> None:
    """Executor should keep the original retrieval query on first attempt."""

    plan = [
        make_plan_task(
            tool="document_retrieval",
            description="Retrieve indexed notes about long-context memory.",
            retries=0,
        )
    ]
    state = make_state(plan=plan, current_task_index=0)
    tool_result = ToolResult(
        content="Retrieved memory architecture notes.",
        sources=["data/seed_docs/long_context_memory_overview.md"],
        raw_confidence=0.66,
        tool_name="document_retrieval",
    )

    with patch("agent.nodes.executor._reformulate_retrieval_query") as reformulate_mock, patch(
        "agent.nodes.executor.run_retrieval",
        return_value=tool_result,
    ) as retrieval_mock:
        updated_state = executor_node(state)

    reformulate_mock.assert_not_called()
    retrieval_mock.assert_called_once_with(
        "Retrieve indexed notes about long-context memory.",
        top_k=3,
    )
    assert updated_state["plan"][0]["confidence"] == 0.66


def test_executor_node_routes_analysis_tasks_to_synthesis() -> None:
    """Executor should use direct synthesis for analytical non-Python tasks."""

    plan = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            tool="web_search",
            status="completed",
            result="Found two strong approaches: retrieval plus summarization.",
            sources=["https://example.com/a"],
            confidence=0.9,
        ),
        make_plan_task(
            task_id=2,
            description="Analyze and compare the findings.",
            tool="code_interpreter",
            status="pending",
            retries=0,
        ),
    ]
    plan[1]["dependencies"] = [1]
    state = make_state(plan=plan, current_task_index=1, tool_history=[], iteration_count=3)

    def _fake_run_synthesis(task: dict, plan: list[dict], current_index: int) -> ToolResult:
        assert task["description"] == "Analyze and compare the findings."
        assert "retrieval plus summarization" in plan[0]["result"]
        assert current_index == 1
        return ToolResult(
            content="Structured synthesis says retrieval plus summarization looks strongest.",
            sources=["https://example.com/a"],
            raw_confidence=0.8,
            tool_name="synthesis",
        )

    with patch("agent.nodes.executor._run_synthesis", side_effect=_fake_run_synthesis) as synthesis_mock, patch(
        "agent.nodes.executor.run_code"
    ) as run_code_mock:
        updated_state = executor_node(state)

    assert updated_state["plan"][1]["status"] == "completed"
    assert updated_state["plan"][1]["sources"] == ["https://example.com/a"]
    assert "Structured synthesis" in updated_state["plan"][1]["result"]
    assert updated_state["plan"][1]["confidence"] == 0.8
    assert updated_state["tool_history"] == ["synthesis"]
    synthesis_mock.assert_called_once()
    run_code_mock.assert_not_called()


def test_executor_node_supports_explicit_synthesis_tool() -> None:
    """Executor should call the synthesis path directly when the tool is synthesis."""

    plan = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            tool="web_search",
            status="completed",
            result="Found approaches using retrieval plus summarization.",
            sources=["https://example.com/a"],
            confidence=0.9,
        ),
        make_plan_task(
            task_id=2,
            description=(
                "Summarize findings and identify the best recent methods for long-context memory in LLMs."
            ),
            tool="synthesis",
            status="pending",
        ),
    ]
    plan[1]["dependencies"] = [1]
    state = make_state(plan=plan, current_task_index=1)

    def _fake_run_synthesis(task: dict, plan: list[dict], current_index: int) -> ToolResult:
        assert task["tool"] == "synthesis"
        assert "Found approaches using retrieval plus summarization." in plan[0]["result"]
        return ToolResult(
            content="Direct synthesis from prior results.",
            sources=["https://example.com/a"],
            raw_confidence=0.8,
            tool_name="synthesis",
        )

    with patch("agent.nodes.executor._run_synthesis", side_effect=_fake_run_synthesis) as synthesis_mock:
        updated_state = executor_node(state)

    assert updated_state["plan"][1]["status"] == "completed"
    assert updated_state["plan"][1]["sources"] == ["https://example.com/a"]
    assert updated_state["plan"][1]["confidence"] == 0.8
    synthesis_mock.assert_called_once()


def test_synthesis_prompt_includes_real_sources_and_inline_citation_instruction() -> None:
    """Synthesis prompts should pass through real sources and require source-based citations."""

    supporting_tasks = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            tool="web_search",
            status="completed",
            result="Reflexion improves AlfWorld performance and HotpotQA reasoning quality.",
            sources=[
                "https://arxiv.org/abs/2303.11366",
                "data/seed_docs/reflexion_verbal_reinforcement.md",
            ],
        )
    ]
    synthesis_task = make_plan_task(
        task_id=2,
        description="Analyze and compare the strongest approaches.",
        tool="synthesis",
    )

    prompt, aggregated_sources = graph_module.executor_node.__globals__["_build_synthesis_prompt"](
        synthesis_task,
        supporting_tasks,
    )

    assert "Task 1 found the following from sources [" in prompt
    assert "https://arxiv.org/abs/2303.11366" in prompt
    assert "data/seed_docs/reflexion_verbal_reinforcement.md" in prompt
    assert "cite the original source URL or document name, not the task number" in prompt
    assert "Write citations as (Source: [url or filename]) inline." in prompt
    assert aggregated_sources == [
        "https://arxiv.org/abs/2303.11366",
        "data/seed_docs/reflexion_verbal_reinforcement.md",
    ]


def test_executor_node_injects_critic_feedback_into_retry_fallback_prompt() -> None:
    """Executor should use critic feedback to deepen non-synthesis fallback analysis on retries."""

    plan = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            tool="web_search",
            status="completed",
            result="Found retrieval-augmented generation and self-correction loops.",
            sources=["https://example.com/a"],
            confidence=0.9,
        ),
        make_plan_task(
            task_id=2,
            description="Write a plain-language Python-style debugging plan for the evidence pipeline.",
            tool="code_interpreter",
            status="retrying",
            retries=1,
        ),
    ]
    plan[1]["dependencies"] = [1]
    state = make_state(
        plan=plan,
        current_task_index=1,
    )
    state["critic_feedback"] = "The prior comparison was too vague and lacked concrete examples."

    def _fake_run_code(code: str) -> ToolResult:
        assert "The previous analysis was rejected because:" in code
        assert "too vague and lacked concrete examples" in code
        assert "Structured conclusions:" in code
        return ToolResult(
            content="A deeper structured comparison with examples from the prior evidence.",
            sources=[],
            raw_confidence=1.0,
            tool_name="code_interpreter",
        )

    with patch("agent.nodes.executor.run_code", side_effect=_fake_run_code) as run_code_mock:
        updated_state = executor_node(state)

    assert updated_state["plan"][1]["confidence"] == 0.5
    assert updated_state["plan"][1]["sources"] == ["https://example.com/a"]
    run_code_mock.assert_called_once()


def test_executor_node_keeps_full_confidence_for_real_python_code() -> None:
    """Executor should preserve full confidence for explicit Python code tasks."""

    python_task = make_plan_task(
        description="print('hello from real python')",
        tool="code_interpreter",
    )
    state = make_state(plan=[python_task], current_task_index=0)

    with patch(
        "agent.nodes.executor.run_code",
        return_value=ToolResult(
            content="hello from real python",
            sources=[],
            raw_confidence=0.2,
            tool_name="code_interpreter",
        ),
    ) as run_code_mock:
        updated_state = executor_node(state)

    assert updated_state["plan"][0]["confidence"] == 1.0
    assert updated_state["plan"][0]["status"] == "completed"
    run_code_mock.assert_called_once_with("print('hello from real python')")


def test_graph_logging_accumulates_task_rows_and_applies_critic_score() -> None:
    """Graph wrappers should accumulate executor rows and fill in critic scores later."""

    plan = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            tool="web_search",
            status="completed",
            result="Found two approaches.",
            sources=["https://example.com/a"],
            confidence=0.8,
            retries=1,
        )
    ]
    state = make_state(plan=plan, current_task_index=0)

    executor_state = {
        **state,
        "plan": plan,
        "iteration_count": 1,
    }
    with patch("agent.graph.executor_node", return_value=executor_state):
        updated_state = graph_module._executor_with_logging(state)

    assert updated_state["wandb_task_rows"] == [
        {
            "task_id": 1,
            "description": "Search for recent papers.",
            "tool": "web_search",
            "confidence": 0.8,
            "retries": 1,
            "status": "completed",
            "critic_score": None,
        }
    ]

    critic_state = {
        **updated_state,
        "critic_score": 0.6,
        "iteration_count": 2,
    }
    with patch("agent.graph.critic_node", return_value=critic_state), patch("agent.graph.wandb.run", None):
        scored_state = graph_module._critic_with_logging(updated_state)

    assert scored_state["wandb_task_rows"][0]["critic_score"] == 0.6


def test_run_web_search_filters_low_quality_domains_and_formats_sources() -> None:
    """Search should keep stronger sources and format evidence clearly."""

    long_unimem_content = (
        "A unified framework for memory augmentation in long-context modeling by Wei Xu. "
        "The paper describes how memory modules can be separated into short-term and long-term stores, "
        "how retrieval conditions the current generation step, and how evaluation depends on preserving "
        "high-recall evidence over long horizons. It also compares retrieval, summarization, and hybrid "
        "memory strategies across multiple settings where prompt-only baselines become unstable."
    )
    long_nature_content = (
        "A peer-reviewed article on memory-centric architectures. "
        "The study analyzes how explicit memory interfaces, retrieval quality, and long-horizon control "
        "affect factual reliability. It reports that architectures using structured external memory retain "
        "more relevant evidence under distracting context and provide clearer grounding for downstream "
        "reasoning than prompt-only systems operating near their context limits."
    )
    raw_fallback_content = (
        "This raw content field includes the full article text needed for the key finding. "
        "It explains the benchmark setup, the compared systems, and the main conclusions in enough detail "
        "to survive the minimum-length filter. The result is intentionally longer than three hundred "
        "characters so that the search tool keeps it after filtering instead of discarding it as too short."
    )

    fake_response = {
        "answer": "Recent work highlights memory augmentation and hierarchical memory.",
        "results": [
            {
                "title": "LinkedIn recap",
                "url": "https://www.linkedin.com/posts/example-memory-post",
                "content": "A generic social post about memory.",
                "score": 0.95,
            },
            {
                "title": '"UniMem" paper',
                "url": "https://openreview.net/forum?id=gQAEGSGVnN",
                "content": long_unimem_content,
                "score": 0.72,
            },
            {
                "title": "Nature paper",
                "url": "https://www.nature.com/articles/s41467-026-68603-0",
                "content": long_nature_content,
                "score": 0.61,
            },
            {
                "title": "Fallback raw content paper",
                "url": "https://arxiv.org/abs/2602.00015",
                "content": "",
                "raw_content": raw_fallback_content,
                "score": 0.7,
            },
        ],
    }

    client = type("FakeClient", (), {"search": lambda self, *args, **kwargs: fake_response})()

    with patch("agent.tools.search.load_dotenv"), patch(
        "agent.tools.search.os.getenv",
        side_effect=lambda key, default=None: "test-tavily-key" if key == "TAVILY_API_KEY" else default,
    ), patch("agent.tools.search.TavilyClient", return_value=client):
        result = run_web_search("long-context memory in llm agents")

    assert "Search synthesis:" not in result.content
    assert '## "UniMem" paper' in result.content
    assert "## Nature paper" in result.content
    assert "## Fallback raw content paper" in result.content
    assert "URL: https://openreview.net/forum?id=gQAEGSGVnN" in result.content
    assert "Key finding:" in result.content
    assert raw_fallback_content in result.content
    assert "linkedin.com" not in result.content
    assert result.sources == [
        "https://www.nature.com/articles/s41467-026-68603-0",
        "https://arxiv.org/abs/2602.00015",
        "https://openreview.net/forum?id=gQAEGSGVnN",
    ]
    assert result.raw_confidence == pytest.approx((0.8 + 0.8 + 0.7) / 3, rel=1e-9)


def test_result_weight_gives_bonus_to_high_trust_domains() -> None:
    """Search confidence should favor higher-trust academic domains."""

    arxiv_result = {
        "url": "https://arxiv.org/abs/2602.00015",
        "title": '"G-MemLLM" paper',
        "content": "A research paper on gated memory by Jane Doe.",
    }
    social_result = {
        "url": "https://www.linkedin.com/posts/example",
        "content": "A social summary of memory systems.",
    }

    assert search_module._result_weight(arxiv_result) == 0.85
    assert search_module._is_allowed_result(social_result) is False


def test_result_weight_penalizes_future_years() -> None:
    """Search confidence should drop for speculative future-dated results."""

    future_result = {
        "url": "https://openreview.net/forum?id=test",
        "title": '"Future Memory System"',
        "content": "John Smith reports a benchmark result expected in 2026.",
    }

    assert pytest.approx(search_module._result_weight(future_result), rel=1e-9) == 0.7


def test_result_weight_uses_news_blog_tier_without_random_bonus() -> None:
    """News and blog sources should use the fixed lower-confidence tier."""

    blog_result = {
        "url": "https://developer.nvidia.com/blog/example",
        "title": "NVIDIA blog summary",
        "content": "A practical overview of memory systems.",
    }

    assert search_module._result_weight(blog_result) == 0.55


def test_is_allowed_result_discards_short_findings() -> None:
    """Search should discard results whose best finding text is too short."""

    short_result = {
        "url": "https://arxiv.org/abs/2602.00015",
        "content": "Too short.",
        "raw_content": "",
    }

    assert search_module._is_allowed_result(short_result) is False


def test_critic_node_retries_low_scoring_result_and_resets_tool_history() -> None:
    """Critic should increment retries and stay on the same task for low scores."""

    plan = [
        make_plan_task(
            status="completed",
            result="Memory approaches are improving.",
            sources=["https://example.com/blog"],
            confidence=0.4,
            retries=0,
        )
    ]
    state = make_state(
        plan=plan,
        current_task_index=0,
        tool_history=["web_search", "web_search"],
        iteration_count=2,
    )

    with patch("agent.nodes.critic.load_dotenv"), patch(
        "agent.nodes.critic.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch("agent.nodes.critic._build_llm", return_value=object()), patch(
        "agent.nodes.critic._invoke_critic",
        return_value=make_critic_output(0.5),
    ), patch("agent.nodes.critic._log_critic_evaluation") as log_mock:
        updated_state = critic_node(state)

    assert updated_state["critic_score"] == 0.5
    assert updated_state["should_retry"] is True
    assert updated_state["current_task_index"] == 0
    assert updated_state["plan"][0]["retries"] == 1
    assert updated_state["plan"][0]["status"] == "retrying"
    assert updated_state["tool_history"] == []
    assert "too vague" in updated_state["critic_feedback"]
    log_mock.assert_called_once()


def test_critic_node_advances_to_next_task_when_score_is_high() -> None:
    """Critic should advance when the result clears the retry threshold."""

    plan = [
        make_plan_task(
            task_id=1,
            status="completed",
            result="Detailed answer with citations.",
            sources=["https://example.com/paper"],
            confidence=0.9,
            retries=1,
        ),
        make_plan_task(task_id=2, tool="document_retrieval"),
    ]
    state = make_state(
        plan=plan,
        current_task_index=0,
        tool_history=["web_search"],
        iteration_count=3,
    )

    with patch("agent.nodes.critic.load_dotenv"), patch(
        "agent.nodes.critic.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch("agent.nodes.critic._build_llm", return_value=object()), patch(
        "agent.nodes.critic._invoke_critic",
        return_value=make_critic_output(0.9),
    ), patch("agent.nodes.critic._log_critic_evaluation"):
        updated_state = critic_node(state)

    assert updated_state["should_retry"] is False
    assert updated_state["current_task_index"] == 1
    assert updated_state["plan"][0]["status"] == "completed"
    assert updated_state["tool_history"] == ["web_search"]
    assert updated_state["iteration_count"] == 4


def test_critic_node_circuit_breaker_disables_retry_after_max_iterations() -> None:
    """Critic should not retry once the graph has exceeded the iteration limit."""

    plan = [
        make_plan_task(
            status="completed",
            result="Thin answer with weak grounding.",
            sources=["https://example.com/post"],
            confidence=0.2,
            retries=0,
        )
    ]
    state = make_state(
        plan=plan,
        current_task_index=0,
        tool_history=["web_search"],
        iteration_count=MAX_ITERATIONS + 1,
    )

    with patch("agent.nodes.critic.load_dotenv"), patch(
        "agent.nodes.critic.os.getenv",
        side_effect=lambda key, default=None: "test-openai-key" if key == "OPENAI_API_KEY" else default,
    ), patch("agent.nodes.critic._build_llm", return_value=object()), patch(
        "agent.nodes.critic._invoke_critic",
        return_value=make_critic_output(0.2),
    ), patch("agent.nodes.critic._log_critic_evaluation"):
        updated_state = critic_node(state)

    assert updated_state["should_retry"] is False
    assert updated_state["current_task_index"] == 1
    assert updated_state["plan"][0]["retries"] == 0
    assert updated_state["plan"][0]["status"] == "completed"


def test_critic_build_llm_uses_env_configured_model() -> None:
    """Critic should read the chat model name from the environment."""

    with patch("agent.nodes.critic.ChatOpenAI") as chat_model, patch(
        "agent.nodes.critic.os.getenv",
        side_effect=lambda key, default=None: "critic-model" if key == "OPENAI_CHAT_MODEL" else default,
    ):
        critic_module._build_llm("test-key")

    chat_model.assert_called_once_with(
        model="critic-model",
        api_key="test-key",
        temperature=0,
    )


def test_report_node_builds_clean_summary_with_sources() -> None:
    """Report node should produce a structured summary and deduplicated sources."""

    plan = [
        make_plan_task(
            task_id=1,
            description="Search for recent papers.",
            status="completed",
            result="Found benchmark and memory papers.",
            sources=["https://example.com/a", "https://example.com/b"],
            confidence=0.8,
        ),
        make_plan_task(
            task_id=2,
            description="Compare approaches.",
            tool="code_interpreter",
            status="completed",
            result="Hybrid retrieval plus summarization performed best.",
            sources=["https://example.com/b", "https://example.com/c"],
            confidence=0.9,
        ),
    ]
    state = make_state(plan=plan, current_task_index=2)
    state["critic_feedback"] = "Need stronger citations for one claim."

    updated_state = report_node(state)

    assert "Research Question:" in updated_state["final_report"]
    assert "Summary: Completed 2 of 2 tasks." in updated_state["final_report"]
    assert "Task Findings:" in updated_state["final_report"]
    assert "All Sources:" in updated_state["final_report"]
    assert updated_state["final_report"].count("https://example.com/b") == 1
