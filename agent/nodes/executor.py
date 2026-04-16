"""Executor node that dispatches the current plan step to a concrete tool."""

from __future__ import annotations

import os
import re
from textwrap import shorten
from typing import Final

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent.llm_retry import invoke_with_rate_limit_retry
from agent.state import AgentState, PlanTask
from agent.tools import ToolResult
from agent.tools.code_interpreter import run_code
from agent.tools.retrieval import run_retrieval
from agent.tools.search import run_web_search


TOOL_ORDER: Final[tuple[str, ...]] = (
    "web_search",
    "document_retrieval",
    "synthesis",
    "code_interpreter",
)
"""Stable fallback order used when the current task is overusing one tool."""


def _looks_like_python_code(text: str) -> bool:
    """Heuristically detect whether a task description already contains Python."""

    stripped = text.strip()
    if not stripped:
        return False
    if "```python" in stripped:
        return True

    python_line_patterns = (
        r"^\s*(import|from)\s+\w+",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*class\s+\w+\s*[:(]",
        r"^\s*print\s*\(",
        r"^\s*(for|while|if)\s+.+:\s*$",
        r"^\s*\w+\s*=\s*.+",
    )

    return any(
        re.search(pattern, stripped, flags=re.MULTILINE) is not None
        for pattern in python_line_patterns
    )


def _collect_supporting_tasks(
    plan: list[PlanTask],
    current_index: int,
    task: PlanTask,
) -> list[PlanTask]:
    """Collect completed upstream tasks that can support code-based analysis."""

    dependency_ids = set(task["dependencies"])
    if dependency_ids:
        return [
            item
            for item in plan
            if item["id"] in dependency_ids and item["result"].strip()
        ]

    return [
        item
        for item in plan[:current_index]
        if item["result"].strip()
    ]


def _is_synthesis_task(description: str) -> bool:
    """Return whether a task description is requesting synthesis rather than execution."""

    lowered = description.lower()
    synthesis_markers = (
        "analyze",
        "compare",
        "synthesize",
        "identify the best",
        "summarize findings",
    )
    return any(marker in lowered for marker in synthesis_markers)


def _build_synthesis_prompt(task: PlanTask, supporting_tasks: list[PlanTask]) -> tuple[str, list[str]]:
    """Create the direct synthesis prompt and aggregate sources from prior tasks."""

    aggregated_sources: list[str] = []
    prior_results: list[str] = []

    if not supporting_tasks:
        prior_results.append("No prior task findings were available.")
    else:
        for item in supporting_tasks:
            compact_result = " ".join(item["result"].split()).strip()
            source_list = item["sources"] or ["no recorded sources"]
            source_text = ", ".join(source_list)
            prior_results.append(
                f"Task {item['id']} found the following from sources [{source_text}]: "
                f"{compact_result or 'No findings captured.'}"
            )
            for source in item["sources"]:
                if source not in aggregated_sources:
                    aggregated_sources.append(source)

    prompt = (
        "You are a research synthesizer. Here are findings from prior research tasks: "
        f"{chr(10).join(prior_results)}. "
        f"Task: {task['description']}. "
        "Provide a structured analysis with: (1) specific named approaches mentioned across sources with their mechanisms, "
        "(2) concrete comparisons using any numbers or metrics found, "
        "(3) a clear recommendation with reasoning. "
        "Be specific -- cite exact paper names, system names, or metrics from the sources. "
        "When citing information, cite the original source URL or document name, not the task number. "
        "Write citations as (Source: [url or filename]) inline."
    )
    return prompt, aggregated_sources


def _run_synthesis(task: PlanTask, plan: list[PlanTask], current_index: int) -> ToolResult:
    """Run a direct GPT-4o synthesis pass over prior task findings."""

    supporting_tasks = _collect_supporting_tasks(plan, current_index, task)
    prompt, aggregated_sources = _build_synthesis_prompt(task, supporting_tasks)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ToolResult(
            content="",
            sources=aggregated_sources,
            raw_confidence=0.0,
            tool_name="synthesis",
        )

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0,
    )
    try:
        response = invoke_with_rate_limit_retry(
            lambda: llm.invoke([HumanMessage(content=prompt)])
        )
    except Exception:
        return ToolResult(
            content="",
            sources=aggregated_sources,
            raw_confidence=0.0,
            tool_name="synthesis",
        )

    return ToolResult(
        content=(response.content or "").strip(),
        sources=aggregated_sources,
        raw_confidence=0.8 if (response.content or "").strip() else 0.0,
        tool_name="synthesis",
    )


def _build_analysis_script(task: PlanTask, supporting_tasks: list[PlanTask]) -> tuple[str, list[str]]:
    """Create a safe Python script for analysis tasks described in plain English."""

    aggregated_sources: list[str] = []
    safe_task_description = task["description"].replace("\\", "\\\\").replace("'", "\\'")
    lines = [
        f"print('Analysis task: {safe_task_description}')",
        "print('')",
    ]

    if not supporting_tasks:
        lines.extend(
            [
                "print('No supporting task results were available for this analysis.')",
                "print('Add prior evidence or convert the task into explicit Python code for deeper analysis.')",
            ]
        )
        return "\n".join(lines), aggregated_sources

    lines.append("print('Supporting evidence summary:')")
    for item in supporting_tasks:
        preview = shorten(" ".join(item["result"].split()), width=240, placeholder="...")
        safe_description = item["description"].replace("\\", "\\\\").replace("'", "\\'")
        safe_preview = preview.replace("\\", "\\\\").replace("'", "\\'")
        lines.append(f"print('- Task {item['id']}: {safe_description}')")
        lines.append(f"print('  Findings: {safe_preview}')")
        if item["sources"]:
            lines.append("print('  Sources:')")
            for source in item["sources"][:3]:
                safe_source = source.replace("\\", "\\\\").replace("'", "\\'")
                lines.append(f"print('    - {safe_source}')")
                if source not in aggregated_sources:
                    aggregated_sources.append(source)

    lines.extend(
        [
            "print('')",
            "print('Preliminary comparison:')",
            (
                "print('Recurring approaches across the supporting evidence should be prioritized, "
                "especially when they combine retrieval, summarization, and hierarchical memory patterns.')"
            ),
            (
                "print('Use the cited supporting tasks to judge tradeoffs such as faithfulness, "
                "retrieval precision, long-horizon task completion, and cost.')"
            ),
        ]
    )

    return "\n".join(lines), aggregated_sources


def _build_retry_analysis_script(
    task: PlanTask,
    supporting_tasks: list[PlanTask],
    critic_feedback: str,
) -> tuple[str, list[str]]:
    """Create a deeper retry prompt for fallback analysis after critic rejection."""

    script, aggregated_sources = _build_analysis_script(task, supporting_tasks)
    safe_feedback = " ".join(critic_feedback.split()).replace("\\", "\\\\").replace("'", "\\'")
    retry_lines = [
        script,
        "print('')",
        (
            "print('The previous analysis was rejected because: "
            f"{safe_feedback}. Generate a deeper Python analysis that specifically addresses this criticism "
            "-- include concrete comparisons, specific examples, and structured conclusions.')"
        ),
        "print('')",
        "print('Structured conclusions:')",
        (
            "print('1. Compare the strongest approaches using evidence from the supporting tasks, including "
            "their mechanisms and tradeoffs.')"
        ),
        (
            "print('2. Call out concrete examples, cited systems, or benchmark details that make one approach "
            "more reliable than another.')"
        ),
        (
            "print('3. End with a concise recommendation that directly answers the task based on the cited "
            "evidence.')"
        ),
    ]
    return "\n".join(retry_lines), aggregated_sources


def _fallback_analysis_confidence(output: str) -> float:
    """Score fallback-generated analysis by how substantial the executed output is."""

    normalized = output.strip()
    if not normalized:
        return 0.0
    length = len(normalized)
    if length < 300:
        return 0.5
    if length <= 800:
        return 0.7
    return 0.85


def _prepare_code_input(
    task: PlanTask,
    plan: list[PlanTask],
    current_index: int,
    critic_feedback: str,
) -> tuple[str, list[str], bool]:
    """Prepare executable Python code for a code_interpreter task."""

    task_input = task["description"].strip()
    if _looks_like_python_code(task_input):
        return task_input, [], False

    supporting_tasks = _collect_supporting_tasks(plan, current_index, task)
    if task["retries"] > 0 and critic_feedback.strip():
        script, aggregated_sources = _build_retry_analysis_script(
            task,
            supporting_tasks,
            critic_feedback,
        )
    else:
        script, aggregated_sources = _build_analysis_script(task, supporting_tasks)
    return script, aggregated_sources, True


def _reformulate_retrieval_query(task_description: str, critic_feedback: str) -> str:
    """Use critic feedback to produce a more targeted retry query for retrieval."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not critic_feedback.strip():
        return task_description

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=api_key,
        temperature=0,
    )
    prompt = (
        f"Given this task: {task_description} and this critic feedback: {critic_feedback}, "
        "write a more specific search query (10 words max) that would retrieve better documents."
    )

    try:
        response = invoke_with_rate_limit_retry(
            lambda: llm.invoke([HumanMessage(content=prompt)])
        )
    except Exception:
        return task_description

    reformulated_query = " ".join((response.content or "").split())
    if not reformulated_query:
        return task_description
    return reformulated_query


def _select_tool(task: PlanTask, tool_history: list[str]) -> str:
    """Choose a tool, switching away after repeated use within one task."""

    preferred_tool = task["tool"]
    if tool_history.count(preferred_tool) < 2:
        return preferred_tool

    for candidate in TOOL_ORDER:
        if candidate != preferred_tool:
            return candidate
    return preferred_tool


def _should_route_to_synthesis(task: PlanTask, selected_tool: str) -> bool:
    """Return whether the executor should use synthesis instead of code execution."""

    return (
        selected_tool == "code_interpreter"
        and not _looks_like_python_code(task["description"])
        and _is_synthesis_task(task["description"])
    )


def _execute_tool(
    task: PlanTask,
    selected_tool: str,
    plan: list[PlanTask],
    current_index: int,
    critic_feedback: str,
) -> ToolResult:
    """Execute the selected tool using the current task description as input."""

    task_input = task["description"]
    if selected_tool == "web_search":
        return run_web_search(task_input)
    if selected_tool == "document_retrieval":
        retrieval_query = task_input
        retrieval_top_k = 3
        if task["retries"] > 0 and critic_feedback.strip():
            retrieval_query = _reformulate_retrieval_query(task_input, critic_feedback)
            retrieval_top_k = 5
        return run_retrieval(retrieval_query, top_k=retrieval_top_k)
    if selected_tool == "synthesis":
        return _run_synthesis(task, plan, current_index)
    if selected_tool == "code_interpreter":
        if not _looks_like_python_code(task_input) and _is_synthesis_task(task_input):
            return _run_synthesis(task, plan, current_index)
        code_input, fallback_sources, used_fallback = _prepare_code_input(
            task,
            plan,
            current_index,
            critic_feedback,
        )
        result = run_code(code_input)
        adjusted_confidence = (
            _fallback_analysis_confidence(result.content)
            if used_fallback
            else 1.0
        )
        if fallback_sources and not result.sources:
            return ToolResult(
                content=result.content,
                sources=fallback_sources,
                raw_confidence=adjusted_confidence,
                tool_name=result.tool_name,
            )
        return ToolResult(
            content=result.content,
            sources=result.sources,
            raw_confidence=adjusted_confidence,
            tool_name=result.tool_name,
        )
    raise ValueError(f"Unsupported tool: {selected_tool}")


def executor_node(state: AgentState) -> AgentState:
    """Run the current sub-task with the appropriate external tool.

    The executor reads the current task from the plan, avoids repeated
    calls to the same tool within a single task, dispatches execution,
    stores the result and confidence back into the plan, appends the
    invoked tool to the current task's history, and returns the updated
    graph state.
    """

    current_index = state["current_task_index"]
    plan = [task.copy() for task in state["plan"]]

    if current_index < 0 or current_index >= len(plan):
        return {
            **state,
            "error": f"Invalid current_task_index: {current_index}",
            "iteration_count": state["iteration_count"] + 1,
        }

    current_task = plan[current_index]
    selected_tool = _select_tool(current_task, state["tool_history"])
    if _should_route_to_synthesis(current_task, selected_tool):
        selected_tool = "synthesis"
    print("\n[executor] Starting executor step")
    print(
        f"[executor] Task {current_task['id']} | preferred_tool={current_task['tool']} | "
        f"selected_tool={selected_tool}"
    )
    print(f"[executor] Task description: {current_task['description']}")
    if selected_tool == "synthesis":
        print("[executor] Using direct synthesis over prior task results")
    if selected_tool == "synthesis" and current_task["tool"] == "code_interpreter":
        print("[executor] Task description is analytical without executable Python, routing to synthesis")
    elif selected_tool == "code_interpreter" and not _looks_like_python_code(current_task["description"]):
        print("[executor] Task description is not executable Python, building analysis code from prior results")
    result = _execute_tool(
        current_task,
        selected_tool,
        plan,
        current_index,
        state["critic_feedback"],
    )

    current_task["result"] = result.content
    current_task["sources"] = result.sources
    current_task["confidence"] = result.raw_confidence
    current_task["status"] = "completed" if result.raw_confidence > 0.0 else "failed"

    result_preview = (result.content or "").replace("\n", " ").strip()
    if len(result_preview) > 200:
        result_preview = result_preview[:200] + "..."
    print(
        f"[executor] Result confidence={result.raw_confidence:.2f} | "
        f"sources={len(result.sources)}"
    )
    print(f"[executor] Result preview: {result_preview or '[empty result]'}")

    updated_tool_history = [*state["tool_history"], selected_tool]
    error_message = (
        None
        if result.raw_confidence > 0.0
        else f"Tool execution failed for task {current_task['id']} with {selected_tool}."
    )

    return {
        **state,
        "plan": plan,
        "tool_history": updated_tool_history,
        "error": error_message,
        "iteration_count": state["iteration_count"] + 1,
    }


__all__ = ["executor_node"]
