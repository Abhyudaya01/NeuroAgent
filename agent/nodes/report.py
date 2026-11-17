"""Report node for synthesizing a final answer from completed sub-tasks."""

from __future__ import annotations

from agent.state import AgentState


def _unique_sources(plan: list[dict]) -> list[str]:
    """Collect unique source references from completed tasks."""

    seen: set[str] = set()
    ordered_sources: list[str] = []
    for task in plan:
        for source in task.get("sources", []):
            if source and source not in seen:
                seen.add(source)
                ordered_sources.append(source)
    return ordered_sources


def _task_section(task: dict) -> str:
    """Render a single task result section for the final report."""

    confidence = task.get("confidence", 0.0)
    result = task.get("result", "").strip() or "No result captured."
    section_lines = [
        f"Task {task['id']}: {task['description']}",
        f"Status: {task['status']} | Confidence: {confidence:.2f}",
        "Findings:",
        result,
    ]
    return "\n".join(section_lines)


def report_node(state: AgentState) -> AgentState:
    """Build a cleaner final report from completed task outputs."""

    sections: list[str] = []
    print("\n[report] Starting report step")
    for task in state["plan"]:
        task_result = task["result"].strip()
        if not task_result:
            continue
        sections.append(_task_section(task))

    completed_tasks = sum(
        1 for task in state["plan"] if task["status"] == "completed" and task["result"].strip()
    )
    total_tasks = len(state["plan"])
    sources = _unique_sources(state["plan"])

    summary_lines = [
        f"Research Question: {state['research_question']}",
        (
            "Summary: "
            f"Completed {completed_tasks} of {total_tasks} tasks. "
            f"The run collected {len(sources)} unique sources."
        ),
    ]
    if state["critic_feedback"]:
        summary_lines.append(f"Last Critic Feedback: {state['critic_feedback']}")

    report_parts = ["\n".join(summary_lines)]
    if sections:
        report_parts.append("Task Findings:\n" + "\n\n".join(sections))
    if sources:
        report_parts.append("All Sources:\n" + "\n".join(f"- {source}" for source in sources))

    final_report = "\n\n".join(report_parts).strip()
    print(f"[report] Included {len(sections)} task sections in final report")
    if final_report:
        preview = final_report.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[report] Final report preview: {preview}")
    else:
        print("[report] Final report is empty")
    return {
        **state,
        "final_report": final_report,
    }


__all__ = ["report_node"]
