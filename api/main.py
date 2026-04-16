"""FastAPI app for deploying the NeuroAgent research workflow."""

from __future__ import annotations

import os
from statistics import mean
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from agent.graph import run_agent
from api.schemas import HealthResponse, RunAgentRequest, RunAgentResponse, TaskSummary


app = FastAPI(
    title="NeuroAgent API",
    description=(
        "A modular research-agent API that plans, retrieves evidence, critiques "
        "intermediate outputs, and synthesizes a final report."
    ),
    version="0.1.0",
)


LANDING_PAGE_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NeuroAgent API</title>
    <style>
      :root {
        --bg: #f4f1e8;
        --ink: #1e2b22;
        --accent: #0f766e;
        --card: #fffdf8;
        --muted: #5c6b63;
        --border: #d8d2c1;
      }
      body {
        margin: 0;
        font-family: Georgia, "Times New Roman", serif;
        background: radial-gradient(circle at top, #fff7e7 0%, var(--bg) 55%);
        color: var(--ink);
      }
      .wrap {
        max-width: 880px;
        margin: 0 auto;
        padding: 56px 24px 72px;
      }
      .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 32px;
        box-shadow: 0 18px 45px rgba(30, 43, 34, 0.08);
      }
      .eyebrow {
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        font-size: 0.82rem;
        margin-bottom: 14px;
      }
      h1 {
        margin: 0 0 14px;
        font-size: clamp(2.1rem, 4vw, 3.4rem);
        line-height: 1.05;
      }
      p {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
      }
      .actions, .grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 22px;
      }
      .grid {
        margin-top: 28px;
      }
      a.button, .pill {
        display: inline-block;
        text-decoration: none;
        border-radius: 999px;
        padding: 11px 18px;
        border: 1px solid var(--border);
        background: #ffffff;
        color: var(--ink);
      }
      a.button.primary {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
      }
      .card {
        flex: 1 1 220px;
        min-width: 220px;
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
      }
      code, pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      }
      pre {
        white-space: pre-wrap;
        background: #fcfaf2;
        border: 1px solid var(--border);
        padding: 16px;
        border-radius: 14px;
        overflow-x: auto;
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <div class="eyebrow">LangGraph Research Agent</div>
        <h1>NeuroAgent API</h1>
        <p>
          NeuroAgent is a modular research agent that plans work, retrieves evidence,
          critiques intermediate outputs, retries weak steps, and returns a
          source-grounded final report.
        </p>
        <div class="actions">
          <a class="button primary" href="/docs">Open API Docs</a>
          <a class="button" href="/health">Health Check</a>
        </div>
        <div class="grid">
          <div class="card">
            <strong>What it does</strong>
            <p>Planning, retrieval, synthesis, critique, retries, and final reporting in one API.</p>
          </div>
          <div class="card">
            <strong>Main endpoint</strong>
            <p><code>POST /run</code> executes the full workflow for a research question.</p>
          </div>
          <div class="card">
            <strong>Protection</strong>
            <p>Set <code>NEURO_AGENT_API_KEY</code> and send it as <code>X-API-Key</code> for protected access.</p>
          </div>
        </div>
        <pre>curl -X POST "/run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: &lt;your-api-key&gt;" \
  -d '{"question":"What is Reflexion in LLM agents?"}'</pre>
      </section>
    </div>
  </body>
</html>
"""


def _enforce_api_key(x_api_key: str | None) -> None:
    """Protect the expensive /run endpoint when an API key is configured."""

    expected_api_key = os.getenv("NEURO_AGENT_API_KEY", "").strip()
    if not expected_api_key:
        return
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/", response_class=HTMLResponse, tags=["meta"])
def root() -> HTMLResponse:
    """Render a lightweight landing page for the public deployment URL."""

    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Return health metadata for uptime and deployment checks."""

    return HealthResponse(status="ok", service="neuro-agent-api")


@app.post("/run", response_model=RunAgentResponse, tags=["agent"])
def run_research_agent(
    payload: RunAgentRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> RunAgentResponse:
    """Execute the full NeuroAgent workflow for a single research question."""

    _enforce_api_key(x_api_key)
    session_id = payload.session_id or f"api-{uuid4().hex[:12]}"

    try:
        final_state = run_agent(question=payload.question, session_id=session_id)
    except Exception as error:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=f"Agent run failed: {error}") from error

    tasks = [
        TaskSummary(
            id=task["id"],
            description=task["description"],
            tool=task["tool"],
            status=task["status"],
            confidence=task["confidence"],
            retries=task["retries"],
            sources=task["sources"],
        )
        for task in final_state["plan"]
    ]
    confidences = [task.confidence for task in tasks]

    return RunAgentResponse(
        session_id=session_id,
        question=payload.question,
        final_report=final_state["final_report"],
        final_confidence=mean(confidences) if confidences else 0.0,
        total_retries=sum(task.retries for task in tasks),
        num_tasks=len(tasks),
        tasks=tasks,
    )
