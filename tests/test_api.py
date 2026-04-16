"""Tests for the public FastAPI deployment surface."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_root_endpoint_returns_html_landing_page() -> None:
    """Root endpoint should render a simple public landing page."""

    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NeuroAgent API" in response.text
    assert "X-API-Key" in response.text


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint should expose a simple deployment check."""

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "neuro-agent-api"}


def test_run_endpoint_returns_agent_summary(monkeypatch) -> None:
    """Run endpoint should serialize the agent output into the API schema."""

    def fake_run_agent(question: str, session_id: str) -> dict:
        assert question == "What is Reflexion?"
        assert session_id == "resume-demo"
        return {
            "research_question": question,
            "session_id": session_id,
            "plan": [
                {
                    "id": 1,
                    "description": "Search for Reflexion evidence.",
                    "tool": "web_search",
                    "dependencies": [],
                    "status": "completed",
                    "result": "Found the Reflexion paper.",
                    "sources": ["https://arxiv.org/abs/2303.11366"],
                    "confidence": 0.7,
                    "retries": 1,
                },
                {
                    "id": 2,
                    "description": "Synthesize findings.",
                    "tool": "synthesis",
                    "dependencies": [1],
                    "status": "completed",
                    "result": "Reflexion improves reliability via verbal feedback.",
                    "sources": ["data/seed_docs/reflexion_verbal_reinforcement.md"],
                    "confidence": 0.8,
                    "retries": 0,
                },
            ],
            "current_task_index": 2,
            "tool_history": [],
            "memory_context": "",
            "critic_score": 0.8,
            "critic_feedback": "",
            "should_retry": False,
            "final_report": "Reflexion uses verbal reinforcement learning.",
            "wandb_task_rows": [],
            "error": None,
            "iteration_count": 0,
        }

    monkeypatch.setattr("api.main.run_agent", fake_run_agent)
    client = TestClient(app)

    response = client.post(
        "/run",
        json={
            "question": "What is Reflexion?",
            "session_id": "resume-demo",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "resume-demo"
    assert payload["num_tasks"] == 2
    assert payload["total_retries"] == 1
    assert payload["final_confidence"] == 0.75
    assert payload["final_report"] == "Reflexion uses verbal reinforcement learning."


def test_run_endpoint_requires_api_key_when_configured(monkeypatch) -> None:
    """Run endpoint should reject missing or incorrect API keys when enabled."""

    monkeypatch.setenv("NEURO_AGENT_API_KEY", "secret-demo-key")
    client = TestClient(app)

    response = client.post(
        "/run",
        json={"question": "What is Reflexion?"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key."}
