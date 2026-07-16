"""Offline API tests for the FastAPI-to-ReAct streaming adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import app


@pytest.fixture(autouse=True)
def restore_agent_runner():
    original = app.state.agent_runner
    yield
    app.state.agent_runner = original


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _result(**overrides):
    values = {
        "status": "completed",
        "final_answer": "Hello there",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello there"},
        ],
        "error_type": None,
        "error_message": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_health_reports_ready_model(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"]


def test_blank_prompt_is_rejected_before_streaming(client: TestClient):
    response = client.post("/chat", json={"prompt": "   "})

    assert response.status_code == 422


def test_chat_streams_ordered_tokens_and_terminal_history(client: TestClient):
    captured = {}

    def fake_runner(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["prior_messages"] = kwargs["prior_messages"]
        callback = kwargs["event_callback"]
        callback({"type": "model_token", "iteration": 1, "content": "Hello"})
        callback({"type": "tool_complete", "observation": "hidden"})
        callback({"type": "model_token", "iteration": 2, "content": " there"})
        return _result()

    app.state.agent_runner = fake_runner
    response = client.post(
        "/chat",
        json={
            "prompt": "Hello",
            "prior_messages": [{"role": "assistant", "content": "Earlier"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert body.index('"content":"Hello"') < body.index('"content":" there"')
    assert "observation" not in body
    assert "event: complete" in body
    assert '"role":"assistant"' in body
    assert captured == {
        "prompt": "Hello",
        "prior_messages": [{"role": "assistant", "content": "Earlier"}],
    }


def test_failed_result_is_an_error_event(client: TestClient):
    app.state.agent_runner = lambda *args, **kwargs: _result(
        status="failed",
        final_answer="",
        messages=[],
        error_type="RuntimeError",
        error_message="Ollama is offline",
    )

    response = client.post("/chat", json={"prompt": "Hello"})

    assert "event: error" in response.text
    assert "Ollama is offline" in response.text
    assert "event: complete" not in response.text


def test_unexpected_exception_is_an_error_event(client: TestClient):
    def broken_runner(*args, **kwargs):
        raise ValueError("broken adapter")

    app.state.agent_runner = broken_runner
    response = client.post("/chat", json={"prompt": "Hello"})

    assert "event: error" in response.text
    assert "broken adapter" in response.text
