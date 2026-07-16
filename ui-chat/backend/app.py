"""FastAPI streaming adapter for the repository's synchronous ReAct loop."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Allow `uvicorn backend.app:app` to reuse the repository package without copying it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ReAct import run_agent  # noqa: E402

from .models import ChatRequest

AgentRunner = Callable[..., Any]
_STREAM_END = object()

app = FastAPI(title="ReAct Chat API", version="0.1.0")
app.state.agent_runner = run_agent


def _max_iterations() -> int:
    raw_value = os.getenv("AGENT_MAX_ITERATIONS", "4")
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 4


def _sse(event: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"


def _run_in_thread(
    request: ChatRequest,
    runner: AgentRunner,
    events: queue.Queue[object],
) -> None:
    def receive_agent_event(event: dict[str, Any]) -> None:
        if event.get("type") == "model_token":
            events.put(
                (
                    "token",
                    {
                        "content": str(event.get("content", "")),
                        "iteration": event.get("iteration"),
                    },
                )
            )

    try:
        result = runner(
            request.prompt,
            prior_messages=[
                message.model_dump(exclude_none=True)
                for message in request.prior_messages
            ],
            model=os.getenv("AGENT_MODEL", "llama3.2:3b"),
            max_iterations=_max_iterations(),
            event_callback=receive_agent_event,
        )
        if result.status == "failed":
            events.put(
                (
                    "error",
                    {
                        "code": result.error_type or "agent_failed",
                        "message": result.error_message
                        or "The agent could not complete this response.",
                    },
                )
            )
        else:
            events.put(
                (
                    "complete",
                    {
                        "status": result.status,
                        "final_answer": result.final_answer,
                        "messages": result.messages,
                    },
                )
            )
    except Exception as exc:  # The stream is already open, so report in-band.
        events.put(
            (
                "error",
                {
                    "code": type(exc).__name__,
                    "message": str(exc) or "The agent service encountered an error.",
                },
            )
        )
    finally:
        events.put(_STREAM_END)


def _event_stream(request: ChatRequest, runner: AgentRunner) -> Iterator[str]:
    events: queue.Queue[object] = queue.Queue()
    worker = threading.Thread(
        target=_run_in_thread,
        args=(request, runner, events),
        name="react-agent-turn",
        daemon=True,
    )
    worker.start()

    while True:
        try:
            item = events.get(timeout=15)
        except queue.Empty:
            yield ": keep-alive\n\n"
            continue
        if item is _STREAM_END:
            break
        event, payload = item
        yield _sse(event, payload)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "llama3.2:3b"),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request, app.state.agent_runner),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
