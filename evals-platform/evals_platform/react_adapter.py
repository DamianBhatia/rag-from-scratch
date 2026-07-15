"""Thin adapter around the real ReAct agent implementation."""

from __future__ import annotations

import sys
from typing import Any, Callable

from .config import REPOSITORY_ROOT

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ReAct.main import AgentRunResult, run_agent  # noqa: E402


class ReactAdapter:
    """Invoke the repository agent without duplicating its control loop."""

    def __init__(self, chat_client: Callable[..., Any] | None = None) -> None:
        self.chat_client = chat_client

    def run(
        self,
        prompt: str,
        *,
        model: str,
        max_iterations: int,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRunResult:
        return run_agent(
            prompt,
            model=model,
            max_iterations=max_iterations,
            event_callback=event_callback,
            chat_client=self.chat_client,
        )
