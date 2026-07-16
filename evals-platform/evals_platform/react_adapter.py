"""Boundary adapter from the evaluation platform to the canonical ReAct agent.

The eval package lives below the repository root, so this module first ensures
the root is importable and then imports ``ReAct.main.run_agent``. The adapter adds
no control-loop behavior: its purpose is to provide a narrow, injectable service
surface while guaranteeing evaluations exercise exactly the same implementation
as the terminal agent.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from .config import REPOSITORY_ROOT

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ReAct.main import AgentRunResult, run_agent  # noqa: E402


class ReactAdapter:
    """Invoke the repository agent without duplicating its control loop.

    An optional chat client is retained and forwarded to each run. Tests use this
    seam to provide fixed streaming responses without requiring Ollama.
    """

    def __init__(self, chat_client: Callable[..., Any] | None = None) -> None:
        """Create an adapter with an optional Ollama-compatible chat client."""

        self.chat_client = chat_client

    def run(
        self,
        prompt: str,
        *,
        model: str,
        max_iterations: int,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRunResult:
        """Run one prompt and return the agent's native structured result."""

        return run_agent(
            prompt,
            model=model,
            max_iterations=max_iterations,
            event_callback=event_callback,
            chat_client=self.chat_client,
        )
