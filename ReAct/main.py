"""Instrumented ReAct-style tool-calling agent backed by Ollama.

The module owns the repository's canonical agent loop. During each iteration it
sends the accumulated conversation and tool schemas to a streaming chat model,
captures text and requested tool calls, executes those calls against a registry,
adds observations to the conversation, and asks the model to continue. A turn
ends when the model returns no tool calls, the model raises an error, or the
configured iteration limit is exhausted.

Every model and tool action is recorded as an :class:`AgentStep` and returned in
an :class:`AgentRunResult`. Optional event callbacks expose the same activity to
interactive clients while it happens. Both the chat client and tool registry are
injectable, which keeps the control loop deterministic and offline-testable even
though its normal runtime uses Ollama.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import ollama

LANGUAGE_MODEL = "llama3.2:3b"
MAX_ITERATIONS = 4

DEFAULT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a specific city location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. London, Tokyo",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


@dataclass
class AgentStep:
    """One observable model or tool action in an agent trajectory.

    Model steps carry generated ``content``. Tool steps carry the selected tool,
    parsed arguments, resulting observation, and any execution error. Latency is
    measured per action so evaluators can separate inference time from tool time.
    ``iteration`` identifies the model turn that produced or followed the step.
    """

    kind: str
    iteration: int
    content: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0


@dataclass
class AgentRunResult:
    """Complete structured output from one invocation of :func:`run_agent`.

    The record combines the final outcome with the complete model-ready message
    history and the evaluation-friendly step trajectory. ``status`` describes
    whether execution completed, completed after a recoverable tool error, or
    failed. ``termination_reason`` gives the concrete stop condition.
    """

    prompt: str
    model: str
    max_iterations: int
    status: str
    termination_reason: str
    final_answer: str
    messages: list[dict[str, Any]]
    steps: list[AgentStep]
    iterations: int
    total_latency_ms: float
    model_latency_ms: float
    tool_latency_ms: float
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert this result and all nested steps to dictionaries."""

        return asdict(self)


EventCallback = Callable[[dict[str, Any]], None]
ChatClient = Callable[..., Iterable[Any]]


def get_current_weather(location: str) -> str:
    """Return deterministic example weather data for a city.

    London, Tokyo, and New York have fixed observations so agent behavior can be
    evaluated repeatably. Other locations receive an explicit unavailable-data
    observation rather than fabricated weather.
    """

    database = {
        "london": "15°C, Rainy 🌧️",
        "tokyo": "26°C, Sunny ☀️",
        "new york": "22°C, Windy 💨",
    }
    return database.get(
        location.lower(), "Weather data not available for this location."
    )


AVAILABLE_TOOLS: dict[str, Callable[..., Any]] = {
    "get_current_weather": get_current_weather
}


def llm_call(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    model: str = LANGUAGE_MODEL,
    chat_client: ChatClient | None = None,
) -> Iterable[Any]:
    """Start a streaming model call with an Ollama-compatible client.

    Args:
        messages: Conversation history in Ollama's role/content format.
        tools: Function schemas advertised to the model; defaults to the weather
            tool schema when omitted.
        model: Ollama model tag used for inference.
        chat_client: Optional replacement for ``ollama.chat``, primarily used by
            tests and alternate front ends.

    Returns:
        An iterable of streaming response chunks. Consumption and aggregation
        are deliberately handled by :func:`run_agent`.
    """

    client = chat_client or ollama.chat
    return client(
        model=model,
        messages=list(messages),
        stream=True,
        tools=list(tools if tools is not None else DEFAULT_TOOL_SCHEMAS),
    )


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    return value


def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = _value(tool_call, "function", {})
    name = str(_value(function, "name", ""))
    raw_arguments = _value(function, "arguments", {})
    if isinstance(raw_arguments, str):
        parsed = json.loads(raw_arguments)
    else:
        parsed = _plain(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return name, parsed


def _emit(callback: EventCallback | None, event: dict[str, Any]) -> None:
    if callback is not None:
        try:
            callback(event)
        except Exception:
            # UI callbacks must never change agent behavior.
            pass


def run_agent(
    prompt: str,
    *,
    prior_messages: Sequence[Mapping[str, Any]] | None = None,
    model: str = LANGUAGE_MODEL,
    max_iterations: int = MAX_ITERATIONS,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    tool_registry: Mapping[str, Callable[..., Any]] | None = None,
    event_callback: EventCallback | None = None,
    chat_client: ChatClient | None = None,
) -> AgentRunResult:
    """Execute one user turn through the bounded ReAct loop.

    Args:
        prompt: Non-empty user request to append to the conversation.
        prior_messages: Optional prior conversation for multi-turn callers.
        model: Ollama model tag used for each reasoning turn.
        max_iterations: Maximum number of model calls before forced termination.
        tool_schemas: Tool definitions exposed to the model.
        tool_registry: Mapping from tool names to executable Python callables.
        event_callback: Best-effort receiver for token, model-complete, and
            tool-complete events. Callback failures never affect agent behavior.
        chat_client: Optional Ollama-compatible client for dependency injection.

    Returns:
        An :class:`AgentRunResult` containing the answer, status, message history,
        trajectory, timing totals, and any terminal error.

    Raises:
        ValueError: If ``prompt`` is blank or ``max_iterations`` is less than one.

    Tool failures are converted into observations so the model gets a chance to
    recover. Model failures and iteration exhaustion return failed results rather
    than raising, making partial trajectories available to the evaluation layer.
    """

    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    messages = [dict(message) for message in (prior_messages or [])]
    messages.append({"role": "user", "content": prompt})
    tools = list(tool_schemas if tool_schemas is not None else DEFAULT_TOOL_SCHEMAS)
    registry = dict(tool_registry if tool_registry is not None else AVAILABLE_TOOLS)
    steps: list[AgentStep] = []
    final_answer = ""
    model_latency_ms = 0.0
    tool_latency_ms = 0.0
    had_tool_error = False
    started = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        model_started = time.perf_counter()
        content_parts: list[str] = []
        tool_calls: list[Any] = []
        try:
            response = llm_call(
                messages,
                tools,
                model=model,
                chat_client=chat_client,
            )
            for chunk in response:
                message_chunk = _value(chunk, "message", {})
                content = _value(message_chunk, "content", "") or ""
                if content:
                    content_parts.append(str(content))
                    _emit(
                        event_callback,
                        {"type": "model_token", "iteration": iteration, "content": content},
                    )
                chunk_tool_calls = _value(message_chunk, "tool_calls", None)
                if chunk_tool_calls:
                    tool_calls.extend(list(chunk_tool_calls))
        except Exception as exc:
            elapsed = (time.perf_counter() - model_started) * 1000
            model_latency_ms += elapsed
            steps.append(
                AgentStep(
                    kind="model",
                    iteration=iteration,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    latency_ms=elapsed,
                )
            )
            return AgentRunResult(
                prompt=prompt,
                model=model,
                max_iterations=max_iterations,
                status="failed",
                termination_reason="model_error",
                final_answer=final_answer,
                messages=messages,
                steps=steps,
                iterations=iteration,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                model_latency_ms=model_latency_ms,
                tool_latency_ms=tool_latency_ms,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        model_elapsed = (time.perf_counter() - model_started) * 1000
        model_latency_ms += model_elapsed
        assistant_content = "".join(content_parts)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = _plain(tool_calls)
        messages.append(assistant_message)
        steps.append(
            AgentStep(
                kind="model",
                iteration=iteration,
                content=assistant_content,
                latency_ms=model_elapsed,
            )
        )
        _emit(
            event_callback,
            {
                "type": "model_complete",
                "iteration": iteration,
                "content": assistant_content,
                "tool_call_count": len(tool_calls),
                "latency_ms": model_elapsed,
            },
        )

        if not tool_calls:
            final_answer = assistant_content
            return AgentRunResult(
                prompt=prompt,
                model=model,
                max_iterations=max_iterations,
                status="completed_with_errors" if had_tool_error else "completed",
                termination_reason="final_answer",
                final_answer=final_answer,
                messages=messages,
                steps=steps,
                iterations=iteration,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                model_latency_ms=model_latency_ms,
                tool_latency_ms=tool_latency_ms,
            )

        for raw_tool_call in tool_calls:
            tool_started = time.perf_counter()
            tool_name = ""
            arguments: dict[str, Any] = {}
            observation = ""
            success = True
            error_type = None
            error_message = None
            try:
                tool_name, arguments = _parse_tool_call(raw_tool_call)
                if tool_name not in registry:
                    raise LookupError(f"Unknown tool: {tool_name or '<missing name>'}")
                observation = str(registry[tool_name](**arguments))
            except Exception as exc:
                success = False
                had_tool_error = True
                error_type = type(exc).__name__
                error_message = str(exc)
                observation = f"Tool error: {error_message}"

            tool_elapsed = (time.perf_counter() - tool_started) * 1000
            tool_latency_ms += tool_elapsed
            step = AgentStep(
                kind="tool",
                iteration=iteration,
                tool_name=tool_name or None,
                arguments=arguments,
                observation=observation,
                success=success,
                error_type=error_type,
                error_message=error_message,
                latency_ms=tool_elapsed,
            )
            steps.append(step)
            messages.append(
                {
                    "role": "tool",
                    "content": observation,
                    "tool_name": tool_name,
                }
            )
            _emit(event_callback, {"type": "tool_complete", **asdict(step)})

    return AgentRunResult(
        prompt=prompt,
        model=model,
        max_iterations=max_iterations,
        status="failed",
        termination_reason="max_iterations",
        final_answer=final_answer,
        messages=messages,
        steps=steps,
        iterations=max_iterations,
        total_latency_ms=(time.perf_counter() - started) * 1000,
        model_latency_ms=model_latency_ms,
        tool_latency_ms=tool_latency_ms,
        error_type="MaxIterationsError",
        error_message=f"Agent did not produce a final answer in {max_iterations} iterations.",
    )


def main() -> None:
    """Run a stateful terminal chat over the canonical agent loop.

    Conversation messages are carried between prompts until the user enters
    ``exit``/``quit`` or interrupts input. Streaming model text and completed tool
    calls are rendered from events, while final status comes from the structured
    result returned by :func:`run_agent`.
    """

    conversation: list[dict[str, Any]] = []
    print("ReAct weather agent. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        def render_event(event: dict[str, Any]) -> None:
            if event["type"] == "model_token":
                print(event["content"], end="", flush=True)
            elif event["type"] == "tool_complete":
                print(
                    f"\n⚙️ [Agent Action] {event.get('tool_name')}"
                    f"({event.get('arguments')}) -> {event.get('observation')}"
                )

        result = run_agent(
            user_input,
            prior_messages=conversation,
            event_callback=render_event,
        )
        conversation = result.messages
        if result.status == "failed":
            print(f"\n❌ {result.error_message}")
        print(
            f"\n✅ [Agent Finished] {result.termination_reason} "
            f"after {result.iterations} model turn(s)."
        )


if __name__ == "__main__":
    main()