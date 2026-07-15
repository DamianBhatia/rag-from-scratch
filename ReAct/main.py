"""A small, instrumented ReAct-style agent loop backed by Ollama."""

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
    """One observable model or tool action in an agent run."""

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
    """Structured output from one invocation of :func:`run_agent`."""

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
        return asdict(self)


EventCallback = Callable[[dict[str, Any]], None]
ChatClient = Callable[..., Iterable[Any]]


def get_current_weather(location: str) -> str:
    """Return deterministic example weather data for a supported location."""

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
    """Start a streaming model call using an injectable Ollama-compatible client."""

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
    """Execute one agent turn and return its complete observable trajectory."""

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
    """Run the original interactive terminal experience."""

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