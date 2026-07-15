from __future__ import annotations

from ReAct.main import run_agent


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return iter(response)


def tool_chunk(name="get_current_weather", arguments=None):
    return [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": name,
                            "arguments": arguments or {"location": "London"},
                        }
                    }
                ],
            }
        }
    ]


def test_run_agent_captures_one_tool_trajectory_and_coherent_messages():
    client = FakeChat(
        [
            tool_chunk(),
            [
                {"message": {"content": "London is ", "tool_calls": []}},
                {"message": {"content": "rainy.", "tool_calls": []}},
            ],
        ]
    )

    result = run_agent("Weather in London?", chat_client=client)

    assert result.status == "completed"
    assert result.final_answer == "London is rainy."
    assert [step.kind for step in result.steps] == ["model", "tool", "model"]
    assert result.steps[1].observation.startswith("15°C")
    assistant_messages = [m for m in result.messages if m["role"] == "assistant"]
    assert len(assistant_messages) == 2
    assert assistant_messages[-1]["content"] == "London is rainy."


def test_run_agent_no_tool_and_model_error():
    completed = run_agent(
        "Say hello",
        chat_client=FakeChat(
            [[{"message": {"content": "Hello!", "tool_calls": []}}]]
        ),
    )
    assert completed.termination_reason == "final_answer"
    assert completed.iterations == 1

    failed = run_agent("Say hello", chat_client=FakeChat([RuntimeError("offline")]))
    assert failed.status == "failed"
    assert failed.termination_reason == "model_error"
    assert failed.error_type == "RuntimeError"


def test_unknown_tool_is_observed_and_max_iterations_is_reported():
    client = FakeChat(
        [
            tool_chunk("missing_tool", {"value": 1}),
            [[{"message": {"content": "Recovered", "tool_calls": []}}][0]],
        ]
    )
    recovered = run_agent("Use a tool", chat_client=client)
    assert recovered.status == "completed_with_errors"
    tool_step = next(step for step in recovered.steps if step.kind == "tool")
    assert tool_step.success is False
    assert tool_step.error_type == "LookupError"

    maxed = run_agent(
        "Keep using tools",
        max_iterations=1,
        chat_client=FakeChat([tool_chunk()]),
    )
    assert maxed.status == "failed"
    assert maxed.termination_reason == "max_iterations"
