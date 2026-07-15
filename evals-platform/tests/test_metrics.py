from __future__ import annotations

from types import SimpleNamespace

from evals_platform.metrics import calculate_metrics
from evals_platform.models import EvalCase, ExpectedToolCall


def result(steps, answer="", termination="final_answer"):
    return SimpleNamespace(
        steps=steps,
        final_answer=answer,
        iterations=2,
        termination_reason=termination,
        model_latency_ms=10,
        tool_latency_ms=2,
        total_latency_ms=12,
    )


def test_unordered_tool_calls_match_and_text_metrics_are_calculated():
    case = EvalCase(
        case_id="multi",
        prompt="compare",
        expected_tool_calls=(
            ExpectedToolCall("weather", {"location": "London"}),
            ExpectedToolCall("weather", {"location": "Tokyo"}),
        ),
        expected_observation_terms=("rainy", "sunny"),
        expected_answer_terms=("London", "Tokyo"),
        forbidden_answer_terms=("snow",),
    )
    actual = result(
        [
            {"kind": "tool", "tool_name": "weather", "arguments": {"location": " tokyo "}, "observation": "Sunny", "success": True},
            {"kind": "tool", "tool_name": "weather", "arguments": {"location": "LONDON"}, "observation": "Rainy", "success": True},
        ],
        "Tokyo is sunny and London is rainy.",
    )

    metrics = calculate_metrics(case, actual)

    assert metrics["tool_call_f1"] == 1
    assert metrics["argument_match"] == 1
    assert metrics["observation_term_recall"] == 1
    assert metrics["answer_term_recall"] == 1
    assert metrics["forbidden_term_violation"] is False


def test_missing_extra_and_no_tool_edge_cases():
    expected = EvalCase(
        case_id="one",
        prompt="test",
        expected_tool_calls=(ExpectedToolCall("weather", {"location": "London"}),),
    )
    metrics = calculate_metrics(
        expected,
        result(
            [
                {"kind": "tool", "tool_name": "weather", "arguments": {"location": "London"}, "success": True},
                {"kind": "tool", "tool_name": "extra", "arguments": {}, "success": False},
            ]
        ),
    )
    assert metrics["tool_call_precision"] == 0.5
    assert metrics["tool_call_recall"] == 1
    assert metrics["tool_execution_success_rate"] == 0.5

    no_tools = calculate_metrics(EvalCase(case_id="none", prompt="hello"), result([]))
    assert no_tools["tool_call_f1"] == 1
    assert no_tools["argument_match"] is None
    assert no_tools["answer_term_recall"] is None
