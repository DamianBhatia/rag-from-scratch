"""Explainable deterministic metrics for agent trajectories and final answers.

Scoring compares the case contract with recorded tool steps and answer text. Tool
calls are matched as an unordered multiset after case/whitespace normalization,
so repeated calls are counted correctly while harmless text casing differences
are ignored. Argument quality, expected observation/answer term recall, forbidden
terms, execution success, iteration behavior, and latency are reported separately
instead of collapsing behavior into one opaque score.

The module accepts dataclass-, object-, or dictionary-style agent results. This
keeps scoring usable both immediately after execution and after JSON round trips.
No model calls occur here, making results repeatable for the same inputs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

from .models import EvalCase


def _step_dict(step: Any) -> dict[str, Any]:
    """Convert supported trajectory step representations to a plain mapping."""

    if isinstance(step, dict):
        return step
    if is_dataclass(step):
        return asdict(step)
    return vars(step)


def _normalize(value: Any) -> Any:
    """Recursively normalize strings and mapping keys for exact comparisons."""

    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return {str(key).casefold(): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _term_recall(terms: tuple[str, ...], text: str) -> float | None:
    """Return case-insensitive substring recall, or ``None`` without expectations."""

    if not terms:
        return None
    normalized = " ".join(text.casefold().split())
    return sum(" ".join(term.casefold().split()) in normalized for term in terms) / len(terms)


def _tool_match_metrics(
    expected: list[tuple[str, Any]], actual: list[tuple[str, Any]]
) -> tuple[float, float, float]:
    """Calculate multiset precision, recall, and F1 for normalized tool calls."""

    if not expected and not actual:
        return 1.0, 1.0, 1.0
    remaining = list(actual)
    matched = 0
    for expected_call in expected:
        try:
            index = remaining.index(expected_call)
        except ValueError:
            continue
        matched += 1
        remaining.pop(index)
    precision = matched / len(actual) if actual else 0.0
    recall = matched / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _argument_metrics(
    expected_calls: list[tuple[str, dict[str, Any]]],
    actual_calls: list[tuple[str, dict[str, Any]]],
) -> tuple[float | None, float | None]:
    """Measure expected argument value accuracy and expected-key presence.

    Calls are paired by tool name, choosing the remaining candidate with the most
    matching values. This avoids penalizing multi-call cases solely because the
    model emitted valid calls in a different order.
    """

    if not expected_calls:
        return None, None
    remaining = list(actual_calls)
    expected_key_count = 0
    present_key_count = 0
    matching_value_count = 0

    for expected_name, expected_args in expected_calls:
        candidates = [
            (index, call)
            for index, call in enumerate(remaining)
            if call[0] == expected_name
        ]
        candidate_index = None
        if candidates:
            candidate_index = max(
                candidates,
                key=lambda candidate: sum(
                    key in candidate[1][1]
                    and _normalize(candidate[1][1][key]) == _normalize(expected_value)
                    for key, expected_value in expected_args.items()
                ),
            )[0]
        if candidate_index is None:
            expected_key_count += max(1, len(expected_args))
            continue
        _, actual_args = remaining.pop(candidate_index)
        if not expected_args:
            expected_key_count += 1
            present_key_count += 1
            matching_value_count += 1
            continue
        for key, expected_value in expected_args.items():
            expected_key_count += 1
            if key in actual_args:
                present_key_count += 1
                if _normalize(actual_args[key]) == _normalize(expected_value):
                    matching_value_count += 1

    if not expected_key_count:
        return None, None
    return matching_value_count / expected_key_count, present_key_count / expected_key_count


def calculate_metrics(case: EvalCase, agent_result: Any) -> dict[str, Any]:
    """Calculate all deterministic metrics for one case and agent trajectory.

    Metrics with no applicable expectation are returned as ``None`` rather than
    implying success or failure. The exception is tool-call scoring: when neither
    expected nor actual calls exist, precision, recall, and F1 are all ``1.0``.
    Timing and iteration fields are copied into the same flat metric dictionary
    so storage and charting can consume one consistent payload.
    """

    raw_steps = (
        agent_result.get("steps", [])
        if isinstance(agent_result, dict)
        else getattr(agent_result, "steps", [])
    )
    steps = [_step_dict(step) for step in raw_steps]
    tool_steps = [step for step in steps if step.get("kind") == "tool"]

    expected_calls_raw = [
        (call.name, dict(call.arguments)) for call in case.expected_tool_calls
    ]
    actual_calls_raw = [
        (step.get("tool_name") or "", dict(step.get("arguments") or {}))
        for step in tool_steps
    ]
    expected_calls = [(_normalize(name), _normalize(args)) for name, args in expected_calls_raw]
    actual_calls = [(_normalize(name), _normalize(args)) for name, args in actual_calls_raw]
    precision, recall, f1 = _tool_match_metrics(expected_calls, actual_calls)
    argument_match, argument_completeness = _argument_metrics(
        [(_normalize(name), args) for name, args in expected_calls_raw],
        [(_normalize(name), args) for name, args in actual_calls_raw],
    )

    observations = "\n".join(str(step.get("observation") or "") for step in tool_steps)
    def result_value(name: str, default: Any) -> Any:
        if isinstance(agent_result, dict):
            return agent_result.get(name, default)
        return getattr(agent_result, name, default)

    final_answer = str(result_value("final_answer", ""))
    forbidden_pattern = [
        term
        for term in case.forbidden_answer_terms
        if re.search(re.escape(term), final_answer, flags=re.IGNORECASE)
    ]
    successful_tools = sum(bool(step.get("success")) for step in tool_steps)

    return {
        "tool_call_precision": precision,
        "tool_call_recall": recall,
        "tool_call_f1": f1,
        "argument_match": argument_match,
        "argument_completeness": argument_completeness,
        "observation_term_recall": _term_recall(
            case.expected_observation_terms, observations
        ),
        "answer_term_recall": _term_recall(case.expected_answer_terms, final_answer),
        "forbidden_term_violation": bool(forbidden_pattern),
        "forbidden_terms_found": forbidden_pattern,
        "tool_execution_success_rate": (
            successful_tools / len(tool_steps) if tool_steps else 1.0
        ),
        "tool_call_count": len(tool_steps),
        "iteration_count": int(result_value("iterations", 0)),
        "max_iterations_reached": (
            result_value("termination_reason", "") == "max_iterations"
        ),
        "model_latency_ms": float(result_value("model_latency_ms", 0.0)),
        "tool_latency_ms": float(result_value("tool_latency_ms", 0.0)),
        "total_latency_ms": float(result_value("total_latency_ms", 0.0)),
    }
