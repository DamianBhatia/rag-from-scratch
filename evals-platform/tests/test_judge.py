"""Unit tests for strict parsing of optional LLM-judge responses.

These tests stay offline by exercising only JSON extraction and score validation.
They cover a common fenced response emitted by chat models and rejection of
scores outside the documented zero-to-one range.
"""

import pytest

from evals_platform.judge import OllamaJudge


def test_judge_parser_accepts_fenced_json():
    """Markdown-fenced judge JSON is extracted and normalized successfully."""

    value = OllamaJudge.parse_response(
        '```json\n{"task_success": 1, "groundedness": 0.75, '
        '"tool_appropriateness": 0.5, "rationale": "Mostly good"}\n```'
    )
    assert value["groundedness"] == 0.75
    assert value["rationale"] == "Mostly good"


def test_judge_parser_rejects_invalid_score():
    """Out-of-range judge scores fail validation instead of entering storage."""

    with pytest.raises(ValueError):
        OllamaJudge.parse_response(
            '{"task_success": 2, "groundedness": 1, '
            '"tool_appropriateness": 1, "rationale": "bad"}'
        )
