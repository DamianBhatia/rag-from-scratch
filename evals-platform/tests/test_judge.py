import pytest

from evals_platform.judge import OllamaJudge


def test_judge_parser_accepts_fenced_json():
    value = OllamaJudge.parse_response(
        '```json\n{"task_success": 1, "groundedness": 0.75, '
        '"tool_appropriateness": 0.5, "rationale": "Mostly good"}\n```'
    )
    assert value["groundedness"] == 0.75
    assert value["rationale"] == "Mostly good"


def test_judge_parser_rejects_invalid_score():
    with pytest.raises(ValueError):
        OllamaJudge.parse_response(
            '{"task_success": 2, "groundedness": 1, '
            '"tool_appropriateness": 1, "rationale": "bad"}'
        )
