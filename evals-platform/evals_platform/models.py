"""Typed records shared by evaluation, persistence, and UI layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpectedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpectedToolCall":
        return cls(name=value["name"], arguments=dict(value.get("arguments", {})))


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    prompt: str
    expected_tool_calls: tuple[ExpectedToolCall, ...] = ()
    expected_observation_terms: tuple[str, ...] = ()
    expected_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    difficulty: str = "unspecified"
    max_iterations: int | None = None
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value: dict[str, Any], schema_version: int = 1) -> "EvalCase":
        return cls(
            case_id=value["id"],
            prompt=value["prompt"],
            expected_tool_calls=tuple(
                ExpectedToolCall.from_dict(item)
                for item in value.get("expected_tool_calls", [])
            ),
            expected_observation_terms=tuple(value.get("expected_observation_terms", [])),
            expected_answer_terms=tuple(value.get("expected_answer_terms", [])),
            forbidden_answer_terms=tuple(value.get("forbidden_answer_terms", [])),
            tags=tuple(value.get("tags", [])),
            difficulty=value.get("difficulty", "unspecified"),
            max_iterations=value.get("max_iterations"),
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["id"] = value.pop("case_id")
        return value


@dataclass(frozen=True)
class EvalSettings:
    model: str
    max_iterations: int
    judge_enabled: bool = False
    judge_model: str | None = None


@dataclass
class EvaluationResult:
    result_id: str
    run_id: str
    case: EvalCase
    status: str
    agent_result: dict[str, Any]
    metrics: dict[str, Any]
    judge: dict[str, Any] | None = None
    created_at: str = ""


def load_case_suite(path: Path) -> list[EvalCase]:
    """Load and validate the versioned JSON case suite."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("Case suite must contain a top-level 'cases' array.")
    schema_version = int(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported case suite schema version: {schema_version}")
    cases = [EvalCase.from_dict(item, schema_version) for item in payload["cases"]]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case IDs must be unique.")
    return cases
