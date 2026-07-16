"""Typed contracts shared by evaluation, persistence, and presentation layers.

Evaluation cases are loaded from a versioned JSON suite into immutable records,
which prevents expectations from changing during a run. Settings similarly form
an immutable execution snapshot. ``EvaluationResult`` combines those inputs with
the mutable outputs produced by the agent, metrics, and optional judge before the
storage layer serializes them.

Keeping these records free of Streamlit, Ollama, and SQLite dependencies makes
them suitable for tests, alternate interfaces, and future case-suite migrations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpectedToolCall:
    """One expected tool name and exact normalized argument mapping."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExpectedToolCall":
        """Construct an expected call from its JSON-compatible representation."""

        return cls(name=value["name"], arguments=dict(value.get("arguments", {})))


@dataclass(frozen=True)
class EvalCase:
    """Versioned behavioral contract for one agent prompt.

    Expectations are optional and independently scored. Tool calls describe the
    required function/argument pairs; observation and answer terms measure text
    coverage; forbidden terms identify explicit answer violations. Tags and
    difficulty support dashboard filtering, while ``max_iterations`` can tighten
    or relax the run-level bound for this case only.
    """

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
        """Create a case from one object in the versioned JSON case suite."""

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
        """Return a serializable case snapshot using the external ``id`` key."""

        value = asdict(self)
        value["id"] = value.pop("case_id")
        return value


@dataclass(frozen=True)
class EvalSettings:
    """Immutable model and iteration settings captured for an evaluation run."""

    model: str
    max_iterations: int
    judge_enabled: bool = False
    judge_model: str | None = None


@dataclass
class EvaluationResult:
    """Persistable output for one case within a larger evaluation run.

    ``agent_result`` is the serialized canonical trajectory, ``metrics`` contains
    deterministic measurements, and ``judge`` contains optional model-based
    scoring or a structured judge error.
    """

    result_id: str
    run_id: str
    case: EvalCase
    status: str
    agent_result: dict[str, Any]
    metrics: dict[str, Any]
    judge: dict[str, Any] | None = None
    created_at: str = ""


def load_case_suite(path: Path) -> list[EvalCase]:
    """Load and validate a version-1 JSON case suite.

    Validation checks the top-level shape, supported schema version, and case-ID
    uniqueness. Individual required keys are validated naturally while records
    are constructed, so malformed suites fail before any run is created.

    Args:
        path: File containing ``schema_version`` and a top-level ``cases`` array.

    Returns:
        Cases in their source order.
    """

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
