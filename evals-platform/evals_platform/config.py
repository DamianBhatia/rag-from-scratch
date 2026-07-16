"""Environment-aware configuration and path discovery for the eval platform.

Paths are derived from this package's location rather than the process working
directory, allowing the dashboard and tests to start from different folders.
``load_config`` overlays a small set of ``EVALS_*`` environment variables on
local defaults and returns an immutable snapshot consumed by the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = PLATFORM_ROOT.parent


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime settings shared by the dashboard and evaluation services.

    ``database_path`` and ``case_suite_path`` identify local durable inputs and
    outputs. The remaining fields define default agent and optional judge model
    behavior; individual dashboard runs may override them without mutating this
    object.
    """

    database_path: Path
    case_suite_path: Path
    default_model: str
    default_max_iterations: int
    judge_model: str
    judge_enabled: bool


def _as_bool(value: str) -> bool:
    """Interpret common affirmative environment-variable values as ``True``."""

    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    """Resolve default settings and optional ``EVALS_*`` overrides.

    Relative path overrides remain relative to the process working directory;
    default paths are absolute and rooted under ``evals-platform``. Integer
    conversion errors are intentionally surfaced early as configuration errors.
    """

    return AppConfig(
        database_path=Path(
            os.getenv("EVALS_DATABASE_PATH", PLATFORM_ROOT / "data" / "evals.db")
        ).expanduser(),
        case_suite_path=Path(
            os.getenv(
                "EVALS_CASE_SUITE_PATH", PLATFORM_ROOT / "data" / "eval_cases.json"
            )
        ).expanduser(),
        default_model=os.getenv("EVALS_AGENT_MODEL", "llama3.2:3b"),
        default_max_iterations=int(os.getenv("EVALS_MAX_ITERATIONS", "4")),
        judge_model=os.getenv("EVALS_JUDGE_MODEL", "llama3.2:3b"),
        judge_enabled=_as_bool(os.getenv("EVALS_JUDGE_ENABLED", "false")),
    )
