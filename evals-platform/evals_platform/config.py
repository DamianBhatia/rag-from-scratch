"""Configuration and repository-relative paths for the eval platform."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = PACKAGE_ROOT.parent
REPOSITORY_ROOT = PLATFORM_ROOT.parent


@dataclass(frozen=True)
class AppConfig:
    database_path: Path
    case_suite_path: Path
    default_model: str
    default_max_iterations: int
    judge_model: str
    judge_enabled: bool


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    """Load defaults with optional environment overrides."""

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
