"""Pytest path bootstrap for evaluation-platform tests.

Tests import both the nested ``evals_platform`` package and the repository-level
``ReAct`` package. Adding those two roots to ``sys.path`` keeps test invocation
independent of the current working directory and avoids requiring an editable
package installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PLATFORM_ROOT.parent
for path in (PLATFORM_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
