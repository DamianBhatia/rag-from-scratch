from __future__ import annotations

import sys
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PLATFORM_ROOT.parent
for path in (PLATFORM_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
