"""Ensure the ui-chat directory is importable when tests run from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

UI_CHAT_ROOT = Path(__file__).resolve().parents[2]
if str(UI_CHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_CHAT_ROOT))
