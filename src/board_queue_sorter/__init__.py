"""Drag-and-drop board case list with click-to-select."""

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_PARENT_DIR, "frontend", "build")

_component_func = components.declare_component("board_queue_sorter", path=_BUILD_DIR)


def board_queue_sorter(
    cases: list[dict[str, str]],
    active_id: str | None = None,
    key: Any = None,
) -> dict[str, Any] | None:
    """Render a sortable board case list.

    Each case dict uses keys: ``id``, ``title``, ``status``, ``status_class``.
    Returns ``{"order": [...], "selected": str, "event": "reorder"|"select"}``.
    """
    default_order = [case["id"] for case in cases]
    default = {
        "order": default_order,
        "selected": active_id or (default_order[0] if default_order else ""),
        "event": None,
    }
    value = _component_func(cases=cases, activeId=active_id or "", default=default, key=key)
    if not isinstance(value, dict):
        return None
    return value
