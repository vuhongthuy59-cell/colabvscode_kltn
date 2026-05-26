from __future__ import annotations

import os

MAIN_START_DATE = "2022-01-01"
MAIN_END_DATE = "2025-12-31"
DEMO_START_DATE = "2026-01-01"
DEMO_END_DATE = "2026-04-30"

INCLUDE_2026_APPEND_ENV = "KLTN_INCLUDE_2026_APPEND"


def include_2026_append() -> bool:
    value = os.getenv(INCLUDE_2026_APPEND_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def active_dataset_scope() -> str:
    if include_2026_append():
        return f"demo/robustness scope ({MAIN_START_DATE} to {DEMO_END_DATE})"
    return f"main thesis scope ({MAIN_START_DATE} to {MAIN_END_DATE})"
