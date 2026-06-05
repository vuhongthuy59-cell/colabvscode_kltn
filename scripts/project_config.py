from __future__ import annotations

import os
from pathlib import Path

MAIN_START_DATE = "2022-01-01"
MAIN_END_DATE = "2025-12-31"
DEMO_START_DATE = "2026-01-01"
DEMO_END_DATE = "2026-04-30"

INCLUDE_2026_APPEND_ENV = "KLTN_INCLUDE_2026_APPEND"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
LOCAL_OUTPUT_ROOT = OUTPUT_ROOT / "local"
COLAB_OUTPUT_ROOT = OUTPUT_ROOT / "colab"
REPORT_OUTPUT_ROOT = OUTPUT_ROOT / "report"


def local_output(step_name: str) -> Path:
    return LOCAL_OUTPUT_ROOT / step_name


def colab_output(step_name: str) -> Path:
    return COLAB_OUTPUT_ROOT / step_name


def report_output(step_name: str) -> Path:
    return REPORT_OUTPUT_ROOT / step_name


def include_2026_append() -> bool:
    value = os.getenv(INCLUDE_2026_APPEND_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "y"}


def active_dataset_scope() -> str:
    if include_2026_append():
        return f"demo/robustness scope ({MAIN_START_DATE} to {DEMO_END_DATE})"
    return f"main thesis scope ({MAIN_START_DATE} to {MAIN_END_DATE})"
