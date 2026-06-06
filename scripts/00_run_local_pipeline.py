from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PIPELINE_STEPS = [
    ("01", "clean_price_data", "scripts/01_prepare_price_data.py"),
    ("02", "clean_news_data", "scripts/02_prepare_news_data.py"),
    ("03", "validate_news_labels", "scripts/03_train_news_labeler.py"),
    ("04", "build_company_relationships", "scripts/04_prepare_company_relationships.py"),
    ("05", "build_event_graph_dataset", "scripts/05_build_event_graph_dataset.py"),
    ("06", "train_tabular_baselines", "scripts/06_train_baseline_models.py"),
    ("07", "train_gnn_ablation", "scripts/07_train_gnn_ablation_models.py"),
    ("08", "tune_selected_gnn", "scripts/08_tune_selected_gnn.py"),
    ("12", "train_hybrid_mlp_gat", "scripts/12_train_hybrid_mlp_gat.py"),
    ("14", "train_residual_hybrid_gnn", "scripts/14_train_residual_hybrid_gnn.py"),
    ("09", "evaluate_and_report", "scripts/09_evaluate_and_report.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local thesis forecasting pipeline.")
    parser.add_argument(
        "--from-step",
        default=None,
        help="First step id/name to run, for example 05 or build_event_graph_dataset.",
    )
    parser.add_argument(
        "--to-step",
        default=None,
        help="Last step id/name to run, for example 09 or evaluate_and_report.",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip heavier GNN steps 07, 08, 12, 14 and only refresh data/baseline/report steps.",
    )
    return parser.parse_args()


def find_step_index(step_key: str | None, default: int) -> int:
    if step_key is None:
        return default
    normalized = step_key.strip().lower()
    for idx, (step_id, step_name, _) in enumerate(PIPELINE_STEPS):
        if normalized in {step_id.lower(), step_name.lower()}:
            return idx
    available = ", ".join(f"{step_id}:{step_name}" for step_id, step_name, _ in PIPELINE_STEPS)
    raise ValueError(f"Unknown step {step_key!r}. Available steps: {available}")


def run_step(step_id: str, step_name: str, script_path: str) -> None:
    started = time.time()
    print(f"\n[{step_id}] {step_name}")
    print(f"Running: {sys.executable} {script_path}")
    subprocess.run([sys.executable, script_path], cwd=ROOT, check=True)
    elapsed = time.time() - started
    print(f"Finished {step_id}:{step_name} in {elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    start_idx = find_step_index(args.from_step, 0)
    end_idx = find_step_index(args.to_step, len(PIPELINE_STEPS) - 1)
    if start_idx > end_idx:
        raise ValueError("--from-step must not come after --to-step")

    heavy_steps = {"07", "08", "12", "14"}
    selected_steps = PIPELINE_STEPS[start_idx : end_idx + 1]
    for step_id, step_name, script_path in selected_steps:
        if args.skip_heavy and step_id in heavy_steps:
            print(f"\n[{step_id}] {step_name}")
            print("Skipped because --skip-heavy was set.")
            continue
        run_step(step_id, step_name, script_path)

    print("\nLocal pipeline completed.")


if __name__ == "__main__":
    main()
