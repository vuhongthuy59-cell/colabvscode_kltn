from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from project_config import colab_output, local_output, report_output

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = report_output("11_regression_metrics")


def compute_group_metrics(df: pd.DataFrame, group_cols: list[str], source: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "source": source,
                "n": len(group),
                "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
                "rmse": float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))),
                "r2": float(r2_score(group["y_true"], group["y_pred"])),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_metric_frames() -> list[pd.DataFrame]:
    frames = []

    baseline_file = local_output("06_baseline_models") / "baseline_predictions.csv"
    if baseline_file.exists():
        baseline = pd.read_csv(baseline_file)
        frames.append(compute_group_metrics(baseline, ["model", "split"], "06_baseline_models"))

    ablation_file = colab_output("07_gnn_ablation_models") / "ablation_predictions.csv"
    if ablation_file.exists():
        ablation = pd.read_csv(ablation_file)
        frames.append(compute_group_metrics(ablation, ["model", "split"], "07_gnn_ablation_models"))

    hybrid_file = colab_output("12_hybrid_mlp_gat") / "hybrid_predictions.csv"
    if hybrid_file.exists():
        hybrid = pd.read_csv(hybrid_file)
        frames.append(compute_group_metrics(hybrid, ["model", "split"], "12_hybrid_mlp_gat"))

    return frames


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_metric_frames()
    if not frames:
        raise FileNotFoundError("No prediction files were found for R2 computation.")

    metrics = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = ["split", "r2", "mae", "model"]
    metrics = metrics.sort_values(sort_cols, ascending=[True, False, True, True]).reset_index(drop=True)
    metrics.to_csv(OUT_DIR / "r2_metrics.csv", index=False)

    test_metrics = metrics[metrics["split"].eq("test")].sort_values("r2", ascending=False)
    test_metrics.to_csv(OUT_DIR / "r2_metrics_test.csv", index=False)

    print("Generated R2 metrics:")
    print((OUT_DIR / "r2_metrics.csv").as_posix())
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
