from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "15_gnn_blending_experiments"

BASELINE_PREDICTIONS = ROOT / "outputs" / "05_train_baselines" / "baseline_predictions.csv"
ABLATION_PREDICTIONS = ROOT / "outputs" / "06_train_gnn_ablation_models" / "ablation_predictions.csv"
RESIDUAL_PREDICTIONS = ROOT / "outputs" / "14_residual_gnn_experiments" / "residual_gnn_predictions.csv"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred, squared=False)),
    )


def prediction_frame(df: pd.DataFrame, output_name: str, mask: pd.Series) -> pd.DataFrame:
    cols = ["split", "snapshot_id", "node_idx", "y_true", "y_pred"]
    out = df.loc[mask, cols].copy()
    out = out.drop_duplicates(["split", "snapshot_id", "node_idx"])
    return out.rename(columns={"y_pred": output_name})


def load_wide_predictions() -> tuple[pd.DataFrame, list[str]]:
    baseline = pd.read_csv(BASELINE_PREDICTIONS)
    ablation = pd.read_csv(ABLATION_PREDICTIONS)
    residual = pd.read_csv(RESIDUAL_PREDICTIONS)

    frames = [
        prediction_frame(baseline, "linear", baseline["model"].eq("Linear Regression")),
        prediction_frame(baseline, "rf", baseline["model"].eq("Random Forest")),
        prediction_frame(ablation, "gnn_relationship", ablation["model"].eq("GNN + Relationship")),
        prediction_frame(
            residual,
            "residual_top30",
            residual["model"].eq("Residual TopK Graph MLP") & residual["variant"].eq("top30_h64_d010"),
        ),
        prediction_frame(
            residual,
            "residual_top50",
            residual["model"].eq("Residual TopK Graph MLP") & residual["variant"].eq("top50_h128_d015"),
        ),
    ]

    keys = ["split", "snapshot_id", "node_idx", "y_true"]
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on=keys, how="inner")
    return wide, ["linear", "rf", "gnn_relationship", "residual_top30", "residual_top50"]


def candidate_weights(n_models: int) -> list[np.ndarray]:
    weights = []
    for idx in range(n_models):
        weight = np.zeros(n_models)
        weight[idx] = 1.0
        weights.append(weight)

    for left in range(n_models):
        for right in range(left + 1, n_models):
            for alpha in np.linspace(0, 1, 101):
                weight = np.zeros(n_models)
                weight[left] = alpha
                weight[right] = 1.0 - alpha
                weights.append(weight)

    rng = np.random.default_rng(42)
    for _ in range(5000):
        weights.append(rng.dirichlet(np.ones(n_models)))
    return weights


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide, model_cols = load_wide_predictions()

    val = wide[wide["split"].eq("validation")].copy()
    test = wide[wide["split"].eq("test")].copy()
    x_val = val[model_cols].to_numpy()
    x_test = test[model_cols].to_numpy()
    y_val = val["y_true"].to_numpy()
    y_test = test["y_true"].to_numpy()

    result_rows = []
    for col in model_cols:
        for split_name, frame in [("validation", val), ("test", test)]:
            mae, rmse = metrics(frame["y_true"].to_numpy(), frame[col].to_numpy())
            result_rows.append(
                {
                    "model": col,
                    "split": split_name,
                    "mae": mae,
                    "rmse": rmse,
                    "weights": col,
                    "note": "single model prediction",
                }
            )

    best_weight = None
    best_val_mae = float("inf")
    for weight in candidate_weights(len(model_cols)):
        pred_val = x_val @ weight
        val_mae, _ = metrics(y_val, pred_val)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weight = weight

    if best_weight is None:
        raise RuntimeError("No candidate blend weight was evaluated.")

    weight_summary = "; ".join(f"{name}={weight:.4f}" for name, weight in zip(model_cols, best_weight))
    for split_name, x_split, y_split in [("validation", x_val, y_val), ("test", x_test, y_test)]:
        pred = x_split @ best_weight
        mae, rmse = metrics(y_split, pred)
        result_rows.append(
            {
                "model": "validation_selected_blend",
                "split": split_name,
                "mae": mae,
                "rmse": rmse,
                "weights": weight_summary,
                "note": "nonnegative blend weights selected on validation MAE",
            }
        )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "model"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "blend_results.csv", index=False)

    predictions = test[["split", "snapshot_id", "node_idx", "y_true"]].copy()
    predictions["y_pred"] = x_test @ best_weight
    predictions["model"] = "validation_selected_blend"
    predictions.to_csv(OUT_DIR / "blend_test_predictions.csv", index=False)

    print("Generated blend results:")
    print((OUT_DIR / "blend_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
