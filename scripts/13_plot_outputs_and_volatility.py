from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from project_config import colab_output, local_output, report_output

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = report_output("10_report_assets")


def save_fig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=180)
    plt.close()


def load_predictions() -> pd.DataFrame:
    frames = []
    files = [
        local_output("06_baseline_models") / "baseline_predictions.csv",
        colab_output("07_gnn_ablation_models") / "ablation_predictions.csv",
        colab_output("12_hybrid_mlp_gat") / "hybrid_predictions.csv",
    ]
    for file in files:
        if file.exists():
            frames.append(pd.read_csv(file))
    if not frames:
        raise FileNotFoundError("No prediction files found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_target_distribution(snapshot_index: pd.DataFrame) -> None:
    cols = [
        ("target_raw_future_volatility_5d", "Raw future volatility 5d"),
        ("target_abnormal_volatility_5d", "Abnormal volatility 5d"),
        ("target_log_abnormal_volatility_5d", "Log abnormal volatility 5d"),
    ]
    plt.figure(figsize=(13, 4))
    for idx, (col, title) in enumerate(cols, start=1):
        plt.subplot(1, 3, idx)
        plt.hist(snapshot_index[col].dropna(), bins=60, color="#2f5d7c", alpha=0.85)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
    save_fig("figure_target_distribution_log_abnormal.png")


def plot_volatility_over_time(snapshot_index: pd.DataFrame) -> None:
    daily = (
        snapshot_index.groupby("event_trading_date")
        .agg(
            raw_future_vol=("target_raw_future_volatility_5d", "mean"),
            baseline_vol=("target_baseline_rolling_vol_20", "mean"),
            abnormal_vol=("target_abnormal_volatility_5d", "mean"),
            log_abnormal=("target_log_abnormal_volatility_5d", "mean"),
        )
        .reset_index()
    )
    daily["event_trading_date"] = pd.to_datetime(daily["event_trading_date"])
    daily = daily.sort_values("event_trading_date")

    plt.figure(figsize=(12, 5))
    plt.plot(daily["event_trading_date"], daily["raw_future_vol"], label="future realized vol 5d", linewidth=1.2)
    plt.plot(daily["event_trading_date"], daily["baseline_vol"], label="rolling vol 20d baseline", linewidth=1.2)
    plt.ylabel("Volatility")
    plt.title("Average raw future volatility vs baseline volatility by event date")
    plt.legend()
    save_fig("figure_volatility_raw_vs_baseline_by_date.png")

    plt.figure(figsize=(12, 5))
    plt.axhline(0, color="#444444", linewidth=0.8)
    plt.plot(daily["event_trading_date"], daily["abnormal_vol"], label="abnormal volatility", linewidth=1.2)
    plt.ylabel("future_vol_5d - rolling_vol_20")
    plt.title("Average abnormal volatility by event date")
    plt.legend()
    save_fig("figure_abnormal_volatility_by_date.png")


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    test = metrics[metrics["split"].eq("test")].sort_values("mae")
    x = range(len(test))
    plt.figure(figsize=(12, 5))
    plt.bar(x, test["mae"], color="#8a5a44", alpha=0.9)
    plt.xticks(list(x), test["model"], rotation=30, ha="right")
    plt.ylabel("MAE on log_abnormal_volatility_5d")
    plt.title("Test MAE comparison")
    save_fig("figure_log_abnormal_model_mae.png")


def plot_prediction_scatter(predictions: pd.DataFrame) -> None:
    test = predictions[predictions["split"].eq("test")].copy()
    keep_models = ["Linear Regression", "GNN + News", "Hybrid MLP-GAT"]
    test = test[test["model"].isin(keep_models)]

    plt.figure(figsize=(13, 4))
    for idx, model in enumerate(keep_models, start=1):
        data = test[test["model"].eq(model)]
        plt.subplot(1, 3, idx)
        plt.scatter(data["y_true"], data["y_pred"], s=8, alpha=0.35, color="#446b4f")
        low = min(data["y_true"].min(), data["y_pred"].min())
        high = max(data["y_true"].max(), data["y_pred"].max())
        plt.plot([low, high], [low, high], color="#222222", linewidth=1)
        plt.title(model)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
    save_fig("figure_prediction_scatter_log_abnormal.png")


def main() -> None:
    snapshot_index = pd.read_csv(local_output("05_event_graph_dataset") / "snapshot_index.csv")
    metrics = pd.read_csv(report_output("11_regression_metrics") / "r2_metrics_test.csv")
    predictions = load_predictions()

    plot_target_distribution(snapshot_index)
    plot_volatility_over_time(snapshot_index)
    plot_model_comparison(metrics)
    plot_prediction_scatter(predictions)

    print("Generated output/volatility plots:")
    for name in [
        "figure_target_distribution_log_abnormal.png",
        "figure_volatility_raw_vs_baseline_by_date.png",
        "figure_abnormal_volatility_by_date.png",
        "figure_log_abnormal_model_mae.png",
        "figure_prediction_scatter_log_abnormal.png",
    ]:
        print((OUT_DIR / name).as_posix())


if __name__ == "__main__":
    main()
