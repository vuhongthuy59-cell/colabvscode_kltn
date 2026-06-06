from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from project_config import local_output, report_output

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = report_output("09_model_evaluation")
ASSET_DIR = report_output("10_report_assets")
METRIC_DIR = report_output("11_regression_metrics")


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_fig(name: str) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / name, dpi=180)
    plt.close()


def load_predictions() -> pd.DataFrame:
    files = [
        ("06_baseline_models", local_output("06_baseline_models") / "baseline_predictions.csv"),
        ("07_gnn_ablation_models", local_output("07_gnn_ablation_models") / "ablation_predictions.csv"),
        ("12_hybrid_mlp_gat", local_output("12_hybrid_mlp_gat") / "hybrid_predictions.csv"),
        ("14_residual_hybrid_gnn", local_output("14_residual_hybrid_gnn") / "residual_predictions.csv"),
    ]
    frames = []
    for source, file in files:
        if not file.exists():
            continue
        frame = pd.read_csv(file)
        frame["source"] = source
        frame = normalize_model_names(frame)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No prediction files found for evaluation.")
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ablation_corr = df["source"].eq("07_gnn_ablation_models") & df["model"].eq("GNN Correlation Only")
    df.loc[ablation_corr, "model"] = "GNN Correlation Only (Ablation)"
    return df


def compute_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (source, model, split), group in predictions.groupby(["source", "model", "split"]):
        rows.append(
            {
                "model": model,
                "split": split,
                "source": source,
                "n": len(group),
                "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
                "rmse": rmse(group["y_true"], group["y_pred"]),
                "r2": float(r2_score(group["y_true"], group["y_pred"])),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["split", "r2", "mae"], ascending=[True, False, True])
    test_metrics = metrics[metrics["split"].eq("test")].sort_values("r2", ascending=False)
    return metrics.reset_index(drop=True), test_metrics.reset_index(drop=True)


def add_context(predictions: pd.DataFrame) -> pd.DataFrame:
    snapshot_index = pd.read_csv(local_output("05_event_graph_dataset") / "snapshot_index.csv")
    articles = pd.read_csv(local_output("02_news_data") / "news_articles.csv")
    ticker_map = pd.read_json(local_output("05_event_graph_dataset") / "ticker_to_idx.json", typ="series")
    idx_to_ticker = {int(idx): ticker for ticker, idx in ticker_map.items()}

    edge_count_cols = [col for col in snapshot_index.columns if col.startswith("num_edges_")]
    context_cols = [
        "snapshot_id",
        "category",
        "general_sentiment",
        "n_mapped_tickers",
        "target_count",
        "num_edges",
    ] + edge_count_cols
    enriched = predictions.merge(snapshot_index[context_cols], on="snapshot_id", how="left")
    enriched = enriched.merge(articles[["article_id", "title", "sentiment_label"]], on="article_id", how="left")
    enriched["ticker"] = enriched["node_idx"].map(idx_to_ticker)
    enriched["absolute_error"] = (enriched["y_true"] - enriched["y_pred"]).abs()
    return enriched


def grouped_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": len(group),
                "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
                "rmse": rmse(group["y_true"], group["y_pred"]),
                "mean_y_true": float(group["y_true"].mean()),
                "mean_y_pred": float(group["y_pred"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols + ["mae"]).reset_index(drop=True)


def build_case_studies(enriched: pd.DataFrame) -> pd.DataFrame:
    test_df = enriched[enriched["split"].eq("test")].copy()
    best_model = grouped_metrics(test_df, ["model"]).sort_values("mae").iloc[0]["model"]
    model_df = test_df[test_df["model"].eq(best_model)].copy().sort_values("absolute_error")

    best_cases = model_df.head(15).assign(case_type="lowest_error")
    worst_cases = model_df.tail(15).sort_values("absolute_error", ascending=False).assign(case_type="highest_error")
    high_vol_cases = model_df.sort_values("y_true", ascending=False).head(15).assign(case_type="highest_realized_volatility")

    cols = [
        "case_type",
        "model",
        "split",
        "article_id",
        "event_trading_date",
        "ticker",
        "category",
        "sentiment_label",
        "general_sentiment",
        "y_true",
        "y_pred",
        "absolute_error",
        "title",
    ]
    return pd.concat([best_cases, worst_cases, high_vol_cases], ignore_index=True)[cols]


def load_model_result_tables() -> pd.DataFrame:
    files = [
        ("06_baseline_models", local_output("06_baseline_models") / "baseline_results.csv"),
        ("07_gnn_ablation_models", local_output("07_gnn_ablation_models") / "ablation_results.csv"),
        ("12_hybrid_mlp_gat", local_output("12_hybrid_mlp_gat") / "hybrid_results.csv"),
        ("14_residual_hybrid_gnn", local_output("14_residual_hybrid_gnn") / "residual_results.csv"),
    ]
    frames = []
    for source, file in files:
        if not file.exists():
            continue
        frame = pd.read_csv(file)
        frame["source"] = source
        frame = normalize_model_names(frame)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No model result files found.")
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(["split", "mae", "model"]).reset_index(drop=True)


def plot_model_metrics(comparison: pd.DataFrame) -> None:
    test = comparison[comparison["split"].eq("test")].copy().sort_values("mae")
    x = range(len(test))
    width = 0.38
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], test["mae"], width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], test["rmse"], width=width, label="RMSE")
    plt.xticks(list(x), test["model"], rotation=25, ha="right")
    plt.ylabel("Error")
    plt.title("Test-set model comparison")
    plt.legend()
    save_fig("figure_model_comparison_test.png")


def plot_rf_importance() -> None:
    file = local_output("06_baseline_models") / "rf_feature_importance.csv"
    if not file.exists():
        return
    importance = pd.read_csv(file).head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(importance["feature"], importance["importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance")
    save_fig("figure_rf_feature_importance_top15.png")


def plot_training_curves() -> None:
    file = local_output("07_gnn_ablation_models") / "full_model_training_log.csv"
    if not file.exists():
        return
    log = pd.read_csv(file)
    plt.figure(figsize=(10, 5))
    for model, group in log.groupby("model"):
        plt.plot(group["epoch"], group["val_mae"], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("GNN ablation validation curves")
    plt.legend()
    save_fig("figure_gnn_validation_mae.png")


def plot_category_error(category_metrics: pd.DataFrame) -> None:
    best_model = category_metrics.groupby("model")["mae"].mean().sort_values().index[0]
    data = category_metrics[category_metrics["model"].eq(best_model)].sort_values("mae")
    plt.figure(figsize=(10, 5))
    plt.bar(data["category"], data["mae"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title(f"Test MAE by news category: {best_model}")
    save_fig("figure_best_model_category_mae.png")


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


def plot_mae_from_metrics(test_metrics: pd.DataFrame) -> None:
    test = test_metrics.copy().sort_values("mae")
    x = range(len(test))
    plt.figure(figsize=(12, 5))
    plt.bar(x, test["mae"], color="#8a5a44", alpha=0.9)
    plt.xticks(list(x), test["model"], rotation=30, ha="right")
    plt.ylabel("MAE on log_abnormal_volatility_5d")
    plt.title("Test MAE comparison")
    save_fig("figure_log_abnormal_model_mae.png")


def plot_prediction_scatter(predictions: pd.DataFrame) -> None:
    test = predictions[predictions["split"].eq("test")].copy()
    keep_models = [
        "Linear Regression",
        "Residual Hybrid GNN (MAE shrinkage)",
        "Residual Hybrid GNN (R2 tuned)",
    ]
    test = test[test["model"].isin(keep_models)]
    keep_models = [model for model in keep_models if model in set(test["model"])]
    if not keep_models:
        return

    plt.figure(figsize=(max(5, 4 * len(keep_models)), 4))
    for idx, model in enumerate(keep_models, start=1):
        data = test[test["model"].eq(model)]
        plt.subplot(1, len(keep_models), idx)
        plt.scatter(data["y_true"], data["y_pred"], s=8, alpha=0.35, color="#446b4f")
        low = min(data["y_true"].min(), data["y_pred"].min())
        high = max(data["y_true"].max(), data["y_pred"].max())
        plt.plot([low, high], [low, high], color="#222222", linewidth=1)
        plt.title(model)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
    save_fig("figure_prediction_scatter_log_abnormal.png")


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    metrics, test_metrics = compute_metrics(predictions)
    enriched = add_context(predictions)
    test_enriched = enriched[enriched["split"].eq("test")]
    category_metrics = grouped_metrics(test_enriched, ["model", "category"])
    ticker_metrics = grouped_metrics(test_enriched, ["model", "ticker"])
    daily_metrics = grouped_metrics(test_enriched, ["model", "event_trading_date"])
    case_studies = build_case_studies(enriched)
    comparison = load_model_result_tables()

    test_table = comparison[comparison["split"].eq("test")].copy()
    keep_cols = [
        "model",
        "input",
        "graph",
        "price_features",
        "news_features",
        "relationship_edges",
        "co_mention_edges",
        "mae",
        "rmse",
        "note",
    ]
    test_table = test_table[[col for col in keep_cols if col in test_table.columns]]
    for col in test_table.columns:
        if test_table[col].dtype == object:
            test_table[col] = test_table[col].fillna("")

    save_csv(metrics, METRIC_DIR / "r2_metrics.csv")
    save_csv(test_metrics, METRIC_DIR / "r2_metrics_test.csv")
    save_csv(enriched, EVAL_DIR / "all_model_predictions_enriched.csv")
    save_csv(category_metrics, EVAL_DIR / "model_error_by_category.csv")
    save_csv(ticker_metrics, EVAL_DIR / "model_error_by_ticker.csv")
    save_csv(daily_metrics, EVAL_DIR / "model_error_by_event_date.csv")
    save_csv(case_studies, EVAL_DIR / "case_study_results.csv")
    save_csv(comparison, ASSET_DIR / "table_all_model_metrics.csv")
    save_csv(test_table, ASSET_DIR / "table_model_comparison_test.csv")
    save_csv(category_metrics, ASSET_DIR / "table_error_by_category.csv")
    save_csv(ticker_metrics, ASSET_DIR / "table_error_by_ticker.csv")
    save_csv(case_studies, ASSET_DIR / "table_case_studies.csv")

    snapshot_index = pd.read_csv(local_output("05_event_graph_dataset") / "snapshot_index.csv")
    plot_model_metrics(comparison)
    plot_rf_importance()
    plot_training_curves()
    plot_category_error(category_metrics)
    plot_target_distribution(snapshot_index)
    plot_volatility_over_time(snapshot_index)
    plot_mae_from_metrics(test_metrics)
    plot_prediction_scatter(predictions)

    print("Generated evaluation metrics, case studies, report tables and figures.")
    print((METRIC_DIR / "r2_metrics_test.csv").as_posix())
    print(test_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
