from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "09_model_evaluation"


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def add_context(predictions: pd.DataFrame) -> pd.DataFrame:
    snapshot_index = pd.read_csv(ROOT / "outputs" / "05_event_graph_dataset" / "snapshot_index.csv")
    articles = pd.read_csv(ROOT / "outputs" / "02_news_data" / "news_articles.csv")
    ticker_map = pd.read_json(ROOT / "outputs" / "05_event_graph_dataset" / "ticker_to_idx.json", typ="series")
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
    article_cols = ["article_id", "title", "sentiment_label"]
    enriched = predictions.merge(snapshot_index[context_cols], on="snapshot_id", how="left")
    enriched = enriched.merge(articles[article_cols], on="article_id", how="left")
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


def build_case_studies(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df[df["split"].eq("test")].copy()
    best_model = grouped_metrics(test_df, ["model"]).sort_values("mae").iloc[0]["model"]
    model_df = test_df[test_df["model"].eq(best_model)].copy()
    model_df = model_df.sort_values("absolute_error")

    best_cases = model_df.head(15).assign(case_type="lowest_error")
    worst_cases = model_df.tail(15).sort_values("absolute_error", ascending=False).assign(case_type="highest_error")
    high_vol_cases = (
        model_df.sort_values("y_true", ascending=False)
        .head(15)
        .assign(case_type="highest_realized_volatility")
    )

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ablation_predictions = pd.read_csv(ROOT / "outputs" / "07_gnn_ablation_models" / "ablation_predictions.csv")
    baseline_predictions = pd.read_csv(ROOT / "outputs" / "06_baseline_models" / "baseline_predictions.csv")
    frames = [baseline_predictions, ablation_predictions]
    hybrid_file = ROOT / "outputs" / "12_hybrid_mlp_gat" / "hybrid_predictions.csv"
    if hybrid_file.exists():
        frames.append(pd.read_csv(hybrid_file))
    predictions = pd.concat(frames, ignore_index=True)
    enriched = add_context(predictions)

    category_metrics = grouped_metrics(enriched[enriched["split"].eq("test")], ["model", "category"])
    ticker_metrics = grouped_metrics(enriched[enriched["split"].eq("test")], ["model", "ticker"])
    daily_metrics = grouped_metrics(enriched[enriched["split"].eq("test")], ["model", "event_trading_date"])
    case_studies = build_case_studies(enriched)

    enriched.to_csv(OUT_DIR / "all_model_predictions_enriched.csv", index=False)
    category_metrics.to_csv(OUT_DIR / "model_error_by_category.csv", index=False)
    ticker_metrics.to_csv(OUT_DIR / "model_error_by_ticker.csv", index=False)
    daily_metrics.to_csv(OUT_DIR / "model_error_by_event_date.csv", index=False)
    case_studies.to_csv(OUT_DIR / "case_study_results.csv", index=False)

    print("Generated evaluation and case-study outputs.")
    print(f"Case studies: {len(case_studies)}")
    print(f"Category metric rows: {len(category_metrics)}")


if __name__ == "__main__":
    main()
