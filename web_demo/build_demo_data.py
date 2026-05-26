from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "web_demo" / "data" / "demo-data.json"

MAX_ARTICLES = 1500
MODELS = [
    "GNN + News",
    "Full Model",
    "GNN + Relationship",
    "GNN Correlation Only",
    "Random Forest",
    "Linear Regression",
    "Rolling Volatility",
]


def clean_record(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [
        {key: clean_record(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def main() -> None:
    predictions = pd.read_csv(PROCESSED / "all_model_predictions_enriched.csv")
    articles = pd.read_csv(PROCESSED / "news_articles.csv")
    mentions = pd.read_csv(PROCESSED / "news_mentions.csv")
    relationships = pd.read_csv(PROCESSED / "company_relationships.csv")
    prices = pd.read_csv(PROCESSED / "stock_prices.csv")
    ticker_metadata = pd.read_csv(PROCESSED / "ticker_metadata.csv")
    ticker_aliases = pd.read_csv(PROCESSED / "ticker_aliases.csv")
    model_metrics = pd.read_csv(PROCESSED / "report_outputs" / "table_all_model_metrics.csv")
    case_studies = pd.read_csv(PROCESSED / "case_study_results.csv")
    category_metrics = pd.read_csv(PROCESSED / "model_error_by_category.csv")

    test_predictions = predictions[predictions["split"].eq("test")].copy()
    model_pool = [model for model in MODELS if model in set(test_predictions["model"])]
    primary_model = "GNN + News" if "GNN + News" in model_pool else model_pool[0]

    primary = test_predictions[test_predictions["model"].eq(primary_model)].copy()
    primary["rank_score"] = primary["absolute_error"].rank(method="first")

    case_article_ids = case_studies["article_id"].dropna().astype(str).unique().tolist()
    recent_ids = (
        primary.sort_values(["event_trading_date", "article_id"], ascending=[False, False])
        ["article_id"]
        .drop_duplicates()
        .head(MAX_ARTICLES // 2)
        .tolist()
    )
    diverse_ids = (
        primary.sort_values(["category", "rank_score"])
        ["article_id"]
        .drop_duplicates()
        .head(MAX_ARTICLES)
        .tolist()
    )

    selected_ids: list[str] = []
    for article_id in [*case_article_ids, *recent_ids, *diverse_ids]:
        if article_id not in selected_ids:
            selected_ids.append(article_id)
        if len(selected_ids) >= MAX_ARTICLES:
            break

    selected_predictions = test_predictions[test_predictions["model"].isin(model_pool)].copy()
    selected_predictions = selected_predictions[
        [
            "model",
            "article_id",
            "event_trading_date",
            "ticker",
            "y_true",
            "y_pred",
            "absolute_error",
            "category",
            "sentiment_label",
            "general_sentiment",
        ]
    ].sort_values(["article_id", "ticker", "model"])

    selected_articles = articles.copy()
    selected_articles = selected_articles[
        [
            "article_id",
            "published_date",
            "event_trading_date",
            "source",
            "title",
            "category",
            "general_sentiment",
            "sentiment_label",
            "n_mapped_tickers",
        ]
    ].sort_values(["event_trading_date", "article_id"], ascending=[False, False])

    selected_mentions = mentions.copy()
    selected_mentions = selected_mentions[
        [
            "article_id",
            "ticker",
            "company_name",
            "is_primary",
            "mention_count",
            "relevance_score",
            "company_sentiment",
            "mapping_method",
            "matched_text",
        ]
    ].sort_values(["article_id", "is_primary", "relevance_score"], ascending=[True, False, False])

    selected_tickers = sorted(
        set(selected_predictions["ticker"].dropna().astype(str))
        | set(selected_mentions["ticker"].dropna().astype(str))
    )

    selected_relationships = relationships.copy()
    selected_relationships = selected_relationships[
        ["source_ticker", "target_ticker", "relation_type", "weight", "is_directed"]
    ].drop_duplicates()

    selected_prices = prices[["date", "ticker", "close", "volume"]].copy().sort_values(["ticker", "date"])
    ticker_metadata = ticker_metadata[["ticker", "company_name", "industry", "sector", "exchange"]].copy()
    ticker_aliases = ticker_aliases[["ticker", "company_name", "alias", "mapping_method"]].copy()

    metrics = model_metrics[model_metrics["split"].eq("test")].copy()
    metrics = metrics[metrics["model"].isin(model_pool)]
    metrics = metrics[
        [
            "model",
            "mae",
            "rmse",
            "price_features",
            "news_features",
            "relationship_edges",
            "co_mention_edges",
        ]
    ].sort_values("mae")

    category_summary = category_metrics[category_metrics["model"].eq(primary_model)].copy()
    category_summary = category_summary[["category", "n", "mae", "rmse"]].sort_values("mae")

    payload = {
        "meta": {
            "generated_from": "data/processed",
            "primary_model": primary_model,
            "article_count": int(selected_articles["article_id"].nunique()),
            "available_processed_articles": int(articles["article_id"].nunique()),
            "model_ready_article_count": int(selected_predictions["article_id"].nunique()),
            "default_render_limit": 500,
            "ticker_count": int(prices["ticker"].nunique()),
            "models": model_pool,
        },
        "articles": records(selected_articles),
        "mentions": records(selected_mentions),
        "predictions": records(selected_predictions),
        "relationships": records(selected_relationships),
        "prices": records(selected_prices),
        "tickerMetadata": records(ticker_metadata),
        "tickerAliases": records(ticker_aliases),
        "modelMetrics": records(metrics),
        "categoryMetrics": records(category_summary),
        "caseStudies": records(case_studies),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Articles: {payload['meta']['article_count']}; tickers: {payload['meta']['ticker_count']}")


if __name__ == "__main__":
    main()
