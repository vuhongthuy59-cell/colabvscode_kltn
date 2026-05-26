from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

TICKER_FILE = OUT_DIR / "ticker_list.csv"
FEATURE_FILE = OUT_DIR / "stock_features.csv"
LOG_RETURN_FILE = OUT_DIR / "master_log_return.csv"
NEWS_ARTICLES_FILE = OUT_DIR / "news_articles.csv"
NEWS_MENTIONS_FILE = OUT_DIR / "news_mentions.csv"
RELATIONSHIP_FILE = OUT_DIR / "company_relationships.csv"

LOOKBACK_RETURNS = 20
CORR_LOOKBACK = 252
LABEL_HORIZON = 5
CORR_THRESHOLD = 0.15

EDGE_TYPE_MAP = {
    "price_correlation": 0,
    "parent_to_subsidiary": 1,
    "subsidiary_to_parent": 2,
    "same_group": 3,
    "same_industry": 4,
    "news_co_mention": 5,
}


def load_tickers() -> list[str]:
    tickers = pd.read_csv(TICKER_FILE, dtype={"ticker": "string"})["ticker"].astype(str).str.upper().tolist()
    tickers = sorted(tickers)
    if len(tickers) != 118:
        raise ValueError(f"Expected 118 tickers, found {len(tickers)}")
    return tickers


def dense_matrix_from_long(features: pd.DataFrame, dates: list[str], tickers: list[str], value: str) -> np.ndarray:
    matrix = features.pivot(index="date", columns="ticker", values=value).reindex(index=dates, columns=tickers)
    return matrix.to_numpy(dtype=np.float32)


def build_relationship_edges(relationships: pd.DataFrame, ticker_to_idx: dict[str, int]) -> tuple[list[tuple[int, int]], list[float], list[int]]:
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    types: list[int] = []
    for row in relationships.itertuples(index=False):
        source = str(row.source_ticker).upper()
        target = str(row.target_ticker).upper()
        relation_type = str(row.relation_type)
        if source not in ticker_to_idx or target not in ticker_to_idx:
            continue
        if relation_type not in EDGE_TYPE_MAP:
            continue
        edges.append((ticker_to_idx[source], ticker_to_idx[target]))
        weights.append(float(row.weight))
        types.append(EDGE_TYPE_MAP[relation_type])
    return edges, weights, types


def build_corr_edges(window_returns: np.ndarray) -> tuple[list[tuple[int, int]], list[float], list[int]]:
    corr = pd.DataFrame(window_returns).corr(method="pearson", min_periods=60).to_numpy(dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    mask = (abs_corr >= CORR_THRESHOLD) & (~np.eye(abs_corr.shape[0], dtype=bool))
    sources, targets = np.where(mask)
    edges = list(zip(sources.astype(int).tolist(), targets.astype(int).tolist()))
    weights = abs_corr[sources, targets].astype(np.float32).tolist()
    types = [EDGE_TYPE_MAP["price_correlation"]] * len(edges)
    return edges, weights, types


def build_news_edges(mentions: pd.DataFrame, ticker_to_idx: dict[str, int]) -> tuple[list[tuple[int, int]], list[float], list[int]]:
    rows = []
    for row in mentions.itertuples(index=False):
        ticker = str(row.ticker).upper()
        if ticker in ticker_to_idx:
            rows.append((ticker_to_idx[ticker], float(row.relevance_score)))
    if len(rows) < 2:
        return [], [], []

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i, (source, rel_i) in enumerate(rows):
        for j, (target, rel_j) in enumerate(rows):
            if i == j or source == target:
                continue
            edges.append((source, target))
            weights.append(float(min(rel_i, rel_j)))
    types = [EDGE_TYPE_MAP["news_co_mention"]] * len(edges)
    return edges, weights, types


def make_tensor_edges(
    edge_parts: list[tuple[list[tuple[int, int]], list[float], list[int]]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    edge_index_rows: list[tuple[int, int]] = []
    edge_weight: list[float] = []
    edge_type: list[int] = []
    counts = {name: 0 for name in EDGE_TYPE_MAP}

    reverse_edge_type_map = {v: k for k, v in EDGE_TYPE_MAP.items()}
    for edges, weights, types in edge_parts:
        edge_index_rows.extend(edges)
        edge_weight.extend(weights)
        edge_type.extend(types)
        for type_id in types:
            counts[reverse_edge_type_map[type_id]] += 1

    if not edge_index_rows:
        edge_index = torch.empty((2, 0), dtype=torch.int16)
        weights = torch.empty((0,), dtype=torch.float16)
        types = torch.empty((0,), dtype=torch.uint8)
        return edge_index, weights, types, counts

    edge_index = torch.tensor(edge_index_rows, dtype=torch.int16).t().contiguous()
    weights = torch.tensor(edge_weight, dtype=torch.float16)
    types = torch.tensor(edge_type, dtype=torch.uint8)
    return edge_index, weights, types, counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers()
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    idx_to_ticker = {idx: ticker for ticker, idx in ticker_to_idx.items()}

    features = pd.read_csv(FEATURE_FILE)
    features["ticker"] = features["ticker"].astype(str).str.upper()
    features["date"] = features["date"].astype(str)

    log_return_wide = pd.read_csv(LOG_RETURN_FILE)
    dates = log_return_wide["date"].astype(str).tolist()
    date_to_pos = {date: idx for idx, date in enumerate(dates)}
    returns = log_return_wide[tickers].to_numpy(dtype=np.float32)

    rolling_vol_20 = dense_matrix_from_long(features, dates, tickers, "rolling_vol_20")
    volume_ratio_20 = dense_matrix_from_long(features, dates, tickers, "volume_ratio_20")

    articles = pd.read_csv(NEWS_ARTICLES_FILE)
    mentions = pd.read_csv(NEWS_MENTIONS_FILE)
    relationships = pd.read_csv(RELATIONSHIP_FILE)

    articles["event_trading_date"] = articles["event_trading_date"].astype(str)
    articles = articles[articles["is_firm_specific"].eq(1)].copy()
    categories = sorted(articles["category"].dropna().astype(str).unique().tolist())
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    mentions_by_article = {article_id: df for article_id, df in mentions.groupby("article_id", sort=False)}

    relationship_edges = build_relationship_edges(relationships, ticker_to_idx)
    relationship_neighbors: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(len(tickers))}
    for (source, target), weight, _ in zip(*relationship_edges):
        relationship_neighbors[source].append((target, float(weight)))

    snapshots = []
    index_rows = []
    skipped = []
    corr_cache: dict[str, tuple[list[tuple[int, int]], list[float], list[int]]] = {}

    for article in articles.itertuples(index=False):
        article_id = str(article.article_id)
        event_date = str(article.event_trading_date)
        if event_date not in date_to_pos:
            skipped.append((article_id, "event_date_not_in_trading_calendar"))
            continue

        pos = date_to_pos[event_date]
        feature_end = pos - 1
        label_start = pos + 1
        label_end = pos + LABEL_HORIZON
        if feature_end - LOOKBACK_RETURNS + 1 < 0:
            skipped.append((article_id, "insufficient_20d_feature_history"))
            continue
        if pos - CORR_LOOKBACK < 0:
            skipped.append((article_id, "insufficient_252d_correlation_history"))
            continue
        if label_end >= len(dates):
            skipped.append((article_id, "insufficient_5d_future_label"))
            continue

        hist_returns = returns[feature_end - LOOKBACK_RETURNS + 1 : feature_end + 1, :].T
        hist_returns = np.nan_to_num(hist_returns, nan=0.0, posinf=0.0, neginf=0.0)

        risk = rolling_vol_20[feature_end, :]
        volume_ratio = volume_ratio_20[feature_end, :]
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
        volume_ratio = np.nan_to_num(volume_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        article_mentions = mentions_by_article.get(article_id)
        if article_mentions is None or article_mentions.empty:
            skipped.append((article_id, "no_mentions"))
            continue

        direct_sentiment = np.zeros(len(tickers), dtype=np.float32)
        relevance = np.zeros(len(tickers), dtype=np.float32)
        primary = np.zeros(len(tickers), dtype=np.float32)
        mention_count = np.zeros(len(tickers), dtype=np.float32)
        target_mask = np.zeros(len(tickers), dtype=bool)

        direct_nodes = []
        for mention in article_mentions.itertuples(index=False):
            ticker = str(mention.ticker).upper()
            if ticker not in ticker_to_idx:
                continue
            idx = ticker_to_idx[ticker]
            direct_nodes.append(idx)
            direct_sentiment[idx] = float(mention.company_sentiment)
            relevance[idx] = float(mention.relevance_score)
            primary[idx] = float(mention.is_primary)
            mention_count[idx] = float(mention.mention_count)
            target_mask[idx] = True

        if not direct_nodes:
            skipped.append((article_id, "no_mentions_in_universe"))
            continue

        future_returns = returns[label_start : label_end + 1, :]
        future_valid_counts = np.isfinite(future_returns).sum(axis=0)
        if any(future_valid_counts[idx] < 2 for idx in direct_nodes):
            skipped.append((article_id, "insufficient_target_future_returns"))
            continue
        y = np.nanstd(future_returns, axis=0, ddof=1).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        category_one_hot = np.zeros((len(tickers), len(categories)), dtype=np.float32)
        category = str(article.category)
        if category in category_to_idx:
            category_one_hot[direct_nodes, category_to_idx[category]] = 1.0

        related_exposure = np.zeros(len(tickers), dtype=np.float32)
        for source_idx in direct_nodes:
            signal = direct_sentiment[source_idx] if direct_sentiment[source_idx] != 0 else float(article.general_sentiment)
            for target_idx, rel_weight in relationship_neighbors[source_idx]:
                if target_mask[target_idx]:
                    continue
                related_exposure[target_idx] = max(related_exposure[target_idx], abs(signal) * rel_weight)

        x = np.concatenate(
            [
                hist_returns.astype(np.float32),
                risk.reshape(-1, 1).astype(np.float32),
                volume_ratio.reshape(-1, 1).astype(np.float32),
                direct_sentiment.reshape(-1, 1),
                relevance.reshape(-1, 1),
                primary.reshape(-1, 1),
                mention_count.reshape(-1, 1),
                category_one_hot,
                related_exposure.reshape(-1, 1),
            ],
            axis=1,
        )

        if event_date not in corr_cache:
            corr_window = returns[pos - CORR_LOOKBACK : pos, :]
            corr_cache[event_date] = build_corr_edges(corr_window)

        news_edges = build_news_edges(article_mentions, ticker_to_idx)
        edge_index, edge_weight, edge_type, edge_counts = make_tensor_edges(
            [corr_cache[event_date], relationship_edges, news_edges]
        )

        snapshot = {
            "article_id": article_id,
            "event_trading_date": event_date,
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_type": edge_type,
            "y": torch.tensor(y, dtype=torch.float32),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
        }
        snapshots.append(snapshot)
        index_rows.append(
            {
                "snapshot_id": len(snapshots) - 1,
                "article_id": article_id,
                "published_date": article.published_date,
                "event_trading_date": event_date,
                "category": category,
                "general_sentiment": float(article.general_sentiment),
                "n_mapped_tickers": int(article.n_mapped_tickers),
                "target_count": int(target_mask.sum()),
                "num_nodes": len(tickers),
                "num_node_features": int(x.shape[1]),
                "num_edges": int(edge_index.shape[1]),
                **{f"num_edges_{name}": count for name, count in edge_counts.items()},
                "label_horizon": LABEL_HORIZON,
                "lookback_returns": LOOKBACK_RETURNS,
                "corr_lookback": CORR_LOOKBACK,
            }
        )

    torch.save(snapshots, OUT_DIR / "graph_snapshots.pt")

    snapshot_index = pd.DataFrame(index_rows)
    snapshot_index.to_csv(OUT_DIR / "snapshot_index.csv", index=False)

    with open(OUT_DIR / "ticker_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(ticker_to_idx, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "idx_to_ticker.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_ticker, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "edge_type_map.json", "w", encoding="utf-8") as f:
        json.dump(EDGE_TYPE_MAP, f, ensure_ascii=False, indent=2)

    skip_counts = pd.Series([reason for _, reason in skipped]).value_counts().to_dict() if skipped else {}
    quality_rows = [
        ("input_articles", len(articles), "Firm-specific articles from news_articles.csv"),
        ("saved_snapshots", len(snapshots), "Snapshots saved to graph_snapshots.pt"),
        ("skipped_articles", len(skipped), "Articles skipped by temporal/data availability rules"),
        ("num_nodes", len(tickers), "Fixed stock universe size"),
        ("num_node_features", int(snapshot_index["num_node_features"].iloc[0]) if len(snapshot_index) else 0, "20 returns + risk + volume + news + category + exposure"),
        ("category_count", len(categories), "News category one-hot width"),
        ("unique_event_dates", snapshot_index["event_trading_date"].nunique() if len(snapshot_index) else 0, "Correlation cache keys"),
        ("edge_type_count", len(EDGE_TYPE_MAP), "price, relationships, news co-mention"),
        ("mean_edges_per_snapshot", float(snapshot_index["num_edges"].mean()) if len(snapshot_index) else 0, "Average total edges"),
        ("min_edges_per_snapshot", int(snapshot_index["num_edges"].min()) if len(snapshot_index) else 0, "Minimum total edges"),
        ("max_edges_per_snapshot", int(snapshot_index["num_edges"].max()) if len(snapshot_index) else 0, "Maximum total edges"),
        ("mean_target_count", float(snapshot_index["target_count"].mean()) if len(snapshot_index) else 0, "Average directly mentioned tickers"),
    ]
    for reason, count in skip_counts.items():
        quality_rows.append((f"skipped_{reason}", int(count), "Skip reason count"))

    quality = pd.DataFrame(quality_rows, columns=["metric", "value", "note"])
    quality.to_csv(OUT_DIR / "graph_snapshot_quality_report.csv", index=False)

    print("Generated graph snapshot outputs in data/ and data/processed/")
    print(f"Snapshots: {len(snapshots)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Node features: {quality.loc[quality.metric.eq('num_node_features'), 'value'].iloc[0]}")
    print(f"Mean edges: {quality.loc[quality.metric.eq('mean_edges_per_snapshot'), 'value'].iloc[0]}")


if __name__ == "__main__":
    main()
