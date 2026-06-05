from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "05_event_graph_dataset"

TICKER_FILE = ROOT / "outputs" / "01_price_data" / "ticker_list.csv"
FEATURE_FILE = ROOT / "outputs" / "01_price_data" / "stock_features.csv"
LOG_RETURN_FILE = ROOT / "outputs" / "01_price_data" / "master_log_return.csv"
NEWS_ARTICLES_FILE = ROOT / "outputs" / "02_news_data" / "news_articles.csv"
NEWS_MENTIONS_FILE = ROOT / "outputs" / "02_news_data" / "news_mentions.csv"
RELATIONSHIP_FILE = ROOT / "outputs" / "04_company_relationships" / "company_relationships.csv"
TICKER_METADATA_FILE = ROOT / "outputs" / "01_price_data" / "ticker_metadata.csv"
STOCK_MICRO_FILE = DATA_DIR / "processed" / "stock_micro_features.csv"
MARKET_MACRO_FILE = DATA_DIR / "processed" / "market_macro_features.csv"
FEATURE_SCHEMA_FILE = OUT_DIR / "node_feature_schema.json"

LOOKBACK_RETURNS = 20
CORR_LOOKBACK = 252
LABEL_HORIZON = 5
CORR_THRESHOLD = 0.15
CORR_POSITIVE_TOP_K = 10
CORR_NEGATIVE_TOP_K = 5
SECTOR_TOP_K = 5
LOG_TARGET_SCALE = 100.0

RETURN_CLIP = (-0.30, 0.30)
ROLLING_VOL_CLIP = (0.0, 0.20)
VOLUME_RATIO_CLIP = (0.0, 10.0)
MICRO_RATIO_CLIP = (0.0, 10.0)
SHORT_RETURN_CLIP = (-0.30, 0.30)
SHORT_VOL_CLIP = (0.0, 0.20)

EDGE_TYPE_MAP = {
    "corr_positive_top10": 0,
    "corr_negative_top5": 1,
    "ownership": 2,
    "value_chain_curated": 3,
    "sector_top5_only": 4,
    "news_co_mention": 5,
}

OWNERSHIP_RELATION_TYPES = {
    "parent_to_subsidiary",
    "subsidiary_to_parent",
    "same_group",
    "strategic_ecosystem",
    "common_owner",
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


def clean_feature_array(
    values: np.ndarray,
    name: str,
    quality_counts: dict[str, int],
    clip_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    nonfinite = ~np.isfinite(array)
    quality_counts[f"{name}_nonfinite_replaced"] = quality_counts.get(f"{name}_nonfinite_replaced", 0) + int(nonfinite.sum())
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if clip_bounds is None:
        return array

    lower, upper = clip_bounds
    clipped = (array < lower) | (array > upper)
    quality_counts[f"{name}_clipped"] = quality_counts.get(f"{name}_clipped", 0) + int(clipped.sum())
    return np.clip(array, lower, upper).astype(np.float32)


def trailing_window_stats(hist_returns: np.ndarray, quality_counts: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    windows = {
        "5": hist_returns[:, -5:],
        "10": hist_returns[:, -10:],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = [
            clean_feature_array(np.nanstd(windows["5"], axis=1, ddof=1), "realized_vol_lag_5", quality_counts, SHORT_VOL_CLIP),
            clean_feature_array(np.nanstd(windows["10"], axis=1, ddof=1), "realized_vol_lag_10", quality_counts, SHORT_VOL_CLIP),
            clean_feature_array(np.nanmean(windows["5"], axis=1), "return_mean_5", quality_counts, SHORT_RETURN_CLIP),
            clean_feature_array(np.nanmean(windows["10"], axis=1), "return_mean_10", quality_counts, SHORT_RETURN_CLIP),
            clean_feature_array(np.nanmean(np.abs(windows["5"]), axis=1), "abs_return_mean_5", quality_counts, SHORT_VOL_CLIP),
            clean_feature_array(np.nanmean(np.abs(windows["10"]), axis=1), "abs_return_mean_10", quality_counts, SHORT_VOL_CLIP),
            clean_feature_array(np.nanmax(np.abs(windows["5"]), axis=1), "max_abs_return_5", quality_counts, SHORT_RETURN_CLIP),
        ]
    names = [
        "realized_vol_lag_5",
        "realized_vol_lag_10",
        "return_mean_5",
        "return_mean_10",
        "abs_return_mean_5",
        "abs_return_mean_10",
        "max_abs_return_5",
    ]
    return np.stack(stats, axis=1).astype(np.float32), names

def load_optional_micro_features(dates: list[str], tickers: list[str]) -> tuple[np.ndarray, list[str]]:
    names = ["trading_value_ratio_20", "log_market_cap"]
    if not STOCK_MICRO_FILE.exists():
        return np.zeros((len(dates), len(tickers), len(names)), dtype=np.float32), names
    micro = pd.read_csv(STOCK_MICRO_FILE)
    micro["ticker"] = micro["ticker"].astype(str).str.upper()
    micro["date"] = micro["date"].astype(str)
    matrices = [dense_matrix_from_long(micro, dates, tickers, name) for name in names]
    stacked = np.stack(matrices, axis=2)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return stacked, names

def load_industry_features(tickers: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    metadata = pd.read_csv(TICKER_METADATA_FILE)
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    industries = sorted(metadata["industry"].dropna().astype(str).unique().tolist())
    industry_to_idx = {industry: idx for idx, industry in enumerate(industries)}
    matrix = np.zeros((len(tickers), len(industries)), dtype=np.float32)
    industry_by_ticker = metadata.set_index("ticker")["industry"].astype(str).to_dict()
    industry_labels: list[str] = []
    for idx, ticker in enumerate(tickers):
        industry = industry_by_ticker.get(ticker, "")
        industry_labels.append(industry)
        if industry in industry_to_idx:
            matrix[idx, industry_to_idx[industry]] = 1.0
    names = [f"industry_{industry}" for industry in industries]
    return matrix, names, industry_labels

def load_optional_macro_features(dates: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [
        "universe_market_return",
        "universe_market_roll_vol_20",
        "market_liquidity_ratio_20",
        "vni_return_1d",
        "vni_roll_vol_20",
        "vni_available",
        "usd_vnd_return_1d",
        "oil_return_1d",
    ]
    if not MARKET_MACRO_FILE.exists():
        return np.zeros((len(dates), len(names)), dtype=np.float32), names
    macro = pd.read_csv(MARKET_MACRO_FILE)
    macro["date"] = macro["date"].astype(str)
    macro = macro.set_index("date").reindex(dates)
    values = macro[names].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return values, names


def is_relationship_active(row: object, event_date: str | None) -> bool:
    if event_date is None:
        return True
    event_ts = pd.Timestamp(event_date)
    valid_from = str(getattr(row, "valid_from", "") or "").strip()
    valid_to = str(getattr(row, "valid_to", "") or "").strip()
    if valid_from and valid_from.lower() != "nan" and event_ts < pd.Timestamp(valid_from):
        return False
    if valid_to and valid_to.lower() != "nan" and event_ts > pd.Timestamp(valid_to):
        return False
    return True


def build_relationship_edges(
    relationships: pd.DataFrame,
    ticker_to_idx: dict[str, int],
    event_date: str | None = None,
) -> tuple[list[tuple[int, int]], list[float], list[int]]:
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    types: list[int] = []
    for row in relationships.itertuples(index=False):
        source = str(row.source_ticker).upper()
        target = str(row.target_ticker).upper()
        raw_relation_type = str(row.relation_type)
        if raw_relation_type in OWNERSHIP_RELATION_TYPES:
            relation_type = "ownership"
        elif raw_relation_type == "value_chain":
            relation_type = "value_chain_curated"
        else:
            continue
        if not is_relationship_active(row, event_date):
            continue
        if source not in ticker_to_idx or target not in ticker_to_idx:
            continue
        edges.append((ticker_to_idx[source], ticker_to_idx[target]))
        weights.append(float(row.weight))
        types.append(EDGE_TYPE_MAP[relation_type])
    return edges, weights, types


def build_relationship_neighbors(
    relationship_edges: tuple[list[tuple[int, int]], list[float], list[int]],
    num_nodes: int,
) -> dict[int, list[tuple[int, float]]]:
    relationship_neighbors: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(num_nodes)}
    for (source, target), weight, _ in zip(*relationship_edges):
        relationship_neighbors[source].append((target, float(weight)))
    return relationship_neighbors


def build_corr_edges(window_returns: np.ndarray, industry_labels: list[str]) -> tuple[list[tuple[int, int]], list[float], list[int]]:
    corr = pd.DataFrame(window_returns).corr(method="pearson", min_periods=60).to_numpy(dtype=np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    types: list[int] = []
    for source in range(abs_corr.shape[0]):
        positive_ranked = np.argsort(corr[source])[::-1]
        selected = [int(target) for target in positive_ranked if corr[source, target] >= CORR_THRESHOLD and source != target][:CORR_POSITIVE_TOP_K]
        for target in selected:
            edges.append((source, target))
            weights.append(float(abs_corr[source, target]))
            types.append(EDGE_TYPE_MAP["corr_positive_top10"])

        negative_ranked = np.argsort(corr[source])
        selected = [int(target) for target in negative_ranked if corr[source, target] <= -CORR_THRESHOLD and source != target][:CORR_NEGATIVE_TOP_K]
        for target in selected:
            edges.append((source, target))
            weights.append(float(abs_corr[source, target]))
            types.append(EDGE_TYPE_MAP["corr_negative_top5"])

        same_sector_targets = [
            int(target)
            for target in np.argsort(abs_corr[source])[::-1]
            if source != target
            and industry_labels[source]
            and industry_labels[source] == industry_labels[target]
            and abs_corr[source, target] >= CORR_THRESHOLD
        ][:SECTOR_TOP_K]
        for target in same_sector_targets:
            edges.append((source, target))
            weights.append(float(abs_corr[source, target]))
            types.append(EDGE_TYPE_MAP["sector_top5_only"])
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


def build_ticker_date_events(articles: pd.DataFrame, mentions: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    article_cols = [
        "article_id",
        "published_date",
        "event_trading_date",
        "category",
        "general_sentiment",
        "sentiment_label",
        "title",
    ]
    events = mentions.merge(articles[article_cols], on="article_id", how="inner")
    events["ticker"] = events["ticker"].astype(str).str.upper()
    events = events[events["ticker"].isin(set(tickers))].copy()
    if events.empty:
        return pd.DataFrame()

    rows = []
    for (event_date, ticker), group in events.groupby(["event_trading_date", "ticker"], sort=True):
        article_ids = sorted(group["article_id"].astype(str).unique().tolist())
        category_counts = group["category"].astype(str).value_counts()
        dominant_category = str(category_counts.index[0]) if len(category_counts) else "other"
        title_sample = " | ".join(group["title"].dropna().astype(str).drop_duplicates().head(3).tolist())
        rows.append(
            {
                "event_id": f"{event_date}_{ticker}",
                "article_id": article_ids[0],
                "article_ids": ";".join(article_ids),
                "published_date": str(group["published_date"].astype(str).min()),
                "event_trading_date": str(event_date),
                "ticker": ticker,
                "category": dominant_category,
                "title": title_sample,
                "general_sentiment": float(group["general_sentiment"].mean()),
                "sentiment_label": str(group["sentiment_label"].mode().iloc[0]) if not group["sentiment_label"].mode().empty else "",
                "news_count": int(len(article_ids)),
                "mention_rows": int(len(group)),
                "primary_news_count": int(group["is_primary"].fillna(0).astype(float).sum()),
                "max_relevance_score": float(group["relevance_score"].astype(float).max()),
                "mean_relevance_score": float(group["relevance_score"].astype(float).mean()),
                "sentiment_mean": float(group["company_sentiment"].astype(float).mean()),
                "sentiment_max_abs": float(group["company_sentiment"].astype(float).abs().max()),
                "mention_count_sum": float(group["mention_count"].astype(float).sum()),
            }
        )
    return pd.DataFrame(rows)


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


def aggregate_layer_exposure(
    edge_parts: list[tuple[list[tuple[int, int]], list[float], list[int]]],
    source_signal: np.ndarray,
    edge_type_id: int,
    num_nodes: int,
) -> np.ndarray:
    exposure = np.zeros(num_nodes, dtype=np.float32)
    denom = np.zeros(num_nodes, dtype=np.float32)
    for edges, weights, types in edge_parts:
        for (source, target), weight, type_id in zip(edges, weights, types):
            if int(type_id) != int(edge_type_id):
                continue
            exposure[target] += float(weight) * float(source_signal[source])
            denom[target] += float(weight)
    active = denom > 0
    exposure[active] = exposure[active] / denom[active]
    return exposure.astype(np.float32)


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
    micro_features, micro_feature_names = load_optional_micro_features(dates, tickers)
    industry_features, industry_feature_names, industry_labels = load_industry_features(tickers)
    macro_features, macro_feature_names = load_optional_macro_features(dates)

    articles = pd.read_csv(NEWS_ARTICLES_FILE)
    mentions = pd.read_csv(NEWS_MENTIONS_FILE)
    relationships = pd.read_csv(RELATIONSHIP_FILE)

    articles["event_trading_date"] = articles["event_trading_date"].astype(str)
    articles = articles[articles["is_firm_specific"].eq(1)].copy()
    categories = sorted(articles["category"].dropna().astype(str).unique().tolist())
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    event_rows = build_ticker_date_events(articles, mentions, tickers)

    snapshots = []
    index_rows = []
    skipped = []
    feature_quality_counts: dict[str, int] = {}
    corr_cache: dict[str, tuple[list[tuple[int, int]], list[float], list[int]]] = {}
    relationship_edge_cache: dict[str, tuple[list[tuple[int, int]], list[float], list[int]]] = {}
    relationship_neighbor_cache: dict[str, dict[int, list[tuple[int, float]]]] = {}

    for event in event_rows.itertuples(index=False):
        event_id = str(event.event_id)
        article_id = str(event.article_id)
        event_date = str(event.event_trading_date)
        ticker = str(event.ticker).upper()
        if event_date not in date_to_pos:
            skipped.append((event_id, "event_date_not_in_trading_calendar"))
            continue
        if ticker not in ticker_to_idx:
            skipped.append((event_id, "ticker_not_in_universe"))
            continue

        pos = date_to_pos[event_date]
        feature_end = pos - 1
        label_start = pos + 1
        label_end = pos + LABEL_HORIZON
        if feature_end - LOOKBACK_RETURNS + 1 < 0:
            skipped.append((event_id, "insufficient_20d_feature_history"))
            continue
        if pos - CORR_LOOKBACK < 0:
            skipped.append((event_id, "insufficient_252d_correlation_history"))
            continue
        if label_end >= len(dates):
            skipped.append((event_id, "insufficient_5d_future_label"))
            continue

        raw_hist_returns = returns[feature_end - LOOKBACK_RETURNS + 1 : feature_end + 1, :].T
        hist_returns = clean_feature_array(raw_hist_returns, "hist_returns", feature_quality_counts, RETURN_CLIP)
        short_price_stats, short_price_feature_names = trailing_window_stats(raw_hist_returns, feature_quality_counts)

        risk = rolling_vol_20[feature_end, :]
        volume_ratio = volume_ratio_20[feature_end, :]
        micro_at_event = micro_features[feature_end, :, :]
        macro_at_event = np.repeat(macro_features[feature_end, :].reshape(1, -1), len(tickers), axis=0)
        risk = clean_feature_array(risk, "rolling_vol_20", feature_quality_counts, ROLLING_VOL_CLIP)
        volume_ratio = clean_feature_array(volume_ratio, "volume_ratio_20", feature_quality_counts, VOLUME_RATIO_CLIP)
        micro_at_event = clean_feature_array(micro_at_event, "micro_features", feature_quality_counts)
        if micro_at_event.shape[1] > 0:
            micro_at_event[:, 0] = clean_feature_array(
                micro_at_event[:, 0],
                "trading_value_ratio_20",
                feature_quality_counts,
                MICRO_RATIO_CLIP,
            )
        macro_at_event = clean_feature_array(macro_at_event, "macro_features", feature_quality_counts)

        article_ids = str(event.article_ids).split(";")
        event_mentions = mentions[mentions["article_id"].astype(str).isin(article_ids)].copy()
        if event_mentions.empty:
            skipped.append((event_id, "no_mentions"))
            continue

        direct_sentiment = np.zeros(len(tickers), dtype=np.float32)
        relevance = np.zeros(len(tickers), dtype=np.float32)
        primary = np.zeros(len(tickers), dtype=np.float32)
        mention_count = np.zeros(len(tickers), dtype=np.float32)
        target_mask = np.zeros(len(tickers), dtype=bool)

        event_target_idx = ticker_to_idx[ticker]
        direct_nodes = [event_target_idx]
        direct_sentiment[event_target_idx] = float(event.sentiment_mean)
        relevance[event_target_idx] = float(event.max_relevance_score)
        primary[event_target_idx] = float(event.primary_news_count)
        mention_count[event_target_idx] = float(event.mention_count_sum)
        target_mask[event_target_idx] = True

        if not direct_nodes:
            skipped.append((event_id, "no_mentions_in_universe"))
            continue

        if event_date not in relationship_edge_cache:
            relationship_edge_cache[event_date] = build_relationship_edges(relationships, ticker_to_idx, event_date)
            relationship_neighbor_cache[event_date] = build_relationship_neighbors(
                relationship_edge_cache[event_date],
                len(tickers),
            )
        relationship_edges = relationship_edge_cache[event_date]
        relationship_neighbors = relationship_neighbor_cache[event_date]

        future_returns = returns[label_start : label_end + 1, :]
        future_valid_counts = np.isfinite(future_returns).sum(axis=0)
        if any(future_valid_counts[idx] < 2 for idx in direct_nodes):
            skipped.append((event_id, "insufficient_target_future_returns"))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            future_volatility = np.nanstd(future_returns, axis=0, ddof=1).astype(np.float32)
        future_volatility = np.nan_to_num(future_volatility, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_volatility = risk.astype(np.float32)
        abnormal_volatility = (future_volatility - baseline_volatility).astype(np.float32)
        y = np.sign(abnormal_volatility) * np.log1p(LOG_TARGET_SCALE * np.abs(abnormal_volatility))
        y = y.astype(np.float32)

        category_one_hot = np.zeros((len(tickers), len(categories)), dtype=np.float32)
        category = str(event.category)
        if category in category_to_idx:
            category_one_hot[direct_nodes, category_to_idx[category]] = 1.0

        negative_news_count = np.zeros(len(tickers), dtype=np.float32)
        return_shock = clean_feature_array(np.abs(raw_hist_returns[:, -1]) / (risk + 1e-6), "return_shock", feature_quality_counts, (0.0, 10.0))
        volatility_shock = clean_feature_array(short_price_stats[:, 0] / (risk + 1e-6), "volatility_shock", feature_quality_counts, (0.0, 10.0))
        volume_shock = volume_ratio.astype(np.float32)
        sector_shock = np.zeros(len(tickers), dtype=np.float32)

        negative_news_count[event_target_idx] = float(max(0.0, -float(event.sentiment_mean)) * max(1, int(event.news_count)))
        for industry in set(industry_labels):
            if not industry:
                continue
            sector_indices = [idx for idx, label in enumerate(industry_labels) if label == industry]
            sector_values = return_shock[sector_indices]
            sector_mean = float(np.mean(sector_values)) if len(sector_values) else 0.0
            for idx in sector_indices:
                sector_shock[idx] = sector_mean

        if event_date not in corr_cache:
            corr_window = returns[pos - CORR_LOOKBACK : pos, :]
            corr_cache[event_date] = build_corr_edges(corr_window, industry_labels)

        news_edges = build_news_edges(event_mentions, ticker_to_idx)
        edge_parts = [corr_cache[event_date], relationship_edges, news_edges]
        shock_signal = clean_feature_array(
            0.5 * return_shock + 0.5 * volatility_shock,
            "neighbor_shock_signal",
            feature_quality_counts,
            (0.0, 10.0),
        )
        news_signal = clean_feature_array(
            negative_news_count + np.abs(direct_sentiment),
            "neighbor_news_signal",
            feature_quality_counts,
            (0.0, 10.0),
        )
        pos_corr_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            shock_signal,
            EDGE_TYPE_MAP["corr_positive_top10"],
            len(tickers),
        )
        neg_corr_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            shock_signal,
            EDGE_TYPE_MAP["corr_negative_top5"],
            len(tickers),
        )
        ownership_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            shock_signal,
            EDGE_TYPE_MAP["ownership"],
            len(tickers),
        )
        value_chain_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            shock_signal,
            EDGE_TYPE_MAP["value_chain_curated"],
            len(tickers),
        )
        sector_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            shock_signal,
            EDGE_TYPE_MAP["sector_top5_only"],
            len(tickers),
        )
        news_neighbor_exposure = aggregate_layer_exposure(
            edge_parts,
            news_signal,
            EDGE_TYPE_MAP["news_co_mention"],
            len(tickers),
        )

        x = np.concatenate(
            [
                hist_returns.astype(np.float32),
                risk.reshape(-1, 1).astype(np.float32),
                volume_ratio.reshape(-1, 1).astype(np.float32),
                short_price_stats.astype(np.float32),
                micro_at_event.astype(np.float32),
                industry_features.astype(np.float32),
                macro_at_event.astype(np.float32),
                direct_sentiment.reshape(-1, 1),
                relevance.reshape(-1, 1),
                primary.reshape(-1, 1),
                mention_count.reshape(-1, 1),
                category_one_hot,
                return_shock.reshape(-1, 1),
                volatility_shock.reshape(-1, 1),
                volume_shock.reshape(-1, 1),
                negative_news_count.reshape(-1, 1),
                sector_shock.reshape(-1, 1),
                pos_corr_neighbor_exposure.reshape(-1, 1),
                neg_corr_neighbor_exposure.reshape(-1, 1),
                ownership_neighbor_exposure.reshape(-1, 1),
                value_chain_neighbor_exposure.reshape(-1, 1),
                sector_neighbor_exposure.reshape(-1, 1),
                news_neighbor_exposure.reshape(-1, 1),
            ],
            axis=1,
        )
        edge_index, edge_weight, edge_type, edge_counts = make_tensor_edges(
            edge_parts
        )

        snapshot = {
            "article_id": article_id,
            "event_id": event_id,
            "ticker": ticker,
            "article_ids": str(event.article_ids),
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
                "event_id": event_id,
                "article_id": article_id,
                "article_ids": str(event.article_ids),
                "published_date": event.published_date,
                "event_trading_date": event_date,
                "ticker": ticker,
                "category": category,
                "sentiment_label": str(event.sentiment_label),
                "title": str(event.title),
                "general_sentiment": float(event.general_sentiment),
                "n_mapped_tickers": 1,
                "news_count": int(event.news_count),
                "mention_rows": int(event.mention_rows),
                "primary_news_count": int(event.primary_news_count),
                "max_relevance_score": float(event.max_relevance_score),
                "mean_relevance_score": float(event.mean_relevance_score),
                "sentiment_mean": float(event.sentiment_mean),
                "sentiment_max_abs": float(event.sentiment_max_abs),
                "target_count": int(target_mask.sum()),
                "target_raw_future_volatility_5d": float(future_volatility[event_target_idx]),
                "target_baseline_rolling_vol_20": float(baseline_volatility[event_target_idx]),
                "target_abnormal_volatility_5d": float(abnormal_volatility[event_target_idx]),
                "target_log_abnormal_volatility_5d": float(y[event_target_idx]),
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
    feature_quality = pd.DataFrame(
        [
            {"metric": key, "value": value, "note": "Feature cleaning count accumulated while building graph snapshots"}
            for key, value in sorted(feature_quality_counts.items())
        ]
        + [
            {"metric": "return_clip_lower", "value": RETURN_CLIP[0], "note": "Applied to historical return lag features only"},
            {"metric": "return_clip_upper", "value": RETURN_CLIP[1], "note": "Applied to historical return lag features only"},
            {"metric": "rolling_vol_clip_upper", "value": ROLLING_VOL_CLIP[1], "note": "Applied to rolling_vol_20 feature only"},
            {"metric": "volume_ratio_clip_upper", "value": VOLUME_RATIO_CLIP[1], "note": "Applied to volume_ratio_20 feature only"},
            {"metric": "target_y_clipped", "value": 0, "note": "Target realized volatility is not clipped"},
            {
                "metric": "target_definition",
                "value": "future_realized_volatility_5d_minus_rolling_vol_20_t_minus_1",
                "note": "Main regression target is abnormal volatility, not raw future volatility",
            },
        ]
    )
    feature_quality.to_csv(OUT_DIR / "graph_feature_quality_report.csv", index=False)

    with open(OUT_DIR / "ticker_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(ticker_to_idx, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "idx_to_ticker.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_ticker, f, ensure_ascii=False, indent=2)
    with open(OUT_DIR / "edge_type_map.json", "w", encoding="utf-8") as f:
        json.dump(EDGE_TYPE_MAP, f, ensure_ascii=False, indent=2)
    schema = {
        "price_feature_names": [f"log_return_lag_{lag}" for lag in range(20, 0, -1)]
        + ["rolling_vol_20_t_minus_1", "volume_ratio_20_t_minus_1"]
        + short_price_feature_names,
        "micro_feature_names": micro_feature_names,
        "industry_feature_names": industry_feature_names,
        "macro_feature_names": macro_feature_names,
        "news_feature_names": [
            "direct_news_sentiment",
            "news_relevance_score",
            "is_primary",
            "mention_count",
        ]
        + [f"category_{category}" for category in categories],
        "shock_feature_names": [
            "return_shock",
            "volatility_shock",
            "volume_shock",
            "negative_news_count",
            "sector_shock",
        ],
        "exposure_feature_names": [
            "pos_corr_neighbor_exposure",
            "neg_corr_neighbor_exposure",
            "ownership_neighbor_exposure",
            "value_chain_neighbor_exposure",
            "sector_neighbor_exposure",
            "news_neighbor_exposure",
        ],
        "target_name": "log_abnormal_volatility_5d",
        "target_formula": "sign(abnormal_volatility_5d) * log1p(100 * abs(abnormal_volatility_5d))",
        "corr_positive_top_k": CORR_POSITIVE_TOP_K,
        "corr_negative_top_k": CORR_NEGATIVE_TOP_K,
        "sector_top_k": SECTOR_TOP_K,
        "corr_graph_min_abs_corr": CORR_THRESHOLD,
    }
    schema["feature_names"] = (
        schema["price_feature_names"]
        + schema["micro_feature_names"]
        + schema["industry_feature_names"]
        + schema["macro_feature_names"]
        + schema["news_feature_names"]
        + schema["shock_feature_names"]
        + schema["exposure_feature_names"]
    )
    schema["non_news_feature_count"] = (
        len(schema["price_feature_names"])
        + len(schema["micro_feature_names"])
        + len(schema["industry_feature_names"])
        + len(schema["macro_feature_names"])
    )
    with open(FEATURE_SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    skip_counts = pd.Series([reason for _, reason in skipped]).value_counts().to_dict() if skipped else {}
    quality_rows = [
        ("input_ticker_date_events", len(event_rows), "Aggregated ticker-date events built from firm-specific news mentions"),
        ("input_articles", len(articles), "Firm-specific articles from news_articles.csv"),
        ("saved_snapshots", len(snapshots), "Ticker-date event snapshots saved to graph_snapshots.pt"),
        ("skipped_events", len(skipped), "Ticker-date events skipped by temporal/data availability rules"),
        ("num_nodes", len(tickers), "Fixed stock universe size"),
        ("num_node_features", int(snapshot_index["num_node_features"].iloc[0]) if len(snapshot_index) else 0, "price + micro + industry + macro + news + category + exposure"),
        ("category_count", len(categories), "News category one-hot width"),
        ("unique_event_dates", snapshot_index["event_trading_date"].nunique() if len(snapshot_index) else 0, "Correlation cache keys"),
        ("edge_type_count", len(EDGE_TYPE_MAP), "positive/negative correlation, ownership, value-chain, sector, news co-mention"),
        ("corr_positive_top_k", CORR_POSITIVE_TOP_K, "Maximum positive-correlation neighbors kept per source node"),
        ("corr_negative_top_k", CORR_NEGATIVE_TOP_K, "Maximum negative-correlation neighbors kept per source node"),
        ("sector_top_k", SECTOR_TOP_K, "Maximum same-sector correlation neighbors kept per source node"),
        ("target_name", "log_abnormal_volatility_5d", "Regression target used in graph snapshots"),
        ("mean_edges_per_snapshot", float(snapshot_index["num_edges"].mean()) if len(snapshot_index) else 0, "Average total edges"),
        ("min_edges_per_snapshot", int(snapshot_index["num_edges"].min()) if len(snapshot_index) else 0, "Minimum total edges"),
        ("max_edges_per_snapshot", int(snapshot_index["num_edges"].max()) if len(snapshot_index) else 0, "Maximum total edges"),
        ("mean_target_count", float(snapshot_index["target_count"].mean()) if len(snapshot_index) else 0, "Average target tickers per ticker-date event"),
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
