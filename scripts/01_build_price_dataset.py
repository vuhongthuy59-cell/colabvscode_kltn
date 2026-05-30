from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_config import (
    DEMO_END_DATE,
    INCLUDE_2026_APPEND_ENV,
    MAIN_END_DATE,
    MAIN_START_DATE,
    active_dataset_scope,
    include_2026_append,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "01_build_price_dataset"

UNIVERSE_FILE = DATA_DIR / "data_origial" / "universe.csv"
PRICE_FILE = DATA_DIR / "data_origial" / "Stock_Price_2022-2025.csv"
PRICE_APPEND_FILE = DATA_DIR / "data_origial" / "Stock_Price_2026_append.csv"
RELATIONSHIP_FILE = DATA_DIR / "data_origial" / "relationships.xlsx"


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "date",
    "ticker",
    "close",
    "volume",
    "log_return",
    "rolling_vol_20",
    "rolling_vol_60",
    "volume_change_1d",
    "volume_ma_20",
    "volume_ratio_20",
    "abnormal_volume",
]


def normalize_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_FILE, dtype={"ticker": "string"})
    if "ticker" not in universe.columns:
        raise ValueError(f"{UNIVERSE_FILE} must contain a ticker column")

    universe = universe[["ticker"]].copy()
    universe["ticker"] = normalize_ticker(universe["ticker"])
    universe = universe.drop_duplicates().sort_values("ticker").reset_index(drop=True)

    if universe["ticker"].isna().any() or (universe["ticker"] == "").any():
        raise ValueError("universe.csv contains blank tickers")

    return universe


def load_prices(universe: pd.DataFrame) -> pd.DataFrame:
    price_frames = [pd.read_csv(PRICE_FILE)]
    if include_2026_append() and PRICE_APPEND_FILE.exists():
        price_frames.append(pd.read_csv(PRICE_APPEND_FILE))
    prices = pd.concat(price_frames, ignore_index=True)
    prices = prices.loc[:, ~prices.columns.str.startswith("Unnamed:")].copy()

    missing_columns = set(PRICE_COLUMNS) - set(prices.columns)
    if missing_columns:
        raise ValueError(f"{PRICE_FILE} is missing columns: {sorted(missing_columns)}")

    prices = prices[PRICE_COLUMNS].copy()
    prices["ticker"] = normalize_ticker(prices["ticker"])
    prices["date"] = pd.to_datetime(prices["date"], errors="raise").dt.strftime("%Y-%m-%d")
    end_date = DEMO_END_DATE if include_2026_append() else MAIN_END_DATE
    prices = prices[(prices["date"] >= MAIN_START_DATE) & (prices["date"] <= end_date)].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="raise")

    prices["volume"] = prices["volume"].round().astype("int64")
    prices = prices.drop_duplicates(["date", "ticker"], keep="last")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    universe_tickers = set(universe["ticker"])
    price_tickers = set(prices["ticker"])
    if universe_tickers != price_tickers:
        raise ValueError(
            "Ticker mismatch between universe.csv and prices. "
            f"Missing in prices: {sorted(universe_tickers - price_tickers)}; "
            f"Extra in prices: {sorted(price_tickers - universe_tickers)}"
        )

    duplicate_count = prices.duplicated(["date", "ticker"]).sum()
    if duplicate_count:
        raise ValueError(f"stock price data contains {duplicate_count} duplicate date/ticker rows")

    return prices


def build_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    same_sector = pd.read_excel(RELATIONSHIP_FILE, sheet_name=0)
    required = {"source", "target", "sector_group"}
    missing_columns = required - set(same_sector.columns)
    if missing_columns:
        raise ValueError(f"{RELATIONSHIP_FILE} sheet 0 is missing columns: {sorted(missing_columns)}")

    sector_map: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    for _, row in same_sector.iterrows():
        sector = str(row["sector_group"]).strip()
        for col in ["source", "target"]:
            ticker = str(row[col]).strip().upper()
            if not ticker or ticker == "NAN":
                continue
            if ticker in sector_map and sector_map[ticker] != sector:
                conflicts.setdefault(ticker, {sector_map[ticker]}).add(sector)
            sector_map[ticker] = sector

    if conflicts:
        raise ValueError(f"Conflicting sector groups found: {conflicts}")

    metadata = universe.copy()
    metadata["company_name"] = ""
    metadata["industry"] = metadata["ticker"].map(sector_map).fillna("")
    metadata["sector"] = metadata["industry"]
    metadata["exchange"] = ""
    metadata = metadata[["ticker", "company_name", "industry", "sector", "exchange"]]

    missing_industry = metadata.loc[metadata["industry"].eq(""), "ticker"].tolist()
    if missing_industry:
        raise ValueError(f"Missing industry for tickers: {missing_industry}")

    return metadata


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    features = prices[["date", "ticker", "close", "volume"]].copy()
    grouped = features.groupby("ticker", group_keys=False)

    previous_close = grouped["close"].shift(1)
    previous_volume = grouped["volume"].shift(1)

    features["log_return"] = np.log(features["close"] / previous_close)
    features["rolling_vol_20"] = grouped["log_return"].transform(
        lambda s: s.rolling(window=20, min_periods=20).std()
    )
    features["rolling_vol_60"] = grouped["log_return"].transform(
        lambda s: s.rolling(window=60, min_periods=60).std()
    )
    features["volume_change_1d"] = features["volume"] / previous_volume - 1
    features["volume_ma_20"] = grouped["volume"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    features["volume_ratio_20"] = features["volume"] / features["volume_ma_20"]
    features["abnormal_volume"] = np.where(
        features["volume_ratio_20"].notna(),
        features["volume_ratio_20"] > 2,
        pd.NA,
    )
    features["abnormal_volume"] = features["abnormal_volume"].astype("Int64")

    return features[FEATURE_COLUMNS]


def pivot_feature(features: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = features.pivot(index="date", columns="ticker", values=value)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)
    return matrix.reset_index()


def build_master_matrix(features: pd.DataFrame) -> pd.DataFrame:
    wide_parts = []
    for value in [
        "close",
        "log_return",
        "rolling_vol_20",
        "rolling_vol_60",
        "volume",
        "volume_ratio_20",
    ]:
        part = features.pivot(index="date", columns="ticker", values=value).sort_index(axis=1)
        part.columns = [f"{ticker}_{value}" for ticker in part.columns]
        wide_parts.append(part)

    return pd.concat(wide_parts, axis=1).sort_index().reset_index()


def validate_outputs(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    universe_tickers = set(universe["ticker"])
    for name, df in {
        "stock_prices": prices,
        "ticker_metadata": metadata,
        "stock_features": features,
    }.items():
        tickers = set(df["ticker"])
        if tickers != universe_tickers:
            raise ValueError(
                f"{name} ticker set mismatch. "
                f"Missing: {sorted(universe_tickers - tickers)}; "
                f"Extra: {sorted(tickers - universe_tickers)}"
            )

    if len(universe_tickers) != 118:
        raise ValueError(f"Expected 118 tickers, found {len(universe_tickers)}")

    if not prices["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("stock_prices.csv contains non YYYY-MM-DD dates")

    if prices[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("stock_prices.csv contains missing numeric price/volume values")


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    prices = load_prices(universe)
    metadata = build_metadata(universe)
    features = build_features(prices)
    validate_outputs(universe, prices, metadata, features)

    ticker_list = metadata[["ticker", "company_name", "industry", "exchange"]].copy()

    write_csv(ticker_list, "ticker_list.csv")
    write_csv(prices, "stock_prices.csv")
    write_csv(metadata, "ticker_metadata.csv")
    write_csv(features, "stock_features.csv")
    write_csv(pivot_feature(features, "log_return"), "master_log_return.csv")
    write_csv(pivot_feature(features, "rolling_vol_20"), "master_rolling_volatility_20.csv")
    write_csv(pivot_feature(features, "rolling_vol_60"), "master_rolling_volatility_60.csv")
    write_csv(pivot_feature(features, "close"), "master_close.csv")
    write_csv(build_master_matrix(features), "master_matrix.csv")

    print("Generated outputs in data/ and data/processed/")
    print(f"Tickers: {universe['ticker'].nunique()}")
    print(f"Price rows: {len(prices)}")
    print(f"Date range: {prices['date'].min()} to {prices['date'].max()}")
    print(f"Dataset scope: {active_dataset_scope()}")
    if not include_2026_append():
        print(f"2026 append disabled. Set {INCLUDE_2026_APPEND_ENV}=1 only for demo/robustness runs.")


if __name__ == "__main__":
    main()
