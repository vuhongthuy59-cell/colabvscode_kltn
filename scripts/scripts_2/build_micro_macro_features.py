from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"

PRICE_FILE = OUT_DIR / "stock_prices.csv"
LOG_RETURN_FILE = OUT_DIR / "master_log_return.csv"
STOCK_MICRO_FILE = OUT_DIR / "stock_micro_features.csv"
MARKET_MACRO_FILE = OUT_DIR / "market_macro_features.csv"
MICRO_QUALITY_FILE = OUT_DIR / "micro_macro_quality_report.csv"

START_DATE = "2022-01-01"
END_DATE = "2026-04-30"
VNDIRECT_RATIOS_URL = "https://api-finfo.vndirect.com.vn/v4/ratios"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def download_market_cap(ticker: str, session: requests.Session) -> pd.DataFrame:
    params = {
        "q": f"code:{ticker}~ratioCode:MARKETCAP~reportDate:gte:{START_DATE}~reportDate:lte:{END_DATE}",
        "size": 5000,
        "sort": "reportDate",
    }
    response = session.get(VNDIRECT_RATIOS_URL, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "market_cap"])
    raw = pd.DataFrame(rows)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(raw["reportDate"]).dt.strftime("%Y-%m-%d"),
            "market_cap": pd.to_numeric(raw["value"], errors="coerce"),
        }
    ).dropna()


def build_stock_micro_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["trading_value"] = prices["close"] * prices["volume"]
    prices["trading_value_ma_20"] = prices.groupby("ticker")["trading_value"].transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    prices["trading_value_ratio_20"] = prices["trading_value"] / prices["trading_value_ma_20"]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    market_cap_frames = []
    failed_market_cap = []
    tickers = sorted(prices["ticker"].unique())
    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] market cap {ticker}")
        try:
            frame = download_market_cap(ticker, session)
        except Exception as exc:
            print(f"  [WARN] {ticker}: {exc}")
            failed_market_cap.append(ticker)
            continue
        if frame.empty:
            failed_market_cap.append(ticker)
        else:
            market_cap_frames.append(frame)

    if market_cap_frames:
        market_caps = pd.concat(market_cap_frames, ignore_index=True)
    else:
        market_caps = pd.DataFrame(columns=["ticker", "date", "market_cap"])

    merged_frames = []
    for ticker, group in prices.groupby("ticker", sort=True):
        base = group.sort_values("date").copy()
        cap = market_caps[market_caps["ticker"].eq(ticker)].sort_values("date")
        if cap.empty:
            base["market_cap"] = np.nan
        else:
            base["date_key"] = pd.to_datetime(base["date"])
            cap = cap.copy()
            cap["date_key"] = pd.to_datetime(cap["date"])
            base = pd.merge_asof(
                base.sort_values("date_key"),
                cap[["date_key", "market_cap"]].sort_values("date_key"),
                on="date_key",
                direction="backward",
            )
            base = base.drop(columns=["date_key"])
        merged_frames.append(base)

    micro = pd.concat(merged_frames, ignore_index=True)
    micro["market_cap"] = micro.groupby("ticker")["market_cap"].ffill().bfill()
    micro["log_market_cap"] = np.log1p(micro["market_cap"].fillna(0.0))
    micro["trading_value_ratio_20"] = micro["trading_value_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    micro["log_market_cap"] = micro["log_market_cap"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = micro[["date", "ticker", "trading_value", "trading_value_ratio_20", "market_cap", "log_market_cap"]]
    quality = {
        "micro_rows": len(out),
        "micro_tickers": out["ticker"].nunique(),
        "market_cap_failed_tickers": len(failed_market_cap),
        "market_cap_missing_rows_after_fill": int(out["market_cap"].isna().sum()),
    }
    return out, quality


def download_yahoo_close(symbol: str, column_name: str, trading_dates: list[str]) -> pd.DataFrame:
    raw = yf.download(symbol, start=START_DATE, end="2026-05-01", progress=False, auto_adjust=False, threads=False)
    if raw.empty:
        return pd.DataFrame({"date": trading_dates, column_name: np.nan})
    raw = flatten_columns(raw.reset_index())
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"),
            column_name: pd.to_numeric(raw["Close"], errors="coerce"),
        }
    )
    calendar = pd.DataFrame({"date": trading_dates})
    merged = pd.merge(calendar, frame, on="date", how="left")
    merged[column_name] = merged[column_name].ffill()
    return merged


def download_vni_msn(trading_dates: list[str]) -> pd.DataFrame:
    try:
        from vnstock.api.quote import Quote

        q = Quote(symbol="VNINDEX", source="MSN")
        raw = q.history(start=START_DATE, end=END_DATE, interval="1D")
        if raw.empty:
            raise ValueError("VNINDEX history is empty")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(raw["time"]).dt.strftime("%Y-%m-%d"),
                "vni_close": pd.to_numeric(raw["close"], errors="coerce"),
            }
        )
    except Exception as exc:
        print(f"[WARN] Could not download VNINDEX from vnstock/MSN: {exc}")
        frame = pd.DataFrame(columns=["date", "vni_close"])

    calendar = pd.DataFrame({"date": trading_dates})
    merged = pd.merge(calendar, frame, on="date", how="left")
    merged["vni_available"] = merged["vni_close"].notna().astype(int)
    merged["vni_close"] = merged["vni_close"].ffill()
    return merged


def build_market_macro_features(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    log_returns = pd.read_csv(LOG_RETURN_FILE)
    trading_dates = log_returns["date"].astype(str).tolist()
    ticker_cols = [col for col in log_returns.columns if col != "date"]

    macro = pd.DataFrame({"date": trading_dates})
    macro["universe_market_return"] = log_returns[ticker_cols].mean(axis=1, skipna=True).fillna(0.0)
    macro["universe_market_roll_vol_20"] = macro["universe_market_return"].rolling(window=20, min_periods=20).std().fillna(0.0)

    liquidity = prices.copy()
    liquidity["trading_value"] = liquidity["close"] * liquidity["volume"]
    daily_liquidity = liquidity.groupby("date", as_index=False)["trading_value"].sum().rename(
        columns={"trading_value": "market_liquidity"}
    )
    macro = macro.merge(daily_liquidity, on="date", how="left")
    macro["market_liquidity"] = macro["market_liquidity"].fillna(0.0)
    macro["market_liquidity_ma_20"] = macro["market_liquidity"].rolling(window=20, min_periods=20).mean()
    macro["market_liquidity_ratio_20"] = (
        macro["market_liquidity"] / macro["market_liquidity_ma_20"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vni = download_vni_msn(trading_dates)
    macro = macro.merge(vni, on="date", how="left")
    macro["vni_return_1d"] = np.log(macro["vni_close"] / macro["vni_close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    macro["vni_roll_vol_20"] = macro["vni_return_1d"].rolling(window=20, min_periods=20).std()
    macro["vni_return_1d"] = macro["vni_return_1d"].fillna(macro["universe_market_return"])
    macro["vni_roll_vol_20"] = macro["vni_roll_vol_20"].fillna(macro["universe_market_roll_vol_20"])

    usd = download_yahoo_close("USDVND=X", "usd_vnd_close", trading_dates)
    oil = download_yahoo_close("CL=F", "oil_close", trading_dates)
    macro = macro.merge(usd, on="date", how="left").merge(oil, on="date", how="left")
    macro["usd_vnd_return_1d"] = np.log(macro["usd_vnd_close"] / macro["usd_vnd_close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro["oil_return_1d"] = np.log(macro["oil_close"] / macro["oil_close"].shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cols = [
        "date",
        "universe_market_return",
        "universe_market_roll_vol_20",
        "market_liquidity_ratio_20",
        "vni_return_1d",
        "vni_roll_vol_20",
        "vni_available",
        "usd_vnd_return_1d",
        "oil_return_1d",
    ]
    out = macro[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    quality = {
        "macro_rows": len(out),
        "vni_available_rows": int(out["vni_available"].sum()),
        "usd_vnd_nonzero_return_rows": int(out["usd_vnd_return_1d"].ne(0).sum()),
        "oil_nonzero_return_rows": int(out["oil_return_1d"].ne(0).sum()),
    }
    return out, quality


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_FILE)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = prices["date"].astype(str)

    micro, micro_quality = build_stock_micro_features(prices)
    macro, macro_quality = build_market_macro_features(prices)

    micro.to_csv(STOCK_MICRO_FILE, index=False)
    macro.to_csv(MARKET_MACRO_FILE, index=False)

    quality = {**micro_quality, **macro_quality}
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in quality.items()]
    ).to_csv(MICRO_QUALITY_FILE, index=False)

    print(f"Wrote {STOCK_MICRO_FILE}")
    print(f"Wrote {MARKET_MACRO_FILE}")
    print(pd.DataFrame([quality]).to_string(index=False))


if __name__ == "__main__":
    main()
