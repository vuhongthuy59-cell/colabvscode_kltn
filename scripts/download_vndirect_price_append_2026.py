from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TICKER_FILE = ROOT / "data" / "processed" / "ticker_list.csv"
OUT_FILE = ROOT / "data" / "data_origial" / "Stock_Price_2026_append.csv"

START_DATE = "2026-01-01"
END_DATE = "2026-04-30"
API_URL = "https://api-finfo.vndirect.com.vn/v4/stock_prices"


def load_tickers(only: str | None = None, limit: int | None = None) -> list[str]:
    tickers = pd.read_csv(TICKER_FILE, dtype={"ticker": "string"})["ticker"].dropna()
    values = tickers.astype(str).str.strip().str.upper().drop_duplicates().sort_values().tolist()
    if only:
        wanted = {item.strip().upper() for item in only.split(",") if item.strip()}
        values = [ticker for ticker in values if ticker in wanted]
    if limit:
        values = values[:limit]
    if not values:
        raise ValueError("No tickers selected.")
    return values


def download_one(session: requests.Session, ticker: str) -> pd.DataFrame:
    params = {
        "sort": "date",
        "q": f"code:{ticker}~date:gte:{START_DATE}~date:lte:{END_DATE}",
        "size": 1000,
    }
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    raw = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["nmVolume"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    out["volume"] = out["volume"].round().astype("int64")
    out = out[(out["date"] >= START_DATE) & (out["date"] <= END_DATE)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Comma-separated ticker list.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.08)
    args = parser.parse_args()

    tickers = load_tickers(only=args.only, limit=args.limit)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] Downloading {ticker}")
        try:
            data = download_one(session, ticker)
        except Exception as exc:
            print(f"  [FAIL] {ticker}: {exc}")
            failures.append(ticker)
            continue
        if data.empty:
            print(f"  [WARN] No rows for {ticker}")
            failures.append(ticker)
        else:
            print(f"  +{len(data)} rows: {data['date'].min()} -> {data['date'].max()}")
            frames.append(data)
        time.sleep(args.sleep)

    if not frames:
        raise RuntimeError("No price data downloaded.")

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    result = result[["ticker", "date", "open", "high", "low", "close", "volume"]]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_FILE, index=False)

    print("\nDone.")
    print(f"Saved: {OUT_FILE}")
    print(f"Rows: {len(result)}")
    print(f"Tickers: {result['ticker'].nunique()}/{len(tickers)}")
    print(f"Date range: {result['date'].min()} -> {result['date'].max()}")
    print(f"Duplicates: {result.duplicated(['ticker', 'date']).sum()}")
    if failures:
        print(f"Missing/failure tickers ({len(failures)}): {','.join(failures)}")


if __name__ == "__main__":
    main()
