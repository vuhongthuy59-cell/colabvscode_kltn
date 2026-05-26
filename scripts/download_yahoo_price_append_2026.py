from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
TICKER_FILE = ROOT / "data" / "processed" / "ticker_list.csv"
OUT_FILE = ROOT / "data" / "data_origial" / "Stock_Price_2026_append.csv"

START_DATE = "2026-01-01"
END_DATE_EXCLUSIVE = "2026-05-01"
PRICE_SCALE = 1000.0


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


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def download_one(ticker: str) -> pd.DataFrame:
    symbol = f"{ticker}.VN"
    raw = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE_EXCLUSIVE,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])

    raw = flatten_columns(raw.reset_index())
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"),
            "open": pd.to_numeric(raw["Open"], errors="coerce") / PRICE_SCALE,
            "high": pd.to_numeric(raw["High"], errors="coerce") / PRICE_SCALE,
            "low": pd.to_numeric(raw["Low"], errors="coerce") / PRICE_SCALE,
            "close": pd.to_numeric(raw["Close"], errors="coerce") / PRICE_SCALE,
            "volume": pd.to_numeric(raw["Volume"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    out["volume"] = out["volume"].round().astype("int64")
    out = out[(out["date"] >= START_DATE) & (out["date"] <= "2026-04-30")]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None, help="Comma-separated ticker list.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    tickers = load_tickers(only=args.only, limit=args.limit)
    rows: list[pd.DataFrame] = []
    failures: list[str] = []

    for idx, ticker in enumerate(tickers, 1):
        print(f"[{idx}/{len(tickers)}] Downloading {ticker}.VN")
        try:
            data = download_one(ticker)
        except Exception as exc:
            print(f"  [FAIL] {ticker}: {exc}")
            failures.append(ticker)
            continue
        if data.empty:
            print(f"  [WARN] No rows for {ticker}")
            failures.append(ticker)
        else:
            print(f"  +{len(data)} rows: {data['date'].min()} -> {data['date'].max()}")
            rows.append(data)
        time.sleep(args.sleep)

    if not rows:
        raise RuntimeError("No price data downloaded.")

    result = pd.concat(rows, ignore_index=True)
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
