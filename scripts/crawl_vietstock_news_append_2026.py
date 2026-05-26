from __future__ import annotations

import argparse
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver  # type: ignore
from selenium.common.exceptions import (  # type: ignore
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.service import Service  # type: ignore
from selenium.webdriver.common.by import By  # type: ignore
from selenium.webdriver.support import expected_conditions as EC  # type: ignore
from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
from webdriver_manager.chrome import ChromeDriverManager

ROOT = Path(__file__).resolve().parents[1]
TICKER_FILE = ROOT / "data" / "processed" / "ticker_list.csv"
OUTPUT_FILE = ROOT / "data" / "data_origial" / "Vietstock_News_2025_2026_append.xlsx"

START_DATE = "31/01/2025"
END_DATE = "30/04/2026"

WAIT_TIMEOUT = 25
MAX_RETRY_PER_TICKER = 3
SLEEP_BETWEEN_TICKERS = (1.5, 2.8)
SLEEP_BETWEEN_RETRY = (2.0, 4.0)
MAX_PAGES_PER_TICKER = 1000


def load_tickers(limit: int | None = None, only: str | None = None) -> list[str]:
    tickers = pd.read_csv(TICKER_FILE, dtype={"ticker": "string"})["ticker"].dropna()
    values = tickers.astype(str).str.strip().str.upper().drop_duplicates().sort_values().tolist()
    if only:
        wanted = {item.strip().upper() for item in only.split(",") if item.strip()}
        values = [ticker for ticker in values if ticker in wanted]
    if limit:
        values = values[:limit]
    if not values:
        raise ValueError("No tickers selected for crawling.")
    return values


def build_driver(headless: bool) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1440,1000")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    if headless:
        options.add_argument("--headless=new")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    return driver


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def parse_date(date_text: object) -> datetime | None:
    match = re.search(r"(\d{2}/\d{2}/\d{4})", normalize_text(date_text))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d/%m/%Y")
    except ValueError:
        return None


def get_date_range() -> tuple[datetime, datetime]:
    return datetime.strptime(START_DATE, "%d/%m/%Y"), datetime.strptime(END_DATE, "%d/%m/%Y")


def get_page_signature(driver: webdriver.Chrome) -> str:
    paging_text = ""
    first_href = ""
    first_date = ""

    try:
        paging_text = normalize_text(driver.find_element(By.CSS_SELECTOR, "div.pull-left.m-b").text)
    except Exception:
        pass

    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")
        if rows:
            first_row = rows[0]
            first_date = normalize_text(first_row.find_element(By.CLASS_NAME, "col-date").text)
            first_href = first_row.find_element(By.CLASS_NAME, "news-link").get_attribute("href") or ""
    except Exception:
        pass

    return f"{paging_text}|{first_date}|{first_href}"


def wait_for_rows(driver: webdriver.Chrome, timeout: int = WAIT_TIMEOUT) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: len(d.find_elements(By.CSS_SELECTOR, "table.table tbody tr")) > 0
    )


def wait_for_page_change(driver: webdriver.Chrome, old_signature: str, timeout: int = WAIT_TIMEOUT) -> None:
    WebDriverWait(driver, timeout).until(lambda d: get_page_signature(d) != old_signature)


def get_paging_info(driver: webdriver.Chrome) -> tuple[int | None, int | None]:
    try:
        paging_text = normalize_text(driver.find_element(By.CSS_SELECTOR, "div.pull-left.m-b").text)
        nums = re.findall(r"\d+", paging_text)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    except Exception:
        pass
    return None, None


def get_next_button(driver: webdriver.Chrome):
    try:
        li = driver.find_element(By.CSS_SELECTOR, "li[id^='new-page-next']")
        if "disabled" in (li.get_attribute("class") or "").lower():
            return None
        return li.find_element(By.TAG_NAME, "a")
    except NoSuchElementException:
        return None


def set_date_range_on_site(driver: webdriver.Chrome) -> None:
    script = """
    const startDate = arguments[0];
    const endDate = arguments[1];
    function setDate(selector, value) {
        const input = document.querySelector(selector);
        if (!input) return false;
        input.removeAttribute('readonly');
        input.value = value;
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.dispatchEvent(new Event('blur', { bubbles: true }));
        return true;
    }
    return {
        from_ok: setDate('#txtFromDate input', startDate),
        to_ok: setDate('#txtToDate input', endDate)
    };
    """
    result = driver.execute_script(script, START_DATE, END_DATE)
    if not result or not result.get("from_ok") or not result.get("to_ok"):
        raise RuntimeError("Could not set Vietstock date inputs.")


def extract_rows_from_current_page(driver: webdriver.Chrome, ticker: str, seen_keys: set[tuple]) -> list[dict]:
    results = []
    start_dt, end_dt = get_date_range()
    rows = driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")

    for row in rows:
        try:
            date_text = normalize_text(row.find_element(By.CLASS_NAME, "col-date").text)
            date_obj = parse_date(date_text)
            if date_obj is None or not (start_dt <= date_obj <= end_dt):
                continue

            link_el = row.find_element(By.CLASS_NAME, "news-link")
            title = normalize_text(link_el.text)
            href = (link_el.get_attribute("href") or "").strip()
            if not title or not href:
                continue

            key = (ticker, date_obj.strftime("%Y-%m-%d"), title, href)
            if key in seen_keys:
                continue

            seen_keys.add(key)
            results.append(
                {
                    "ticker": ticker,
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "title": title,
                    "url": href,
                    "source": "Vietstock",
                    "crawl_range_start": datetime.strptime(START_DATE, "%d/%m/%Y").strftime("%Y-%m-%d"),
                    "crawl_range_end": datetime.strptime(END_DATE, "%d/%m/%Y").strftime("%Y-%m-%d"),
                }
            )
        except (NoSuchElementException, StaleElementReferenceException):
            continue
        except Exception as exc:
            print(f"  [WARN] Skip one row for {ticker}: {exc}")

    return results


def save_to_excel(rows: list[dict]) -> None:
    if not rows:
        print("No rows to save yet.")
        return

    df = pd.DataFrame(rows)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])
    df = df.drop_duplicates(subset=["ticker", "date", "title", "url"])
    df = df.sort_values(["ticker", "date_dt", "title"], ascending=[True, False, True]).reset_index(drop=True)

    summary = (
        df.groupby("ticker", as_index=False)
        .size()
        .rename(columns={"size": "news_count"})
        .sort_values("news_count", ascending=False)
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}_temp{OUTPUT_FILE.suffix}")
    with pd.ExcelWriter(temp_file, engine="openpyxl") as writer:
        df.drop(columns=["date_dt"]).to_excel(writer, sheet_name="news", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    os.replace(temp_file, OUTPUT_FILE)


def load_existing_data() -> tuple[list[dict], set[tuple], set[str]]:
    if not OUTPUT_FILE.exists():
        return [], set(), set()

    try:
        try:
            df = pd.read_excel(OUTPUT_FILE, sheet_name="news")
        except Exception:
            df = pd.read_excel(OUTPUT_FILE)

        required = {"ticker", "date", "title", "url"}
        if not required.issubset(df.columns):
            return [], set(), set()

        df = df.dropna(subset=list(required)).copy()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["title"] = df["title"].map(normalize_text)
        df["url"] = df["url"].astype(str).str.strip()
        df = df.dropna(subset=["date"])

        rows = df.to_dict("records")
        seen_keys = {
            (row["ticker"], row["date"], row["title"], row["url"])
            for row in rows
        }
        completed_tickers = set(df["ticker"].unique())
        return rows, seen_keys, completed_tickers
    except Exception as exc:
        print(f"[WARN] Could not read existing output for resume: {exc}")
        return [], set(), set()


def crawl_ticker(driver: webdriver.Chrome, ticker: str, seen_keys: set[tuple]) -> list[dict]:
    url = f"https://finance.vietstock.vn/{ticker}/tin-moi-nhat.htm?languageid=1"

    for attempt in range(1, MAX_RETRY_PER_TICKER + 1):
        ticker_data: list[dict] = []
        visited_signatures: set[str] = set()

        try:
            driver.get(url)
            WebDriverWait(driver, WAIT_TIMEOUT).until(EC.element_to_be_clickable((By.ID, "btn-news-filter")))
            set_date_range_on_site(driver)

            search_button = driver.find_element(By.ID, "btn-news-filter")
            before_search = get_page_signature(driver)
            driver.execute_script("arguments[0].click();", search_button)

            wait_for_rows(driver)
            try:
                wait_for_page_change(driver, before_search, timeout=8)
            except TimeoutException:
                pass

            page_counter = 0
            while page_counter < MAX_PAGES_PER_TICKER:
                page_counter += 1
                current_sig = get_page_signature(driver)
                if current_sig in visited_signatures:
                    print(f"  [WARN] {ticker}: repeated page signature, stopping.")
                    break
                visited_signatures.add(current_sig)

                page_rows = extract_rows_from_current_page(driver, ticker, seen_keys)
                ticker_data.extend(page_rows)

                current_page, total_page = get_paging_info(driver)
                if current_page is not None and total_page is not None:
                    print(f"  Page {current_page}/{total_page} | +{len(page_rows)} rows")

                next_button = get_next_button(driver)
                if next_button is None:
                    break

                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                time.sleep(0.3)
                driver.execute_script("arguments[0].click();", next_button)
                wait_for_page_change(driver, current_sig)
                wait_for_rows(driver)

            return ticker_data
        except (TimeoutException, WebDriverException, RuntimeError) as exc:
            print(f"  [RETRY {attempt}/{MAX_RETRY_PER_TICKER}] {ticker}: {exc}")
            if attempt == MAX_RETRY_PER_TICKER:
                print(f"  [FAIL] Skip {ticker} after {MAX_RETRY_PER_TICKER} attempts.")
                return ticker_data
            time.sleep(random.uniform(*SLEEP_BETWEEN_RETRY))

    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated ticker list for a focused crawl.")
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    headless = False if args.visible else args.headless
    selected_tickers = load_tickers(limit=args.limit, only=args.only)
    rows, seen_keys, completed_tickers = load_existing_data()
    remaining = selected_tickers if args.no_skip_existing else [t for t in selected_tickers if t not in completed_tickers]

    print(f"Output: {OUTPUT_FILE}")
    print(f"Date range: {START_DATE} -> {END_DATE}")
    print(f"Existing rows: {len(rows)}")
    print(f"Completed tickers in output: {len(completed_tickers)}")
    print(f"Tickers to crawl: {len(remaining)}")

    if not remaining:
        print("Nothing to crawl.")
        return

    driver = build_driver(headless=headless)
    try:
        for i, ticker in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}] Crawling {ticker}")
            before = len(rows)
            ticker_rows = crawl_ticker(driver, ticker, seen_keys)
            rows.extend(ticker_rows)
            save_to_excel(rows)
            print(f"  --> {ticker}: +{len(rows) - before} rows | total {len(rows)}")
            time.sleep(random.uniform(*SLEEP_BETWEEN_TICKERS))
    finally:
        driver.quit()

    print("\nDone.")
    print(f"Final rows: {len(rows)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
