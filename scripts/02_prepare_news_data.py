from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import (
    DEMO_END_DATE,
    INCLUDE_2026_APPEND_ENV,
    MAIN_END_DATE,
    MAIN_START_DATE,
    active_dataset_scope,
    include_2026_append,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "02_news_data"

NEWS_FILE = DATA_DIR / "data_origial" / "News_2022_2025_2.xlsx"
NEWS_APPEND_FILE = DATA_DIR / "data_origial" / "Vietstock_News_2025_2026_append.xlsx"
NEWS_CRAWL_FILE = DATA_DIR / "data_origial" / "Vietstock_News_2022_2025_crawl.xlsx"
UNIVERSE_FILE = ROOT / "outputs" / "01_price_data" / "ticker_list.csv"
MASTER_CLOSE_FILE = ROOT / "outputs" / "01_price_data" / "master_close.csv"


MANUAL_ALIASES: dict[str, tuple[str, list[str]]] = {
    # === Banking ===
    "ACB": ("ACB", ["ACB", "Á Châu", "ACB Bank", "Ngan hang A Chau"]),
    "BID": ("BIDV", ["BID", "BIDV", "Ngan hang Dau tu va Phat trien"]),
    "CTG": ("VietinBank", ["CTG", "VietinBank", "Ngan hang Cong Thuong"]),
    "EIB": ("Eximbank", ["EIB", "Eximbank", "Ngan hang Xuat nhap khau"]),
    "HDB": ("HDBank", ["HDB", "HDBank", "Ngan hang Phat trien TP HCM"]),
    "LPB": ("LPBank", ["LPB", "LPBank", "LienVietPostBank", "Loc Phat"]),
    "MBB": ("MBBank", ["MBB", "MBBank", "MB Bank", "Ngan hang Quan doi"]),
    "MSB": ("MSB", ["MSB", "MSB Bank", "Ngan hang Hang Hai", "Maritime Bank"]),
    "OCB": ("OCB", ["OCB", "OCB Bank", "Ngan hang Phuong Dong"]),
    "SHB": ("SHB", ["SHB", "SHB Bank", "Ngan hang Sai Gon Ha Noi"]),
    "SSB": ("SSB", ["SSB", "SeABank", "Ngan hang Dong Nam A"]),
    "STB": ("Sacombank", ["STB", "Sacombank", "Ngan hang Sai Gon Thuong Tin"]),
    "TCB": ("Techcombank", ["TCB", "Techcombank", "Ngan hang Ky Thuong"]),
    "TPB": ("TPBank", ["TPB", "TPBank", "Ngan hang Tien Phong"]),
    "VCB": ("Vietcombank", ["VCB", "Vietcombank", "Ngan hang Ngoai Thuong"]),
    "VIB": ("VIB", ["VIB", "VIB Bank", "Ngan hang Quoc te"]),
    "VPB": ("VPBank", ["VPB", "VPBank", "Ngan hang Viet Nam Thinh Vuong"]),
    # === Real Estate ===
    "AGG": ("An Gia", ["AGG", "An Gia", "Bat dong san An Gia"]),
    "BCM": ("Becamex IDC", ["BCM", "Becamex", "Becamex IDC"]),
    "CRE": ("Cen Land", ["CRE", "Cen Land", "Bat dong san The Ky"]),
    "DIG": ("DIC Corp", ["DIG", "DIC Corp", "Dau tu Phat trien Xay dung"]),
    "DXG": ("Dat Xanh", ["DXG", "Dat Xanh", "Bat dong san Dat Xanh"]),
    "HDG": ("Ha Do", ["HDG", "Ha Do", "Bat dong san Ha Do"]),
    "IJC": ("IJC", ["IJC", "Becamex Infrastructure"]),
    "KBC": ("Kinh Bac", ["KBC", "Kinh Bac", "Kinh Bac City"]),
    "KDH": ("Khang Dien", ["KDH", "Khang Dien", "Nha Khang Dien"]),
    "NLG": ("Nam Long", ["NLG", "Nam Long", "Bat dong san Nam Long"]),
    "NTL": ("Lideco", ["NTL", "Lideco", "Do thi Tu Liem", "Phat trien Do thi Tu Liem"]),
    "PDR": ("Phat Dat Realty", ["PDR", "Phat Dat", "Phat Dat Realty", "Phat Dat Real Estate Development"]),
    "SCR": ("SCR", ["SCR", "Sai Gon Construction"]),
    "SIP": ("SIP", ["SIP", "Saigon Infrastructure"]),
    "SZC": ("Sonadezi Chau Duc", ["SZC", "Sonadezi Chau Duc", "Sonadezi"]),
    "VHM": ("Vinhomes", ["VHM", "Vinhomes", "CTCP Vinhomes"]),
    "VIC": ("Vingroup", ["VIC", "Vingroup", "Tap doan Vingroup", "VinFast"]),
    "VRE": ("Vincom Retail", ["VRE", "Vincom Retail", "Vincom"]),
    # === Securities ===
    "AGR": ("Agriseco", ["AGR", "Agriseco", "Chung khoan Agribank"]),
    "BSI": ("BSI", ["BSI", "Chung khoan BIDV"]),
    "CTS": ("CTS", ["CTS", "Chung khoan Vietinbank"]),
    "FTS": ("FPTS", ["FTS", "FPTS", "Chung khoan FPT"]),
    "HCM": ("HSC", ["HCM", "HSC", "Chung khoan HSC", "Chung khoan TP HCM"]),
    "ORS": ("ORS", ["ORS", "Chung khoan Tien Phong"]),
    "SSI": ("SSI Securities", ["SSI", "Chung khoan SSI"]),
    "VCI": ("Vietcap", ["VCI", "Vietcap", "Chung khoan Ban Viet"]),
    "VIX": ("VIX Securities", ["VIX", "Chung khoan VIX"]),
    "VND": ("VNDirect", ["VND", "VNDirect", "Chung khoan VNDirect"]),
    # === Consumer Food & Agri ===
    "ACL": ("ACL", ["ACL", "XNK Thuy san An Giang", "An Giang Fish"]),
    "ANV": ("Nam Viet", ["ANV", "Nam Viet", "Navico"]),
    "BAF": ("BAF Viet Nam", ["BAF", "BAF Viet Nam"]),
    "DBC": ("Dabaco", ["DBC", "Dabaco", "Tap doan Dabaco"]),
    "FMC": ("FMC", ["FMC", "Thuy san Sam Se", "Samse"]),
    "IDI": ("IDI", ["IDI", "IDI Fisheries"]),
    "KDC": ("KIDO", ["KDC", "KIDO", "Kido", "Tap doan KIDO"]),
    "LTG": ("Loc Troi Group", ["LTG", "Loc Troi", "Tap doan Loc Troi"]),
    "MSN": ("Masan Group", ["MSN", "Masan", "Tap doan Masan", "Masan Group"]),
    "PAN": ("PAN Group", ["PAN", "PAN Group", "Tap doan PAN"]),
    "QNS": ("Duong Quang Ngai", ["QNS", "Quang Ngai Sugar", "Duong Quang Ngai"]),
    "SAB": ("Sabeco", ["SAB", "Sabeco", "Bia Sai Gon"]),
    "VHC": ("Vinh Hoan", ["VHC", "Vinh Hoan", "Vinh Hoan Fisheries"]),
    "VNM": ("Vinamilk", ["VNM", "Vinamilk", "Sua Viet Nam"]),
    # === Materials, Chemicals, Steel ===
    "AAA": ("An Phat Bioplastics", ["AAA", "An Phat Bioplastics", "Nhua An Phat Xanh"]),
    "APH": ("An Phat Holdings", ["APH", "An Phat Holdings", "Tap doan An Phat"]),
    "BFC": ("Binh Dien Fertilizer", ["BFC", "Binh Dien", "Phan bon Binh Dien"]),
    "BMP": ("Binh Minh Plastic", ["BMP", "Binh Minh", "Nhua Binh Minh"]),
    "CSV": ("CSV", ["CSV", "Hoa chat Co ban Mien Nam"]),
    "DCM": ("Dam Ca Mau", ["DCM", "Dam Ca Mau", "PVCFC", "Phan bon Ca Mau"]),
    "DGC": ("Duc Giang", ["DGC", "Duc Giang", "Hoa chat Duc Giang"]),
    "DPM": ("Dam Phu My", ["DPM", "Dam Phu My", "PVFCCo", "Phan bon Phu My"]),
    "GVR": ("VRG", ["GVR", "Cao su Viet Nam", "VRG", "Tap doan Cao su"]),
    "HPG": ("Hoa Phat", ["HPG", "Hoa Phat", "Tap doan Hoa Phat"]),
    "HSG": ("Hoa Sen Group", ["HSG", "Hoa Sen", "Tap doan Hoa Sen"]),
    "NKG": ("Nam Kim", ["NKG", "Nam Kim", "Thep Nam Kim"]),
    "PHR": ("PHR", ["PHR", "Cao su Phuoc Hoa"]),
    # === Energy, Utilities, Oil & Gas ===
    "BSR": ("Binh Son Refining", ["BSR", "Binh Son", "Loc hoa dau Binh Son", "Dung Quat"]),
    "BWE": ("BWE", ["BWE", "Cap nuoc Binh Duong"]),
    "GAS": ("PV GAS", ["GAS", "PV GAS", "Khi Viet Nam"]),
    "GEG": ("GEG", ["GEG", "Dien Gia Lai"]),
    "NT2": ("NT2", ["NT2", "Dien Nhon Trach"]),
    "PC1": ("PC1 Group", ["PC1", "PC1 Group", "Xay lap dien 1"]),
    "PGV": ("PV Power", ["PGV", "PV Power", "Power Generation"]),
    "PLX": ("Petrolimex", ["PLX", "Petrolimex", "Tap doan Xang dau Viet Nam"]),
    "POW": ("PV Power", ["POW", "PV Power", "Dien luc Dau khi"]),
    "PPC": ("PPC", ["PPC", "Nhiet dien Pha Lai"]),
    "PVD": ("PV Drilling", ["PVD", "PV Drilling", "Khoan Dau khi", "Petrovietnam Drilling"]),
    "PVS": ("PTSC", ["PVS", "PTSC", "Dich vu Ky thuat Dau khi"]),
    # === Construction & Infrastructure ===
    "C4G": ("C4G", ["C4G", "Cienco4"]),
    "CTD": ("Coteccons", ["CTD", "Coteccons"]),
    "DPG": ("DPG", ["DPG", "Dau tu Phat trien GTVT"]),
    "FCN": ("Fecon", ["FCN", "Fecon", "FECON"]),
    "HHV": ("Deo Ca", ["HHV", "Deo Ca", "Tap doan Deo Ca"]),
    "LCG": ("LCG", ["LCG", "Lizen"]),
    "VCG": ("Vinaconex", ["VCG", "Vinaconex"]),
    # === ICT & Telecom ===
    "CMG": ("CMC Corporation", ["CMG", "CMC", "Tap doan CMC"]),
    "ELC": ("ELC", ["ELC", "Dien tu Viet Nam"]),
    "FOX": ("FPT Telecom", ["FOX", "FPT Telecom"]),
    "FPT": ("FPT Corporation", ["FPT", "Tap doan FPT"]),
    "VGI": ("Viettel Global", ["VGI", "Viettel Global", "Viettel"]),
    # === Healthcare & Pharma ===
    "DBD": ("DBD", ["DBD", "Duoc pham BT"]),
    "DHG": ("Hau Giang Pharma", ["DHG", "Hau Giang", "Duoc Hau Giang"]),
    "DMC": ("DMC", ["DMC", "Duoc pham DMC"]),
    "IMP": ("Imexpharm", ["IMP", "Imexpharm"]),
    "JVC": ("JVC", ["JVC", "Thiet bi Y te Viet Nhat"]),
    "TRA": ("Traphaco", ["TRA", "Traphaco"]),
    # === Retail & Distribution ===
    "DGW": ("Digiworld", ["DGW", "Digiworld", "The gioi so"]),
    "FRT": ("FPT Retail", ["FRT", "FPT Retail", "FPT Shop"]),
    "MWG": ("Mobile World", ["MWG", "The gioi Di dong", "Dien May Xanh", "Mobile World"]),
    "PNJ": ("Phu Nhuan Jewelry", ["PNJ", "PNJ", "Phu Nhuan", "Vang bac Phu Nhuan"]),
    # === Transport, Logistics & Aviation ===
    "AST": ("AST", ["AST", "Dich vu Hang khong Tan Son Nhat"]),
    "GMD": ("Gemadept", ["GMD", "Gemadept"]),
    "HAH": ("Hai An", ["HAH", "Hai An", "Van tai Hai An"]),
    "SCS": ("SCSC", ["SCS", "SCS", "Dich vu Hang hoa Sai Gon"]),
    "SGN": ("Sai Gon Cargo", ["SGN", "Sai Gon Cargo"]),
    "VJC": ("Vietjet", ["VJC", "Vietjet", "Vietjet Air"]),
    "VSC": ("Viconship", ["VSC", "Viconship", "Container Viet Nam"]),
    "VTP": ("Viettel Post", ["VTP", "Viettel Post", "Buu chinh Viettel"]),
    # === Industrials & Holdings ===
    "GEX": ("GELEX", ["GEX", "GELEX", "Gelex"]),
    "REE": ("REE Corporation", ["REE", "Co dien lanh", "REE Corp"]),
    # === Others ===
    "ACV": ("ACV", ["ACV", "Cang hang khong Viet Nam"]),
    "ABB": ("ABBANK", ["ABB", "ABBANK", "Ngan hang An Binh"]),
}


CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("legal_regulatory", ["xử phạt", "bị phạt", "thanh tra", "điều tra", "vi phạm", "đình chỉ", "hủy niêm yết", "huỷ niêm yết", "kiểm soát", "cảnh báo"]),
    ("debt_bond", ["trái phiếu", "nợ", "đáo hạn", "chậm trả", "tái cấu trúc nợ", "lãi trái phiếu"]),
    ("earnings", ["lợi nhuận", "lãi", "lỗ", "doanh thu", "kết quả kinh doanh", "quý", "năm tài chính"]),
    ("ma_ownership", ["mua lại", "sáp nhập", "thâu tóm", "chuyển nhượng", "thoái vốn", "cổ đông lớn", "sở hữu"]),
    ("capital_issuance", ["phát hành", "tăng vốn", "cổ phiếu thưởng", "esop", "riêng lẻ", "chào bán"]),
    ("dividend", ["cổ tức", "chia cổ tức", "trả cổ tức", "giao dịch không hưởng quyền"]),
    ("leadership", ["bổ nhiệm", "từ nhiệm", "chủ tịch", "tổng giám đốc", "ban lãnh đạo", "hđqt", "hội đồng quản trị"]),
    ("project_contract", ["dự án", "hợp đồng", "trúng thầu", "ký kết", "bàn giao", "khởi công"]),
    ("market_industry", ["giá thép", "giá dầu", "ngành ngân hàng", "bất động sản", "xuất khẩu", "vn-index", "thị trường chứng khoán"]),
]

POSITIVE_KEYWORDS = [
    "tăng trưởng",
    "tăng mạnh",
    "vượt kế hoạch",
    "lãi lớn",
    "lãi kỷ lục",
    "lãi ky lục",
    "phục hồi",
    "ký hợp đồng",
    "trúng thầu",
    "mở rộng",
    "chia cổ tức",
    "tăng vốn thành công",
    "hoàn thành",
    "cải thiện",
    "khởi sắc",
    "lãi",
    "tăng",
    "kỷ lục",
]

NEGATIVE_KEYWORDS = [
    "lỗ",
    "giảm mạnh",
    "sụt giảm",
    "lao dốc",
    "chậm trả",
    "trái phiếu quá hạn",
    "quá hạn",
    "xử phạt",
    "bị phạt",
    "thanh tra",
    "điều tra",
    "vi phạm",
    "hủy niêm yết",
    "huỷ niêm yết",
    "đình chỉ",
    "từ nhiệm",
    "phá sản",
    "cảnh báo",
    "kiểm soát",
    "bán tháo",
    "lỗ ròng",
    "giảm",
    "nợ",
]


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(normalize_text(term))
    return re.compile(rf"(?<![0-9a-zA-ZÀ-ỹ]){escaped}(?![0-9a-zA-ZÀ-ỹ])", re.IGNORECASE)


def count_matches(text: str, term: str) -> int:
    if not term:
        return 0
    return len(token_pattern(term).findall(text))


def load_universe() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_FILE, dtype={"ticker": "string"})
    universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
    if len(universe) != 118:
        raise ValueError(f"Expected 118 tickers, found {len(universe)}")
    return universe


def build_aliases(universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in universe["ticker"]:
        company_name, aliases = MANUAL_ALIASES.get(ticker, ("", []))
        alias_values = [ticker, *aliases]
        if company_name:
            alias_values.append(company_name)
        for alias in dict.fromkeys([a.strip() for a in alias_values if a and a.strip()]):
            if alias.upper() == ticker:
                method = "ticker_match"
            elif company_name and normalize_text(alias) == normalize_text(company_name):
                method = "company_name_match"
            else:
                method = "alias_match"
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "alias": alias,
                    "mapping_method": method,
                }
            )
    aliases = pd.DataFrame(rows).sort_values(["ticker", "alias"]).reset_index(drop=True)
    return aliases


def classify_category(title_clean: str) -> str:
    for category, keywords in CATEGORY_RULES:
        if any(keyword in title_clean for keyword in keywords):
            return category
    return "other"


def score_sentiment(title_clean: str) -> float:
    positive = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in title_clean)
    negative = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in title_clean)
    score = (positive - negative) / 3
    return float(np.clip(score, -1, 1))


def sentiment_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


def next_trading_date(published_date: str, trading_dates: list[str]) -> str:
    idx = bisect_left(trading_dates, published_date)
    if idx >= len(trading_dates):
        return ""
    return trading_dates[idx]


def build_mentions_for_article(
    article_id: str,
    source_ticker: str,
    title_clean: str,
    sentiment: float,
    alias_records: list[dict[str, object]],
    company_names: dict[str, str],
) -> list[dict[str, object]]:
    matched: dict[str, dict[str, object]] = {}

    for row in alias_records:
        count = len(row["pattern"].findall(title_clean))
        if count == 0:
            continue

        ticker = str(row["ticker"])
        current = matched.get(ticker)
        method_rank = {"ticker_match": 3, "company_name_match": 2, "alias_match": 1}
        method = str(row["mapping_method"])
        if current is None or count > current["mention_count"] or method_rank[method] > method_rank[current["mapping_method"]]:
            matched[ticker] = {
                "article_id": article_id,
                "ticker": ticker,
                "company_name": row["company_name"],
                "mention_count": count,
                "mapping_method": method,
                "matched_text": row["alias"],
            }

    if source_ticker and source_ticker not in matched:
        matched[source_ticker] = {
            "article_id": article_id,
            "ticker": source_ticker,
            "company_name": company_names.get(source_ticker, ""),
            "mention_count": 1,
            "mapping_method": "source_ticker",
            "matched_text": source_ticker,
        }

    if not matched:
        return []

    # === FIX PRIMARY: chi 1 primary per article ===
    # Priority: source_ticker > ticker_match > company_name_match > alias_match > max mention_count
    primary_candidates = [ticker for ticker, mention in matched.items() if ticker == source_ticker]
    if not primary_candidates:
        primary_candidates = [ticker for ticker, mention in matched.items() if mention["mapping_method"] == "ticker_match"]
    if not primary_candidates:
        primary_candidates = [ticker for ticker, mention in matched.items() if mention["mapping_method"] == "company_name_match"]
    if not primary_candidates:
        primary_candidates = [ticker for ticker, mention in matched.items() if mention["mapping_method"] == "alias_match"]
    if not primary_candidates:
        max_mentions = max(int(m["mention_count"]) for m in matched.values())
        primary_candidates = [ticker for ticker, mention in matched.items() if int(mention["mention_count"]) == max_mentions]
    primary_ticker = primary_candidates[0] if primary_candidates else list(matched.keys())[0]

    rows = []
    for ticker, mention in matched.items():
        is_primary = 1 if ticker == primary_ticker else 0
        method = str(mention["mapping_method"])
        base_relevance = {
            "ticker_match": 1.0,
            "company_name_match": 0.95,
            "alias_match": 0.9,
            "source_ticker": 0.8,
        }.get(method, 0.75)
        relevance = min(1.0, base_relevance + 0.05 * (int(mention["mention_count"]) - 1))
        company_sentiment = sentiment * relevance
        mention.update(
            {
                "is_primary": is_primary,
                "relevance_score": round(relevance, 4),
                "company_sentiment": round(company_sentiment, 6),
            }
        )
        rows.append(mention)

    return rows


def load_raw_news() -> pd.DataFrame:
    old_news = pd.read_excel(NEWS_FILE)
    old_news = old_news.rename(
        columns={
            "Mã": "source_ticker",
            "Ngày": "published_date",
            "Tiêu đề": "title",
            "Đường dẫn": "url",
            "MÃ£": "source_ticker",
            "NgÃ y": "published_date",
            "TiÃªu Ä‘á»": "title",
            "ÄÆ°á»ng dáº«n": "url",
        }
    )

    frames = [old_news]
    if include_2026_append() and NEWS_APPEND_FILE.exists():
        append_news = pd.read_excel(NEWS_APPEND_FILE, sheet_name="news")
        append_news = append_news.rename(
            columns={
                "ticker": "source_ticker",
                "date": "published_date",
                "title": "title",
                "url": "url",
            }
        )
        frames.append(append_news)

    if NEWS_CRAWL_FILE.exists():
        crawl_news = pd.read_excel(NEWS_CRAWL_FILE)
        crawl_news = crawl_news.rename(
            columns={
                "ticker": "source_ticker",
                "date": "published_date",
            }
        )
        frames.append(crawl_news)

    raw_news = pd.concat(frames, ignore_index=True)
    required = {"source_ticker", "published_date", "title", "url"}
    missing = required - set(raw_news.columns)
    if missing:
        raise ValueError(f"News input missing columns: {sorted(missing)}")

    raw_news = raw_news[list(required)].copy()
    raw_news["source_ticker"] = raw_news["source_ticker"].astype(str).str.strip().str.upper()
    raw_news["published_date"] = pd.to_datetime(
        raw_news["published_date"],
        format="mixed",
        dayfirst=True,
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    end_date = DEMO_END_DATE if include_2026_append() else MAIN_END_DATE
    raw_news = raw_news[(raw_news["published_date"] >= MAIN_START_DATE) & (raw_news["published_date"] <= end_date)].copy()
    raw_news["title"] = raw_news["title"].astype(str).map(lambda value: re.sub(r"\s+", " ", value.strip()))
    raw_news["url"] = raw_news["url"].astype(str).str.strip()
    # === FIX DEDUP ===
    before = len(raw_news)
    raw_news["_title_norm"] = raw_news["title"].str.strip().str.lower()
    raw_news = raw_news.drop_duplicates(["source_ticker", "published_date", "_title_norm"], keep="first")
    raw_news = raw_news.drop(columns=["_title_norm"])
    after = len(raw_news)
    if before > after:
        print("  [Dedup] Removed", before - after, "crawl duplicates (", after, "remaining)")
    raw_news = raw_news.drop_duplicates(["source_ticker", "url"], keep="last")
    raw_news = raw_news.drop_duplicates(["source_ticker", "published_date", "title", "url"], keep="last")
    return raw_news.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    aliases = build_aliases(universe)
    alias_rows = aliases.sort_values("alias", key=lambda s: s.str.len(), ascending=False)
    alias_records = []
    for row in alias_rows.itertuples(index=False):
        alias_records.append(
            {
                "ticker": row.ticker,
                "company_name": row.company_name,
                "alias": row.alias,
                "mapping_method": row.mapping_method,
                "pattern": token_pattern(row.alias),
            }
        )
    company_names = aliases.groupby("ticker")["company_name"].first().to_dict()

    raw_news = load_raw_news()
    raw_news = raw_news[raw_news["source_ticker"].isin(set(universe["ticker"]))].copy()
    raw_news = raw_news.sort_values(["published_date", "source_ticker", "title"]).reset_index(drop=True)
    raw_news["article_id"] = [f"N{i:06d}" for i in range(1, len(raw_news) + 1)]
    raw_news["published_at"] = raw_news["published_date"]
    raw_news["source"] = "Vietstock"
    raw_news["title_clean"] = raw_news["title"].map(normalize_text)
    raw_news["category"] = raw_news["title_clean"].map(classify_category)
    raw_news["general_sentiment"] = raw_news["title_clean"].map(score_sentiment)
    raw_news["sentiment_label"] = raw_news["general_sentiment"].map(sentiment_label)

    # === LOAD MANUAL LABELS (override rule-based) ===
    MANUAL_LABELS_FILE = DATA_DIR / "manual_labels.csv"
    if MANUAL_LABELS_FILE.exists():
        manual = pd.read_csv(MANUAL_LABELS_FILE)
        manual = manual.dropna(subset=["manual_sentiment", "manual_category"])
        if len(manual) > 0:
            sent_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            manual["general_sentiment"] = manual["manual_sentiment"].map(sent_map)
            manual["sentiment_label"] = manual["manual_sentiment"]
            manual["category"] = manual["manual_category"]
            
            for _, row in manual.iterrows():
                mask = raw_news["article_id"] == row["article_id"]
                if mask.any():
                    raw_news.loc[mask, "general_sentiment"] = row["general_sentiment"]
                    raw_news.loc[mask, "sentiment_label"] = row["sentiment_label"]
                    raw_news.loc[mask, "category"] = row["category"]
            print(f"  [Manual Labels] Overrode {len(manual)} articles with manual labels")

    trading_dates = pd.read_csv(MASTER_CLOSE_FILE, usecols=["date"])["date"].astype(str).tolist()
    raw_news["event_trading_date"] = raw_news["published_date"].map(lambda d: next_trading_date(d, trading_dates))

    all_mentions = []
    for row in raw_news.itertuples(index=False):
        all_mentions.extend(
            build_mentions_for_article(
                article_id=row.article_id,
                source_ticker=row.source_ticker,
                title_clean=row.title_clean,
                sentiment=float(row.general_sentiment),
                alias_records=alias_records,
                company_names=company_names,
            )
        )

    mentions = pd.DataFrame(
        all_mentions,
        columns=[
            "article_id",
            "ticker",
            "company_name",
            "is_primary",
            "mention_count",
            "relevance_score",
            "company_sentiment",
            "mapping_method",
            "matched_text",
        ],
    )
    mentions = mentions[mentions["ticker"].isin(set(universe["ticker"]))].copy()

    mention_counts = mentions.groupby("article_id")["ticker"].nunique()
    raw_news["n_mapped_tickers"] = raw_news["article_id"].map(mention_counts).fillna(0).astype(int)
    raw_news["is_firm_specific"] = (raw_news["n_mapped_tickers"] > 0).astype(int)
    articles = raw_news[raw_news["is_firm_specific"].eq(1)].copy()
    mentions = mentions[mentions["article_id"].isin(set(articles["article_id"]))].copy()

    articles = articles[
        [
            "article_id",
            "published_at",
            "published_date",
            "event_trading_date",
            "source",
            "title",
            "title_clean",
            "category",
            "general_sentiment",
            "sentiment_label",
            "n_mapped_tickers",
            "is_firm_specific",
        ]
    ]

    aliases.to_csv(OUT_DIR / "ticker_aliases.csv", index=False, encoding="utf-8-sig")
    articles.to_csv(OUT_DIR / "news_articles.csv", index=False, encoding="utf-8-sig")
    mentions.to_csv(OUT_DIR / "news_mentions.csv", index=False, encoding="utf-8-sig")

    print("Generated news outputs in data/ and data/processed/")
    print(f"Articles: {len(articles)}")
    print(f"Mentions: {len(mentions)}")
    print(f"Aliases: {len(aliases)}")
    print(f"Trading-date missing: {articles['event_trading_date'].eq('').sum()}")
    print(f"Dataset scope: {active_dataset_scope()}")
    if not include_2026_append():
        print(f"2026 append disabled. Set {INCLUDE_2026_APPEND_ENV}=1 only for demo/robustness runs.")


if __name__ == "__main__":
    main()
