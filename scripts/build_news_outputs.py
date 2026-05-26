from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

NEWS_FILE = DATA_DIR / "data_origial" / "News_2022_2025_2.xlsx"
UNIVERSE_FILE = DATA_DIR / "processed" / "ticker_list.csv"
MASTER_CLOSE_FILE = DATA_DIR / "processed" / "master_close.csv"


MANUAL_ALIASES: dict[str, tuple[str, list[str]]] = {
    "VIC": ("Vingroup", ["Vingroup", "Tập đoàn Vingroup", "VinFast"]),
    "VHM": ("Vinhomes", ["Vinhomes", "CTCP Vinhomes"]),
    "VRE": ("Vincom Retail", ["Vincom Retail"]),
    "HPG": ("Tập đoàn Hòa Phát", ["Hòa Phát", "Tập đoàn Hòa Phát", "Hoa Phat"]),
    "FPT": ("FPT Corporation", ["FPT", "Tập đoàn FPT", "FPT Corporation"]),
    "MWG": ("Thế Giới Di Động", ["Thế Giới Di Động", "Mobile World", "MWG"]),
    "DGW": ("Thế Giới Số", ["Thế Giới Số", "Digiworld"]),
    "PNJ": ("Vàng bạc Đá quý Phú Nhuận", ["PNJ", "Phú Nhuận", "Vàng bạc Đá quý Phú Nhuận"]),
    "MSN": ("Masan Group", ["Masan", "Tập đoàn Masan", "Masan Group"]),
    "VNM": ("Vinamilk", ["Vinamilk", "Sữa Việt Nam"]),
    "SAB": ("Sabeco", ["Sabeco", "Bia Sài Gòn"]),
    "GAS": ("PV GAS", ["PV GAS", "Tổng Công ty Khí Việt Nam"]),
    "PLX": ("Petrolimex", ["Petrolimex", "Tập đoàn Xăng dầu Việt Nam"]),
    "POW": ("PV Power", ["PV Power", "Điện lực Dầu khí"]),
    "PVD": ("PV Drilling", ["PV Drilling", "Khoan Dầu khí"]),
    "PVS": ("PTSC", ["PTSC", "Dịch vụ Kỹ thuật Dầu khí"]),
    "BSR": ("Bình Sơn Refining", ["Lọc hóa dầu Bình Sơn", "BSR", "Bình Sơn"]),
    "VCB": ("Vietcombank", ["Vietcombank", "Ngoại thương Việt Nam"]),
    "BID": ("BIDV", ["BIDV", "Đầu tư và Phát triển Việt Nam"]),
    "CTG": ("VietinBank", ["VietinBank", "Công Thương Việt Nam"]),
    "TCB": ("Techcombank", ["Techcombank", "Kỹ thương Việt Nam"]),
    "MBB": ("MBBank", ["MBBank", "MB Bank", "Ngân hàng Quân đội"]),
    "VPB": ("VPBank", ["VPBank", "Việt Nam Thịnh Vượng"]),
    "ACB": ("ACB", ["ACB", "Á Châu"]),
    "HDB": ("HDBank", ["HDBank", "Phát triển TP.HCM"]),
    "LPB": ("LPBank", ["LPBank", "LienVietPostBank", "Lộc Phát Việt Nam"]),
    "STB": ("Sacombank", ["Sacombank", "Sài Gòn Thương Tín"]),
    "SHB": ("SHB", ["SHB", "Sài Gòn - Hà Nội", "Sài Gòn Hà Nội"]),
    "SSI": ("SSI Securities", ["SSI", "Chứng khoán SSI"]),
    "VND": ("VNDirect", ["VNDirect", "VNDIRECT"]),
    "HCM": ("HSC", ["HSC", "Chứng khoán HSC", "Chứng khoán TP.HCM"]),
    "VCI": ("Vietcap", ["Vietcap", "Chứng khoán Bản Việt"]),
    "VIX": ("VIX Securities", ["Chứng khoán VIX"]),
    "FTS": ("FPTS", ["FPTS", "Chứng khoán FPT"]),
    "KDH": ("Khang Điền", ["Khang Điền", "Nhà Khang Điền"]),
    "NLG": ("Nam Long", ["Nam Long"]),
    "KBC": ("Kinh Bắc", ["Kinh Bắc", "Kinh Bắc City"]),
    "DIG": ("DIC Corp", ["DIC Corp", "Tổng CTCP Đầu tư Phát triển Xây dựng", "DIC"]),
    "DXG": ("Đất Xanh", ["Đất Xanh", "Dat Xanh"]),
    "PDR": ("Phát Đạt", ["Phát Đạt", "Bất động sản Phát Đạt"]),
    "BCM": ("Becamex IDC", ["Becamex", "Becamex IDC"]),
    "SZC": ("Sonadezi Châu Đức", ["Sonadezi Châu Đức", "Sonadezi Chau Duc"]),
    "GVR": ("Tập đoàn Công nghiệp Cao su Việt Nam", ["GVR", "Cao su Việt Nam", "VRG"]),
    "DGC": ("Hóa chất Đức Giang", ["Đức Giang", "Hóa chất Đức Giang"]),
    "DPM": ("Đạm Phú Mỹ", ["Đạm Phú Mỹ", "PVFCCo"]),
    "DCM": ("Đạm Cà Mau", ["Đạm Cà Mau", "PVCFC"]),
    "BMP": ("Nhựa Bình Minh", ["Nhựa Bình Minh", "Bình Minh Plastic"]),
    "HSG": ("Hoa Sen Group", ["Hoa Sen", "Tập đoàn Hoa Sen"]),
    "NKG": ("Thép Nam Kim", ["Nam Kim", "Thép Nam Kim"]),
    "VHC": ("Vĩnh Hoàn", ["Vĩnh Hoàn"]),
    "ANV": ("Nam Việt", ["Nam Việt", "Navico"]),
    "BAF": ("BAF Việt Nam", ["BAF Việt Nam", "BAF"]),
    "DBC": ("Dabaco", ["Dabaco"]),
    "PAN": ("PAN Group", ["PAN Group", "Tập đoàn PAN"]),
    "KDC": ("KIDO", ["KIDO", "Kido"]),
    "QNS": ("Đường Quảng Ngãi", ["Đường Quảng Ngãi", "Quảng Ngãi Sugar"]),
    "VJC": ("Vietjet", ["Vietjet", "Vietjet Air"]),
    "ACV": ("ACV", ["Tổng công ty Cảng hàng không Việt Nam", "Cảng hàng không Việt Nam"]),
    "GMD": ("Gemadept", ["Gemadept"]),
    "HAH": ("Hải An", ["Hải An", "Vận tải Hải An"]),
    "SCS": ("SCSC", ["SCSC", "Dịch vụ Hàng hóa Sài Gòn"]),
    "VSC": ("Viconship", ["Viconship", "Container Việt Nam"]),
    "VTP": ("Viettel Post", ["Viettel Post", "Bưu chính Viettel"]),
    "REE": ("REE Corporation", ["REE", "Cơ điện lạnh"]),
    "GEX": ("GELEX", ["GELEX", "Gelex"]),
    "PC1": ("PC1 Group", ["PC1", "Xây lắp điện I"]),
    "VCG": ("Vinaconex", ["Vinaconex"]),
    "CTD": ("Coteccons", ["Coteccons"]),
    "HHV": ("Đèo Cả", ["Đèo Cả"]),
    "FCN": ("Fecon", ["Fecon", "FECON"]),
    "DHG": ("Dược Hậu Giang", ["Dược Hậu Giang"]),
    "TRA": ("Traphaco", ["Traphaco"]),
    "IMP": ("Imexpharm", ["Imexpharm"]),
    "CMG": ("CMC Corporation", ["CMC", "Tập đoàn CMC"]),
    "FOX": ("FPT Telecom", ["FPT Telecom"]),
    "VGI": ("Viettel Global", ["Viettel Global"]),
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

    primary_candidates = [
        ticker for ticker, mention in matched.items() if ticker == source_ticker or mention["mapping_method"] == "ticker_match"
    ]
    if not primary_candidates:
        max_mentions = max(int(m["mention_count"]) for m in matched.values())
        primary_candidates = [ticker for ticker, mention in matched.items() if int(mention["mention_count"]) == max_mentions]

    rows = []
    for ticker, mention in matched.items():
        is_primary = 1 if ticker in primary_candidates else 0
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

    raw_news = pd.read_excel(NEWS_FILE)
    raw_news = raw_news.rename(
        columns={
            "Mã": "source_ticker",
            "Ngày": "published_date",
            "Tiêu đề": "title",
            "Đường dẫn": "url",
        }
    )
    required = {"source_ticker", "published_date", "title", "url"}
    missing = required - set(raw_news.columns)
    if missing:
        raise ValueError(f"News input missing columns: {sorted(missing)}")

    raw_news["source_ticker"] = raw_news["source_ticker"].astype(str).str.strip().str.upper()
    raw_news = raw_news[raw_news["source_ticker"].isin(set(universe["ticker"]))].copy()
    raw_news["published_date"] = pd.to_datetime(raw_news["published_date"], dayfirst=True, errors="raise").dt.strftime("%Y-%m-%d")
    raw_news = raw_news.sort_values(["published_date", "source_ticker", "title"]).reset_index(drop=True)
    raw_news["article_id"] = [f"N{i:06d}" for i in range(1, len(raw_news) + 1)]
    raw_news["published_at"] = raw_news["published_date"]
    raw_news["source"] = "Vietstock"
    raw_news["title_clean"] = raw_news["title"].map(normalize_text)
    raw_news["category"] = raw_news["title_clean"].map(classify_category)
    raw_news["general_sentiment"] = raw_news["title_clean"].map(score_sentiment)
    raw_news["sentiment_label"] = raw_news["general_sentiment"].map(sentiment_label)

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


if __name__ == "__main__":
    main()
