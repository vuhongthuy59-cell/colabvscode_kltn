from __future__ import annotations

import itertools
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "04_company_relationships"

METADATA_FILE = ROOT / "outputs" / "01_price_data" / "ticker_metadata.csv"
RELATIONSHIP_FILE = DATA_DIR / "data_origial" / "relationships.xlsx"
OWNERSHIP_GRAPH_FILE = DATA_DIR / "data_2" / "vietnam_ownership_graph_dataset_v1.xlsx"

DEFAULT_SAME_GROUP_WEIGHT = 0.80
DEFAULT_SAME_INDUSTRY_WEIGHT = 0.50
DEFAULT_PARENT_TO_SUB_WEIGHT = 1.00
DEFAULT_SUB_TO_PARENT_WEIGHT = 0.70
DEFAULT_STRATEGIC_ECOSYSTEM_WEIGHT = 0.90
DEFAULT_BUSINESS_CLUSTER_WEIGHT = 0.62
DEFAULT_VALUE_CHAIN_WEIGHT = 0.68


COMPANY_GROUP_SEEDS = [
    ("G001", "Vingroup", ["VIC", "VHM", "VRE"]),
    ("G002", "FPT Group", ["FPT", "FOX"]),
    ("G003", "PetroVietnam ecosystem", ["BSR", "DPM", "DCM", "GAS", "POW", "PVD", "PVS"]),
]


STRATEGIC_ECOSYSTEM_GROUPS = [
    ("E001", "PetroVietnam ecosystem", ["BSR", "DPM", "DCM", "GAS", "POW", "PVD", "PVS"]),
    ("E002", "Vingroup ecosystem before Vincom Retail exit", ["VIC", "VHM", "VRE"], "", "2023-12-31"),
    ("E003", "Vingroup core real estate ecosystem", ["VIC", "VHM"], "2024-01-01", ""),
    ("E004", "An Phat ecosystem", ["APH", "AAA"]),
    ("E005", "VRG rubber industrial ecosystem", ["GVR", "SIP", "PHR"]),
    ("E006", "BIDV financial ecosystem", ["BID", "BSI"]),
    ("E007", "VietinBank financial ecosystem", ["CTG", "CTS"]),
]


BUSINESS_CLUSTERS = [
    ("banking", ["ACB", "BID", "CTG", "EIB", "HDB", "LPB", "MBB", "MSB", "OCB", "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB", "ABB"]),
    ("residential_real_estate", ["AGG", "DIG", "DXG", "KDH", "NLG", "NTL", "PDR", "SCR", "VHM"]),
    ("industrial_real_estate", ["BCM", "IJC", "KBC", "SIP", "SZC"]),
    ("securities", ["AGR", "BSI", "CTS", "FTS", "HCM", "ORS", "SSI", "VCI", "VIX", "VND"]),
    ("construction_infrastructure", ["C4G", "CTD", "DPG", "FCN", "HHV", "LCG", "VCG"]),
    ("steel_materials_chemicals", ["AAA", "APH", "BMP", "CSV", "DGC", "HPG", "HSG", "NKG"]),
    ("oil_gas_power", ["BSR", "GAS", "GEG", "NT2", "PC1", "PGV", "PLX", "POW", "PPC", "PVD", "PVS"]),
    ("fertilizer_agriculture", ["BAF", "BFC", "DBC", "DCM", "DPM", "LTG", "PAN"]),
    ("seafood", ["ACL", "ANV", "FMC", "IDI", "VHC"]),
    ("retail_distribution", ["DGW", "FRT", "MWG", "PNJ"]),
    ("consumer_staples", ["KDC", "MSN", "QNS", "SAB", "VNM"]),
    ("ict_telecom", ["CMG", "ELC", "FOX", "FPT", "VGI"]),
    ("healthcare_pharma", ["DBD", "DHG", "DMC", "IMP", "JVC", "TRA"]),
    ("logistics_airport_port", ["ACV", "AST", "GMD", "HAH", "SCS", "SGN", "VJC", "VSC", "VTP"]),
    ("utilities_infrastructure_holdings", ["BWE", "GEX", "GEG", "PC1", "REE"]),
]


VALUE_CHAIN_GROUPS = [
    ("oil_gas_to_power", ["BSR", "GAS", "PLX", "PVD", "PVS"], ["GEG", "NT2", "PC1", "PGV", "POW", "PPC"]),
    ("energy_to_fertilizer", ["BSR", "GAS", "PLX"], ["BFC", "DCM", "DPM"]),
    ("steel_to_construction", ["HPG", "HSG", "NKG"], ["C4G", "CTD", "DPG", "FCN", "HHV", "LCG", "VCG"]),
    ("construction_to_real_estate", ["C4G", "CTD", "DPG", "FCN", "HHV", "LCG", "VCG"], ["AGG", "BCM", "DIG", "DXG", "IJC", "KBC", "KDH", "NLG", "NTL", "PDR", "SCR", "SIP", "SZC", "VHM"]),
    ("real_estate_to_banking", ["AGG", "BCM", "DIG", "DXG", "IJC", "KBC", "KDH", "NLG", "NTL", "PDR", "SCR", "SIP", "SZC", "VHM", "VIC", "VRE"], ["ACB", "BID", "CTG", "HDB", "MBB", "STB", "TCB", "VCB", "VPB"]),
    ("banking_to_securities", ["ACB", "BID", "CTG", "HDB", "MBB", "STB", "TCB", "VCB", "VPB"], ["AGR", "BSI", "CTS", "FTS", "HCM", "SSI", "VCI", "VIX", "VND"]),
    ("distribution_to_retail", ["DGW"], ["FRT", "MWG"]),
    ("consumer_supply_chain", ["KDC", "MSN", "QNS", "SAB", "VNM"], ["DGW", "FRT", "MWG", "PNJ"]),
    ("fertilizer_to_agriculture", ["BFC", "DCM", "DPM"], ["BAF", "DBC", "LTG", "PAN"]),
]


EDGE_COLUMNS = [
    "source_ticker",
    "target_ticker",
    "relation_type",
    "weight",
    "is_directed",
    "valid_from",
    "valid_to",
    "evidence_source",
    "note",
]


def normalize_ticker(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip().upper()


def normalize_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def active_year_bounds(row: pd.Series) -> tuple[str, str]:
    years = []
    for year in [2022, 2023, 2024, 2025]:
        if year in row.index:
            value = row.loc[year]
        elif str(year) in row.index:
            value = row.loc[str(year)]
        else:
            value = 0
        if int(value or 0) == 1:
            years.append(year)
    if not years:
        return "", ""
    valid_from = f"{min(years)}-01-01"
    valid_to = "" if max(years) >= 2025 else f"{max(years)}-12-31"
    return valid_from, valid_to


def extract_ownership_pct(text: str) -> float | None:
    # Handles forms such as 45,66% and 10%.
    matches = re.findall(r"(\d{1,3}(?:[,.]\d+)?)\s*%", text)
    if not matches:
        return None
    value = float(matches[0].replace(",", "."))
    if 0 < value <= 100:
        return value
    return None


def load_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_FILE, dtype={"ticker": "string"})
    metadata["ticker"] = metadata["ticker"].map(normalize_ticker)
    required = {"ticker", "industry"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"ticker_metadata.csv missing columns: {sorted(missing)}")
    if metadata["ticker"].nunique() != 118:
        raise ValueError(f"Expected 118 metadata tickers, found {metadata['ticker'].nunique()}")
    return metadata


def load_raw_relationships(universe: set[str]) -> pd.DataFrame:
    raw = pd.read_excel(RELATIONSHIP_FILE, sheet_name=1)
    raw = raw.rename(
        columns={
            "mã_nguồn": "source_ticker",
            "mã_đích": "target_ticker",
            "loại_quan_hệ": "raw_relation_type",
            "chiều_quan_hệ": "direction",
            "độ_tin_cậy": "confidence",
            "bằng_chứng": "evidence",
            "url_nguồn": "source_url",
        }
    )
    raw["source_ticker"] = raw["source_ticker"].map(normalize_ticker)
    raw["target_ticker"] = raw["target_ticker"].map(normalize_ticker)
    raw["in_universe"] = raw["source_ticker"].isin(universe) & raw["target_ticker"].isin(universe)
    return raw


def build_company_groups(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group_id, group_name, tickers in COMPANY_GROUP_SEEDS:
        for ticker in tickers:
            if ticker in universe:
                rows.append({"group_id": group_id, "group_name": group_name, "ticker": ticker})

    # Add verified undirected ecosystem pairs from the raw relationship file into pair groups.
    group_no = len(COMPANY_GROUP_SEEDS) + 1
    for raw_row in raw.itertuples(index=False):
        if not raw_row.in_universe:
            continue
        relation = normalize_text(raw_row.raw_relation_type).lower()
        confidence = normalize_text(raw_row.confidence).lower()
        if "chưa xác minh" in confidence:
            continue
        is_group_like = any(key in relation for key in ["cùng hệ sinh thái", "cùng nhóm", "cùng công ty mẹ", "cùng cổ đông"])
        if not is_group_like:
            continue
        existing = {
            tuple(sorted(g["ticker"] for g in rows if g["group_id"] == group_id))
            for group_id in {r["group_id"] for r in rows}
        }
        pair = tuple(sorted([raw_row.source_ticker, raw_row.target_ticker]))
        if any(set(pair).issubset(set(group)) for group in existing):
            continue
        group_id = f"G{group_no:03d}"
        group_no += 1
        group_name = f"{raw_row.source_ticker}-{raw_row.target_ticker} relationship group"
        rows.append({"group_id": group_id, "group_name": group_name, "ticker": raw_row.source_ticker})
        rows.append({"group_id": group_id, "group_name": group_name, "ticker": raw_row.target_ticker})

    groups = pd.DataFrame(rows).drop_duplicates().sort_values(["group_id", "ticker"]).reset_index(drop=True)
    return groups


def build_parent_subsidiary_raw(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    rows = []
    for _, raw_row in raw.iterrows():
        if not bool(raw_row["in_universe"]):
            continue
        relation = normalize_text(raw_row["raw_relation_type"]).lower()
        direction = normalize_text(raw_row["direction"])
        confidence = normalize_text(raw_row["confidence"]).lower()
        evidence = normalize_text(raw_row["evidence"])
        source_url = normalize_text(raw_row["source_url"])
        if "chưa xác minh" in confidence:
            continue
        if "->" not in direction:
            continue
        is_control = any(key in relation for key in ["mẹ-con", "công ty con", "sở hữu/kiểm soát"])
        if not is_control:
            continue
        parent, subsidiary = [normalize_ticker(part) for part in direction.split("->", 1)]
        if parent not in universe or subsidiary not in universe:
            continue
        ownership_pct = extract_ownership_pct(f"{evidence} {source_url}")
        valid_from, valid_to = active_year_bounds(raw_row)
        rows.append(
            {
                "parent_ticker": parent,
                "subsidiary_ticker": subsidiary,
                "ownership_pct": ownership_pct,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "evidence_source": "relationships.xlsx:mqh công ty khác",
                "source_url": source_url,
                "confidence": raw_row["confidence"],
                "raw_relation_type": raw_row["raw_relation_type"],
                "note": evidence,
            }
        )

    return pd.DataFrame(rows)


def build_parent_subsidiary_edges(parent_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_row in parent_raw.itertuples(index=False):
        ownership = None if pd.isna(raw_row.ownership_pct) else float(raw_row.ownership_pct) / 100
        parent_weight = round(ownership if ownership is not None else DEFAULT_PARENT_TO_SUB_WEIGHT, 6)
        child_weight = round((ownership * DEFAULT_SUB_TO_PARENT_WEIGHT) if ownership is not None else DEFAULT_SUB_TO_PARENT_WEIGHT, 6)
        evidence_source = raw_row.evidence_source
        if normalize_text(raw_row.source_url):
            evidence_source = f"{evidence_source}; {raw_row.source_url}"
        weight_note = (
            f"ownership_pct={raw_row.ownership_pct}; weight uses ownership_pct where available"
            if ownership is not None
            else "research-defined default weights because ownership_pct is not available"
        )
        base_note = f"{weight_note}; confidence={raw_row.confidence}; {raw_row.note}"
        rows.extend(
            [
                {
                    "source_ticker": raw_row.parent_ticker,
                    "target_ticker": raw_row.subsidiary_ticker,
                    "relation_type": "parent_to_subsidiary",
                    "weight": parent_weight,
                    "is_directed": 1,
                    "valid_from": raw_row.valid_from,
                    "valid_to": raw_row.valid_to,
                    "evidence_source": evidence_source,
                    "note": base_note,
                },
                {
                    "source_ticker": raw_row.subsidiary_ticker,
                    "target_ticker": raw_row.parent_ticker,
                    "relation_type": "subsidiary_to_parent",
                    "weight": child_weight,
                    "is_directed": 1,
                    "valid_from": raw_row.valid_from,
                    "valid_to": raw_row.valid_to,
                    "evidence_source": evidence_source,
                    "note": base_note,
                },
            ]
        )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def build_same_group_edges(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group_id, group_name), group_df in groups.groupby(["group_id", "group_name"]):
        tickers = sorted(group_df["ticker"].unique())
        if len(tickers) < 2:
            continue
        for source, target in itertools.permutations(tickers, 2):
            rows.append(
                {
                    "source_ticker": source,
                    "target_ticker": target,
                    "relation_type": "same_group",
                    "weight": DEFAULT_SAME_GROUP_WEIGHT,
                    "is_directed": 0,
                    "valid_from": "",
                    "valid_to": "",
                    "evidence_source": "company_groups.csv; relationships.xlsx:mqh công ty khác",
                    "note": f"research-defined same-group edge from {group_id} ({group_name}); weight is not an ownership percentage",
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS).drop_duplicates(["source_ticker", "target_ticker", "relation_type"])


def build_same_industry_edges(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = metadata[metadata["industry"].fillna("").ne("")].copy()
    for industry, group_df in clean.groupby("industry"):
        tickers = sorted(group_df["ticker"].unique())
        for source, target in itertools.permutations(tickers, 2):
            rows.append(
                {
                    "source_ticker": source,
                    "target_ticker": target,
                    "relation_type": "same_industry",
                    "weight": DEFAULT_SAME_INDUSTRY_WEIGHT,
                    "is_directed": 0,
                    "valid_from": "",
                    "valid_to": "",
                    "evidence_source": "ticker_metadata.csv",
                    "note": f"same industry={industry}; research-defined static weight",
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def add_edge_pair(
    rows: list[dict[str, object]],
    source: str,
    target: str,
    relation_type: str,
    weight: float,
    evidence_source: str,
    note: str,
    valid_from: str = "",
    valid_to: str = "",
) -> None:
    if source == target:
        return
    rows.append(
        {
            "source_ticker": source,
            "target_ticker": target,
            "relation_type": relation_type,
            "weight": weight,
            "is_directed": 0,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "evidence_source": evidence_source,
            "note": note,
        }
    )


def build_curated_multilayer_edges(universe: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for item in STRATEGIC_ECOSYSTEM_GROUPS:
        group_id, group_name, tickers = item[:3]
        valid_from = item[3] if len(item) > 3 else ""
        valid_to = item[4] if len(item) > 4 else ""
        active_tickers = [ticker for ticker in tickers if ticker in universe]
        if len(active_tickers) > 8:
            hubs = active_tickers[:3]
            pair_candidates = list(itertools.permutations(hubs, 2))
            for hub in hubs:
                for ticker in active_tickers[3:]:
                    pair_candidates.extend([(hub, ticker), (ticker, hub)])
        else:
            pair_candidates = list(itertools.permutations(active_tickers, 2))
        for source, target in pair_candidates:
            add_edge_pair(
                rows,
                source,
                target,
                "strategic_ecosystem",
                DEFAULT_STRATEGIC_ECOSYSTEM_WEIGHT,
                "curated thesis relationship map",
                f"{group_id} {group_name}; strong ecosystem or ownership-adjacent relationship; weight is research-defined",
                valid_from,
                valid_to,
            )

    for cluster_name, tickers in BUSINESS_CLUSTERS:
        active_tickers = sorted(ticker for ticker in tickers if ticker in universe)
        for source, target in itertools.permutations(active_tickers, 2):
            add_edge_pair(
                rows,
                source,
                target,
                "business_cluster",
                DEFAULT_BUSINESS_CLUSTER_WEIGHT,
                "curated thesis relationship map",
                f"same business cluster={cluster_name}; narrower than generic same_industry",
            )

    for chain_name, upstream, downstream in VALUE_CHAIN_GROUPS:
        upstream_tickers = [ticker for ticker in upstream if ticker in universe][:6]
        downstream_tickers = [ticker for ticker in downstream if ticker in universe][:6]
        for source in upstream_tickers:
            for target in downstream_tickers:
                add_edge_pair(
                    rows,
                    source,
                    target,
                    "value_chain",
                    DEFAULT_VALUE_CHAIN_WEIGHT,
                    "curated thesis relationship map",
                    f"value-chain relation={chain_name}; forward economic exposure",
                )
                add_edge_pair(
                    rows,
                    target,
                    source,
                    "value_chain",
                    round(DEFAULT_VALUE_CHAIN_WEIGHT * 0.85, 6),
                    "curated thesis relationship map",
                    f"value-chain relation={chain_name}; reverse risk feedback exposure",
                )

    return pd.DataFrame(rows, columns=EDGE_COLUMNS).drop_duplicates(["source_ticker", "target_ticker", "relation_type"])


def build_common_owner_edges(universe: set[str]) -> pd.DataFrame:
    if not OWNERSHIP_GRAPH_FILE.exists():
        return pd.DataFrame(columns=EDGE_COLUMNS)

    graph_weights = pd.read_excel(OWNERSHIP_GRAPH_FILE, sheet_name="03_Graph_Weights")
    rows: list[dict[str, object]] = []
    for raw_row in graph_weights.itertuples(index=False):
        ticker_i = normalize_ticker(raw_row.ticker_i)
        ticker_j = normalize_ticker(raw_row.ticker_j)
        if ticker_i not in universe or ticker_j not in universe or ticker_i == ticker_j:
            continue
        weight = float(raw_row.graph_weight)
        common_owner = normalize_text(raw_row.common_owner)
        note = (
            f"common_owner={common_owner}; owner_pct_i={raw_row.owner_pct_i}; "
            f"owner_pct_j={raw_row.owner_pct_j}; method={raw_row.method}; {raw_row.formula_note}"
        )
        for source, target in [(ticker_i, ticker_j), (ticker_j, ticker_i)]:
            add_edge_pair(
                rows,
                source,
                target,
                "common_owner",
                weight,
                "data/data_2/vietnam_ownership_graph_dataset_v1.xlsx:03_Graph_Weights",
                note,
            )

    return pd.DataFrame(rows, columns=EDGE_COLUMNS).drop_duplicates(["source_ticker", "target_ticker", "relation_type"])


def build_quality_report(
    metadata: pd.DataFrame,
    raw: pd.DataFrame,
    groups: pd.DataFrame,
    parent_raw: pd.DataFrame,
    parent_edges: pd.DataFrame,
    same_group_edges: pd.DataFrame,
    same_industry_edges: pd.DataFrame,
    curated_edges: pd.DataFrame,
    common_owner_edges: pd.DataFrame,
    company_relationships: pd.DataFrame,
) -> pd.DataFrame:
    excluded_not_universe = int((~raw["in_universe"]).sum())
    raw_unverified = int(raw["confidence"].fillna("").astype(str).str.lower().str.contains("chưa xác minh").sum())
    duplicate_edges = int(company_relationships.duplicated(["source_ticker", "target_ticker", "relation_type"]).sum())
    rows = [
        ("metadata_ticker_count", metadata["ticker"].nunique(), "Expected 118 tickers"),
        ("metadata_missing_industry", int(metadata["industry"].fillna("").eq("").sum()), "Must be 0 for same_industry generation"),
        ("raw_relationship_rows", len(raw), "Rows in relationships.xlsx sheet mqh công ty khác"),
        ("raw_relationship_rows_excluded_not_in_universe", excluded_not_universe, "Rows where source or target is not in 118 tickers"),
        ("raw_relationship_rows_unverified", raw_unverified, "Rows marked Chưa xác minh are excluded from group/control edges"),
        ("company_group_rows", len(groups), "Rows in company_groups.csv"),
        ("company_group_count", groups["group_id"].nunique(), "Groups used to generate same_group edges"),
        ("parent_subsidiary_raw_rows", len(parent_raw), "Verified control/parent-subsidiary pairs"),
        ("parent_subsidiary_edges", len(parent_edges), "Includes reverse subsidiary_to_parent edges"),
        ("same_group_edges", len(same_group_edges), "Directed rows for GNN message passing; is_directed=0 indicates symmetric relation"),
        ("same_industry_edges", len(same_industry_edges), "Directed rows for each same-industry pair"),
        ("curated_multilayer_edges", len(curated_edges), "Strategic ecosystem, business cluster, and value-chain edges"),
        ("common_owner_edges", len(common_owner_edges), "Ticker-pair edges from verified common-owner graph weights"),
        ("company_relationships_rows", len(company_relationships), "Edges used by GNN: parent/subsidiary, same_group, curated multilayer, common-owner edges"),
        ("duplicate_relationship_rows", duplicate_edges, "Duplicates by source,target,relation_type after merge"),
        ("parent_default_weight_rows", int(parent_raw["ownership_pct"].isna().sum()), "Rows using research-defined default parent/sub weights"),
        ("parent_ownership_weight_rows", int(parent_raw["ownership_pct"].notna().sum()), "Rows using extracted ownership_pct"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    universe = set(metadata["ticker"])
    raw = load_raw_relationships(universe)

    groups = build_company_groups(raw, universe)
    parent_raw = build_parent_subsidiary_raw(raw, universe)
    parent_edges = build_parent_subsidiary_edges(parent_raw)
    same_group_edges = build_same_group_edges(groups)
    same_industry_edges = build_same_industry_edges(metadata)
    curated_edges = build_curated_multilayer_edges(universe)
    common_owner_edges = build_common_owner_edges(universe)
    company_relationships = pd.concat(
        [parent_edges, same_group_edges, curated_edges, common_owner_edges],
        ignore_index=True,
    ).drop_duplicates(["source_ticker", "target_ticker", "relation_type"])
    company_relationships = company_relationships.sort_values(
        ["relation_type", "source_ticker", "target_ticker"]
    ).reset_index(drop=True)
    quality_report = build_quality_report(
        metadata,
        raw,
        groups,
        parent_raw,
        parent_edges,
        same_group_edges,
        same_industry_edges,
        curated_edges,
        common_owner_edges,
        company_relationships,
    )

    write_csv(groups, "company_groups.csv")
    write_csv(parent_raw, "parent_subsidiary_raw.csv")
    write_csv(parent_edges, "parent_subsidiary_edges.csv")
    write_csv(same_group_edges, "same_group_edges.csv")
    write_csv(same_industry_edges, "same_industry_edges.csv")
    write_csv(curated_edges, "curated_multilayer_edges.csv")
    write_csv(common_owner_edges, "common_owner_edges.csv")
    write_csv(company_relationships, "company_relationships.csv")
    write_csv(quality_report, "relationship_quality_report.csv")

    print("Generated relationship outputs in data/ and data/processed/")
    print(f"Parent/subsidiary edges: {len(parent_edges)}")
    print(f"Same-group edges: {len(same_group_edges)}")
    print(f"Same-industry edges: {len(same_industry_edges)}")
    print(f"Curated multilayer edges: {len(curated_edges)}")
    print(f"Common-owner edges: {len(common_owner_edges)}")
    print(f"Company relationships: {len(company_relationships)}")


if __name__ == "__main__":
    main()
