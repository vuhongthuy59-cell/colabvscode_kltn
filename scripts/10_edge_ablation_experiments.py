from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "10_edge_ablation_experiments"

SNAPSHOT_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "edge_type_map.json"
FEATURE_SCHEMA_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "node_feature_schema.json"

RANDOM_STATE = 42
EPOCHS = 20
BATCH_SIZE = 512
HIDDEN_DIM = 48
LR = 1e-3


class OneHopMLP(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred, squared=False)),
    )


def split_snapshot_ids(index: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    ordered = index.sort_values(["event_trading_date", "snapshot_id"]).reset_index(drop=True)
    train_end = int(len(ordered) * 0.70)
    val_end = int(len(ordered) * 0.85)
    return (
        set(ordered.iloc[:train_end]["snapshot_id"].astype(int)),
        set(ordered.iloc[train_end:val_end]["snapshot_id"].astype(int)),
        set(ordered.iloc[val_end:]["snapshot_id"].astype(int)),
    )


def load_feature_columns() -> list[int]:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return list(range(len(schema["feature_names"])))


def selected_edge_mask(
    snapshot: dict,
    edge_type_ids: set[int],
    corr_type_id: int,
    min_corr_weight: float | None,
) -> torch.Tensor:
    edge_type = snapshot["edge_type"].long()
    edge_weight = snapshot["edge_weight"].float()
    selected = torch.zeros(edge_type.shape[0], dtype=torch.bool)
    for type_id in edge_type_ids:
        selected |= edge_type.eq(int(type_id))
    if min_corr_weight is not None and corr_type_id in edge_type_ids:
        corr_too_low = edge_type.eq(corr_type_id) & edge_weight.lt(float(min_corr_weight))
        selected &= ~corr_too_low
    return selected


def aggregate_for_targets(
    snapshot: dict,
    target_nodes: list[int],
    cols: list[int],
    edge_type_ids: set[int],
    corr_type_id: int,
    min_corr_weight: float | None,
) -> tuple[np.ndarray, int]:
    x = snapshot["x"][:, cols].float()
    edge_index = snapshot["edge_index"].long()
    edge_weight = snapshot["edge_weight"].float()
    selected = selected_edge_mask(snapshot, edge_type_ids, corr_type_id, min_corr_weight)
    edge_count = int(selected.sum())
    if edge_count == 0:
        return np.zeros((len(target_nodes), len(cols)), dtype=np.float32), 0

    selected_edges = edge_index[:, selected]
    selected_weights = edge_weight[selected]
    result = []
    for node_idx in target_nodes:
        incoming = selected_edges[1].eq(int(node_idx))
        if int(incoming.sum()) == 0:
            result.append(np.zeros(len(cols), dtype=np.float32))
            continue
        src = selected_edges[0, incoming]
        weights = selected_weights[incoming]
        denom = weights.sum().clamp_min(1e-8)
        agg = (x[src] * weights.unsqueeze(1)).sum(dim=0) / denom
        result.append(agg.numpy().astype(np.float32))
    return np.vstack(result).astype(np.float32), edge_count


def build_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    cols: list[int],
    edge_type_ids: set[int],
    corr_type_id: int,
    min_corr_weight: float | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    valid_snapshot_ids = set(index["snapshot_id"].astype(int))
    rows = []
    graph_rows = []
    y_rows = []
    edge_counts = []

    for snapshot_id, snapshot in enumerate(snapshots):
        if snapshot_id not in valid_snapshot_ids:
            continue
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        if not target_nodes:
            continue
        neighbor_features, edge_count = aggregate_for_targets(
            snapshot,
            target_nodes,
            cols,
            edge_type_ids,
            corr_type_id,
            min_corr_weight,
        )
        edge_counts.append(edge_count)
        for row_pos, node_idx in enumerate(target_nodes):
            node_features = snapshot["x"][node_idx, cols].numpy().astype(np.float32)
            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "article_id": snapshot["article_id"],
                    "event_trading_date": snapshot["event_trading_date"],
                    "node_idx": int(node_idx),
                    "split": "",
                    "y_true": float(snapshot["y"][node_idx]),
                }
            )
            graph_rows.append(np.concatenate([node_features, neighbor_features[row_pos]]).astype(np.float32))
            y_rows.append(float(snapshot["y"][node_idx]))

    return (
        pd.DataFrame(rows),
        np.vstack(graph_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        float(np.mean(edge_counts)) if edge_counts else 0.0,
    )


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    model = OneHopMLP(input_dim=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    x_train = torch.tensor(x[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    x_val = torch.tensor(x[val_mask], dtype=torch.float32)
    y_val = y[val_mask]
    x_test = torch.tensor(x[test_mask], dtype=torch.float32)

    best_state = None
    best_val_mae = float("inf")
    stale_epochs = 0
    for _ in range(EPOCHS):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            pred = model(x_train[batch])
            loss = loss_fn(pred, y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).numpy()
        val_mae, _ = metrics(y_val, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= 5:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return {
            "validation": model(torch.tensor(x[val_mask], dtype=torch.float32)).numpy(),
            "test": model(x_test).numpy(),
        }


def build_edge_configs(edge_type_map: dict[str, int]) -> dict[str, dict]:
    corr = edge_type_map["price_correlation"]
    parent = edge_type_map["parent_to_subsidiary"]
    child = edge_type_map["subsidiary_to_parent"]
    group = edge_type_map["same_group"]
    industry = edge_type_map["same_industry"]
    news = edge_type_map["news_co_mention"]
    all_types = set(edge_type_map.values())
    return {
        "corr_threshold_0_15": {"edge_types": {corr}, "min_corr_weight": 0.15},
        "corr_threshold_0_20": {"edge_types": {corr}, "min_corr_weight": 0.20},
        "corr_threshold_0_30": {"edge_types": {corr}, "min_corr_weight": 0.30},
        "corr_plus_same_industry": {"edge_types": {corr, industry}, "min_corr_weight": 0.15},
        "corr_plus_same_group": {"edge_types": {corr, group}, "min_corr_weight": 0.15},
        "corr_plus_parent_subsidiary": {"edge_types": {corr, parent, child}, "min_corr_weight": 0.15},
        "corr_plus_news_co_mention": {"edge_types": {corr, news}, "min_corr_weight": 0.15},
        "full_graph": {"edge_types": all_types, "min_corr_weight": 0.15},
    }


def main() -> None:
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)
    corr_type_id = int(edge_type_map["price_correlation"])
    feature_cols = load_feature_columns()

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    result_rows = []
    for experiment, config in build_edge_configs(edge_type_map).items():
        print(f"Running {experiment}")
        samples, x_graph, y, mean_edges = build_samples(
            snapshots,
            index,
            feature_cols,
            config["edge_types"],
            corr_type_id,
            config["min_corr_weight"],
        )
        samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
        samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
        samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
        train_mask = samples["split"].eq("train").to_numpy()
        val_mask = samples["split"].eq("validation").to_numpy()
        test_mask = samples["split"].eq("test").to_numpy()

        preds = train_model(x_graph, y, train_mask, val_mask, test_mask)
        for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
            mae, rmse = metrics(y[mask], preds[split_name])
            result_rows.append(
                {
                    "experiment": experiment,
                    "split": split_name,
                    "feature_count": len(feature_cols),
                    "input_dim": int(x_graph.shape[1]),
                    "edge_types": ",".join(str(v) for v in sorted(config["edge_types"])),
                    "min_corr_weight": config["min_corr_weight"],
                    "mean_selected_edges_per_snapshot": mean_edges,
                    "mae": mae,
                    "rmse": rmse,
                    "note": "thresholds below 0.15 require rebuilding graph snapshots because current snapshots were created with CORR_THRESHOLD=0.15",
                }
            )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "experiment"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "edge_ablation_results.csv", index=False)

    print("Generated edge ablation outputs:")
    print((OUT_DIR / "edge_ablation_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
