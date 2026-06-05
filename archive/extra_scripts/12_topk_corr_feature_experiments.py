from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "12_topk_corr_feature_experiments"

SNAPSHOT_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "edge_type_map.json"
FEATURE_SCHEMA_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "node_feature_schema.json"
RF_IMPORTANCE_FILE = ROOT / "outputs" / "05_train_baselines" / "rf_feature_importance.csv"

RANDOM_STATE = 42
EPOCHS = 16
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
    unique_dates = ordered["event_trading_date"].drop_duplicates().reset_index(drop=True)
    train_end = int(len(unique_dates) * 0.70)
    val_end = int(len(unique_dates) * 0.85)
    train_dates = set(unique_dates.iloc[:train_end])
    val_dates = set(unique_dates.iloc[train_end:val_end])
    test_dates = set(unique_dates.iloc[val_end:])
    return (
        set(ordered.loc[ordered["event_trading_date"].isin(train_dates), "snapshot_id"].astype(int)),
        set(ordered.loc[ordered["event_trading_date"].isin(val_dates), "snapshot_id"].astype(int)),
        set(ordered.loc[ordered["event_trading_date"].isin(test_dates), "snapshot_id"].astype(int)),
    )


def load_schema() -> dict:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_indices(schema: dict, names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(schema["feature_names"])}
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown feature names: {missing}")
    return [name_to_idx[name] for name in names]


def top_rf_features(schema: dict, n: int) -> list[str]:
    if not RF_IMPORTANCE_FILE.exists():
        return schema["feature_names"][:n]
    importance = pd.read_csv(RF_IMPORTANCE_FILE)
    ranked = importance["feature"].astype(str).tolist()
    known = set(schema["feature_names"])
    return [name for name in ranked if name in known][:n]


def build_feature_configs(schema: dict) -> dict[str, list[int]]:
    configs = {
        "price_only": schema["price_feature_names"],
        "price_micro_macro": schema["price_feature_names"] + schema["micro_feature_names"] + schema["macro_feature_names"],
        "top_30_rf": top_rf_features(schema, 30),
        "full_59": schema["feature_names"],
    }
    return {name: feature_indices(schema, features) for name, features in configs.items()}


def aggregate_topk_corr_for_targets(
    snapshot: dict,
    target_nodes: list[int],
    cols: list[int],
    corr_type_id: int,
    top_k: int,
) -> tuple[np.ndarray, int]:
    x = snapshot["x"][:, cols].float()
    edge_type = snapshot["edge_type"].long()
    edge_index = snapshot["edge_index"].long()
    edge_weight = snapshot["edge_weight"].float()

    corr_mask = edge_type.eq(int(corr_type_id))
    if int(corr_mask.sum()) == 0:
        return np.zeros((len(target_nodes), len(cols)), dtype=np.float32), 0

    corr_edges = edge_index[:, corr_mask]
    corr_weights = edge_weight[corr_mask]
    result = []
    selected_edge_count = 0

    for node_idx in target_nodes:
        incoming = corr_edges[1].eq(int(node_idx))
        if int(incoming.sum()) == 0:
            result.append(np.zeros(len(cols), dtype=np.float32))
            continue
        src = corr_edges[0, incoming]
        weights = corr_weights[incoming]
        keep = min(int(top_k), int(weights.numel()))
        top_pos = torch.topk(weights, k=keep).indices
        src = src[top_pos]
        weights = weights[top_pos]
        selected_edge_count += keep
        denom = weights.sum().clamp_min(1e-8)
        agg = (x[src] * weights.unsqueeze(1)).sum(dim=0) / denom
        result.append(agg.numpy().astype(np.float32))

    return np.vstack(result).astype(np.float32), selected_edge_count


def build_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    cols: list[int],
    corr_type_id: int,
    top_k: int,
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
        neighbor_features, selected_edges = aggregate_topk_corr_for_targets(
            snapshot, target_nodes, cols, corr_type_id, top_k
        )
        edge_counts.append(selected_edges / max(len(target_nodes), 1))
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    feature_configs = build_feature_configs(schema)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)
    corr_type_id = int(edge_type_map["price_correlation"])

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    result_rows = []
    top_k_values = [5, 10, 20, 40]
    for feature_name, cols in feature_configs.items():
        for top_k in top_k_values:
            experiment = f"{feature_name}_corr_top_{top_k}"
            print(f"Running {experiment}")
            samples, x_graph, y, mean_topk_edges = build_samples(snapshots, index, cols, corr_type_id, top_k)
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
                        "feature_set": feature_name,
                        "feature_count": len(cols),
                        "input_dim": int(x_graph.shape[1]),
                        "top_k_corr_neighbors": top_k,
                        "mean_topk_edges_per_target": mean_topk_edges,
                        "split": split_name,
                        "mae": mae,
                        "rmse": rmse,
                        "note": "top-k is selected from correlation edges already present in graph snapshots",
                    }
                )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "experiment"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "topk_corr_feature_results.csv", index=False)
    print("Generated top-k correlation feature results:")
    print((OUT_DIR / "topk_corr_feature_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
