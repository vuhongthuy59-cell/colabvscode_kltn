from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "08_tuned_gnn"

SNAPSHOT_FILE = ROOT / "outputs" / "05_event_graph_dataset" / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = ROOT / "outputs" / "05_event_graph_dataset" / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = ROOT / "outputs" / "05_event_graph_dataset" / "edge_type_map.json"
FEATURE_SCHEMA_FILE = ROOT / "outputs" / "05_event_graph_dataset" / "node_feature_schema.json"
RF_IMPORTANCE_FILE = ROOT / "outputs" / "06_baseline_models" / "rf_feature_importance.csv"

RANDOM_STATE = 42
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE = 10
TOP_K = 20
TOP_RF_FEATURES = 30


class TunedGraphMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
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


def load_top_rf_feature_cols() -> list[int]:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)
    name_to_idx = {name: idx for idx, name in enumerate(schema["feature_names"])}
    importance = pd.read_csv(RF_IMPORTANCE_FILE)
    selected = [name for name in importance["feature"].astype(str).tolist() if name in name_to_idx]
    return [name_to_idx[name] for name in selected[:TOP_RF_FEATURES]]


def aggregate_topk_corr_for_targets(
    snapshot: dict,
    target_nodes: list[int],
    cols: list[int],
    corr_type_id: int,
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
    rows = []
    selected_count = 0

    for node_idx in target_nodes:
        incoming = corr_edges[1].eq(int(node_idx))
        if int(incoming.sum()) == 0:
            rows.append(np.zeros(len(cols), dtype=np.float32))
            continue
        src = corr_edges[0, incoming]
        weights = corr_weights[incoming]
        keep = min(TOP_K, int(weights.numel()))
        top_pos = torch.topk(weights, k=keep).indices
        src = src[top_pos]
        weights = weights[top_pos]
        selected_count += keep
        denom = weights.sum().clamp_min(1e-8)
        agg = (x[src] * weights.unsqueeze(1)).sum(dim=0) / denom
        rows.append(agg.numpy().astype(np.float32))

    return np.vstack(rows).astype(np.float32), selected_count


def build_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    cols: list[int],
    corr_type_id: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    valid_snapshot_ids = set(index["snapshot_id"].astype(int))
    rows = []
    x_rows = []
    y_rows = []
    selected_edges = []

    for snapshot_id, snapshot in enumerate(snapshots):
        if snapshot_id not in valid_snapshot_ids:
            continue
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        if not target_nodes:
            continue
        neighbor_features, selected_count = aggregate_topk_corr_for_targets(snapshot, target_nodes, cols, corr_type_id)
        selected_edges.append(selected_count / max(len(target_nodes), 1))
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
            neighbor = neighbor_features[row_pos]
            interaction_features = np.concatenate(
                [
                    node_features,
                    neighbor,
                    node_features - neighbor,
                    node_features * neighbor,
                ]
            )
            x_rows.append(interaction_features.astype(np.float32))
            y_rows.append(float(snapshot["y"][node_idx]))

    return (
        pd.DataFrame(rows),
        np.vstack(x_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        float(np.mean(selected_edges)) if selected_edges else 0.0,
    )


def train_variant(
    variant: dict,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_val = x_scaler.transform(x[val_mask]).astype(np.float32)
    x_test = x_scaler.transform(x[test_mask]).astype(np.float32)
    y_train = y_scaler.fit_transform(y[train_mask].reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_raw = y[val_mask]

    model = TunedGraphMLP(input_dim=x.shape[1], hidden_dim=variant["hidden_dim"], dropout=variant["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant["lr"], weight_decay=variant["weight_decay"])
    loss_fn = torch.nn.SmoothL1Loss(beta=variant["huber_beta"])

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    best_state = None
    best_val_mae = float("inf")
    stale_epochs = 0
    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            pred = model(x_train_t[batch])
            loss = loss_fn(pred, y_train_t[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_scaled = model(x_val_t).numpy()
        val_pred = y_scaler.inverse_transform(val_scaled.reshape(-1, 1)).reshape(-1)
        val_mae, val_rmse = metrics(y_val_raw, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        log_rows.append(
            {
                "variant": variant["name"],
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        if stale_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_scaled = model(x_val_t).numpy()
        test_scaled = model(x_test_t).numpy()
    preds = {
        "validation": y_scaler.inverse_transform(val_scaled.reshape(-1, 1)).reshape(-1),
        "test": y_scaler.inverse_transform(test_scaled.reshape(-1, 1)).reshape(-1),
    }
    return pd.DataFrame(log_rows), preds


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        corr_type_id = int(json.load(f)["price_correlation"])

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)
    cols = load_top_rf_feature_cols()

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    samples, x, y, mean_edges = build_samples(snapshots, index, cols, corr_type_id)
    samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
    samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
    samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
    train_mask = samples["split"].eq("train").to_numpy()
    val_mask = samples["split"].eq("validation").to_numpy()
    test_mask = samples["split"].eq("test").to_numpy()

    variants = [
        {"name": "scaled_h64_d010", "hidden_dim": 64, "dropout": 0.10, "lr": 8e-4, "weight_decay": 1e-4, "huber_beta": 0.50},
        {"name": "scaled_h96_d015", "hidden_dim": 96, "dropout": 0.15, "lr": 6e-4, "weight_decay": 2e-4, "huber_beta": 0.50},
        {"name": "scaled_h128_d020", "hidden_dim": 128, "dropout": 0.20, "lr": 5e-4, "weight_decay": 3e-4, "huber_beta": 0.50},
        {"name": "scaled_h64_d000", "hidden_dim": 64, "dropout": 0.00, "lr": 8e-4, "weight_decay": 1e-4, "huber_beta": 0.50},
    ]

    result_rows = []
    logs = []
    for variant in variants:
        print(f"Training {variant['name']} ...")
        log_df, preds = train_variant(variant, x, y, train_mask, val_mask, test_mask)
        logs.append(log_df)
        for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
            mae, rmse = metrics(y[mask], preds[split_name])
            result_rows.append(
                {
                    "model": "Tuned TopK Graph MLP",
                    "variant": variant["name"],
                    "split": split_name,
                    "feature_set": f"top_{TOP_RF_FEATURES}_rf",
                    "feature_count": len(cols),
                    "input_dim": int(x.shape[1]),
                    "top_k_corr_neighbors": TOP_K,
                    "mean_topk_edges_per_target": mean_edges,
                    "hidden_dim": variant["hidden_dim"],
                    "dropout": variant["dropout"],
                    "lr": variant["lr"],
                    "weight_decay": variant["weight_decay"],
                    "mae": mae,
                    "rmse": rmse,
                    "note": "StandardScaler on X/y, SmoothL1 loss, AdamW, early stopping on validation MAE, self-neighbor residual/product interaction features",
                }
            )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "variant"]).reset_index(drop=True)
    pd.concat(logs, ignore_index=True).to_csv(OUT_DIR / "tuned_gnn_training_log.csv", index=False)
    results.to_csv(OUT_DIR / "tuned_gnn_results.csv", index=False)
    print("Generated tuned GNN results:")
    print((OUT_DIR / "tuned_gnn_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
