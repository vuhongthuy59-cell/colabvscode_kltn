from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

SNAPSHOT_FILE = OUT_DIR / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = OUT_DIR / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = OUT_DIR / "edge_type_map.json"
FEATURE_SCHEMA_FILE = OUT_DIR / "node_feature_schema.json"

RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 512
HIDDEN_DIM = 64
LR = 1e-3


class OneHopGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_snapshot_ids(index: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    ordered = index.sort_values(["event_trading_date", "snapshot_id"]).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return (
        set(ordered.iloc[:train_end]["snapshot_id"].astype(int)),
        set(ordered.iloc[train_end:val_end]["snapshot_id"].astype(int)),
        set(ordered.iloc[val_end:]["snapshot_id"].astype(int)),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred, squared=False)),
    )


def aggregate_neighbors(snapshot: dict, feature_cols: np.ndarray, edge_type_ids: set[int]) -> np.ndarray:
    x = snapshot["x"][:, feature_cols].float()
    num_nodes, num_features = x.shape
    edge_type = snapshot["edge_type"].long()
    edge_index = snapshot["edge_index"].long()
    edge_weight = snapshot["edge_weight"].float()

    selected = torch.zeros(edge_type.shape[0], dtype=torch.bool)
    for type_id in edge_type_ids:
        selected |= edge_type.eq(int(type_id))
    if int(selected.sum()) == 0:
        return np.zeros((num_nodes, num_features), dtype=np.float32)

    selected_edges = edge_index[:, selected]
    selected_weights = edge_weight[selected]
    src = selected_edges[0]
    dst = selected_edges[1]

    weighted_messages = x[src] * selected_weights.unsqueeze(1)
    agg = torch.zeros((num_nodes, num_features), dtype=torch.float32)
    denom = torch.zeros((num_nodes,), dtype=torch.float32)
    agg.index_add_(0, dst, weighted_messages)
    denom.index_add_(0, dst, selected_weights)
    agg = agg / denom.clamp_min(1e-8).unsqueeze(1)
    agg[denom.eq(0)] = 0
    return agg.numpy().astype(np.float32)


def build_model_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    model_configs: dict[str, dict],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    index_by_snapshot = index.set_index("snapshot_id").to_dict(orient="index")
    rows = []
    y_values = []
    model_features = {name: [] for name in model_configs}

    for snapshot_id, snapshot in enumerate(snapshots):
        meta = index_by_snapshot.get(snapshot_id)
        if meta is None:
            continue
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        if not target_nodes:
            continue

        agg_by_model = {}
        for model_name, config in model_configs.items():
            feature_cols = config["feature_cols"]
            edge_types = config["edge_types"]
            agg_by_model[model_name] = aggregate_neighbors(snapshot, feature_cols, edge_types)

        for node_idx in target_nodes:
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
            y_values.append(float(snapshot["y"][node_idx]))
            for model_name, config in model_configs.items():
                feature_cols = config["feature_cols"]
                node_features = snapshot["x"][node_idx, feature_cols].numpy().astype(np.float32)
                neighbor_features = agg_by_model[model_name][node_idx]
                model_features[model_name].append(np.concatenate([node_features, neighbor_features]).astype(np.float32))

    samples = pd.DataFrame(rows)
    x_by_model = {name: np.vstack(values).astype(np.float32) for name, values in model_features.items()}
    return samples, x_by_model, np.asarray(y_values, dtype=np.float32)


def train_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[OneHopGNN, pd.DataFrame, dict[str, np.ndarray]]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    model = OneHopGNN(input_dim=x.shape[1], hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    x_train = torch.tensor(x[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    x_val = torch.tensor(x[val_mask], dtype=torch.float32)
    y_val = y[val_mask]
    x_test = torch.tensor(x[test_mask], dtype=torch.float32)

    log_rows = []
    best_state = None
    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            pred = model(x_train[batch])
            loss = loss_fn(pred, y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train).numpy()
            val_pred = model(x_val).numpy()
        train_mae, train_rmse = metrics(y[train_mask], train_pred)
        val_mae, val_rmse = metrics(y_val, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        log_rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predictions = {
            "validation": model(torch.tensor(x[val_mask], dtype=torch.float32)).numpy(),
            "test": model(x_test).numpy(),
        }
    return model, pd.DataFrame(log_rows), predictions


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    feature_dim = int(snapshots[0]["x"].shape[1])
    if FEATURE_SCHEMA_FILE.exists():
        with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
            non_news_feature_count = int(json.load(f)["non_news_feature_count"])
    else:
        non_news_feature_count = 22
    price_cols = np.arange(0, non_news_feature_count)
    news_cols = np.arange(0, feature_dim - 1)
    full_cols = np.arange(0, feature_dim)
    relationship_edge_types = {
        edge_type_map["price_correlation"],
        edge_type_map["parent_to_subsidiary"],
        edge_type_map["subsidiary_to_parent"],
        edge_type_map["same_group"],
        edge_type_map["same_industry"],
    }
    all_edge_types = set(edge_type_map.values())

    model_configs = {
        "GNN Correlation Only": {
            "feature_cols": price_cols,
            "edge_types": {edge_type_map["price_correlation"]},
            "price_features": "Yes",
            "news_features": "No",
            "relationship_edges": "No",
            "co_mention_edges": "No",
            "checkpoint": "gnn_corr_ablation_model.pt",
        },
        "GNN + News": {
            "feature_cols": news_cols,
            "edge_types": {edge_type_map["price_correlation"]},
            "price_features": "Yes",
            "news_features": "Yes",
            "relationship_edges": "No",
            "co_mention_edges": "No",
            "checkpoint": "gnn_news_model.pt",
        },
        "GNN + Relationship": {
            "feature_cols": price_cols,
            "edge_types": relationship_edge_types,
            "price_features": "Yes",
            "news_features": "No",
            "relationship_edges": "Yes",
            "co_mention_edges": "No",
            "checkpoint": "gnn_relationship_model.pt",
        },
        "Full Model": {
            "feature_cols": full_cols,
            "edge_types": all_edge_types,
            "price_features": "Yes",
            "news_features": "Yes",
            "relationship_edges": "Yes",
            "co_mention_edges": "Yes",
            "checkpoint": "full_gnn_model.pt",
        },
    }

    samples, x_by_model, y = build_model_samples(snapshots, index, model_configs)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)
    samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
    samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
    samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"

    train_mask = samples["split"].eq("train").to_numpy()
    val_mask = samples["split"].eq("validation").to_numpy()
    test_mask = samples["split"].eq("test").to_numpy()

    logs = []
    result_rows = []
    prediction_frames = []

    for model_name, config in model_configs.items():
        print(f"Training {model_name} ...")
        x = x_by_model[model_name]
        model, log_df, preds = train_model(model_name, x, y, train_mask, val_mask, test_mask)
        logs.append(log_df)

        for split_name, mask, pred in [
            ("validation", val_mask, preds["validation"]),
            ("test", test_mask, preds["test"]),
        ]:
            mae, rmse = metrics(y[mask], pred)
            result_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "price_features": config["price_features"],
                    "news_features": config["news_features"],
                    "relationship_edges": config["relationship_edges"],
                    "co_mention_edges": config["co_mention_edges"],
                    "mae": mae,
                    "rmse": rmse,
                    "input_dim": int(x.shape[1]),
                }
            )
            frame = samples.loc[mask].copy()
            frame["model"] = model_name
            frame["y_pred"] = pred
            prediction_frames.append(frame)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "input_dim": int(x.shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "feature_cols": config["feature_cols"].tolist(),
            "edge_types": sorted(int(t) for t in config["edge_types"]),
            "edge_type_map": edge_type_map,
            "architecture": "one-hop weighted message passing MLP",
        }
        torch.save(checkpoint, OUT_DIR / config["checkpoint"])

    training_log = pd.concat(logs, ignore_index=True)
    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "model"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[
        ["model", "split", "snapshot_id", "article_id", "event_trading_date", "node_idx", "y_true", "y_pred"]
    ]
    comparison_test = results[results["split"].eq("test")].sort_values("mae").reset_index(drop=True)

    training_log.to_csv(OUT_DIR / "full_model_training_log.csv", index=False)
    results.to_csv(OUT_DIR / "ablation_results.csv", index=False)
    predictions.to_csv(OUT_DIR / "ablation_predictions.csv", index=False)
    comparison_test.to_csv(OUT_DIR / "model_comparison_test.csv", index=False)

    print("Generated week 6 ablation outputs in data/processed/")
    print(comparison_test.to_string(index=False))


if __name__ == "__main__":
    main()
