from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline_utils import assign_temporal_split, regression_metrics
from project_config import local_output

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = local_output("12_hybrid_mlp_gat")

SNAPSHOT_FILE = local_output("05_event_graph_dataset") / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = local_output("05_event_graph_dataset") / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = local_output("05_event_graph_dataset") / "edge_type_map.json"

RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 256
HIDDEN_DIM = 96
DROPOUT = 0.15
LR = 8e-4
WEIGHT_DECAY = 2e-4
MAX_NEIGHBORS = 48


class HybridMLPGAT(torch.nn.Module):
    def __init__(self, feature_dim: int, num_edge_types: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.neighbor_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.edge_embedding = torch.nn.Embedding(num_edge_types, hidden_dim)
        self.edge_weight_encoder = torch.nn.Linear(1, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        self_x: torch.Tensor,
        neighbor_x: torch.Tensor,
        neighbor_type: torch.Tensor,
        neighbor_weight: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_h = self.self_encoder(self_x)
        neighbor_h = self.neighbor_encoder(neighbor_x)
        edge_h = self.edge_embedding(neighbor_type.clamp_min(0))
        weight_h = self.edge_weight_encoder(neighbor_weight.unsqueeze(-1))

        query = self_h.unsqueeze(1)
        attn_input = torch.tanh(query + neighbor_h + edge_h + weight_h)
        scores = self.attention(attn_input).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, -1e9)

        has_neighbor = neighbor_mask.any(dim=1)
        attention = torch.zeros_like(scores)
        if bool(has_neighbor.any()):
            attention[has_neighbor] = torch.softmax(scores[has_neighbor], dim=1)
        neighbor_agg = (neighbor_h * attention.unsqueeze(-1)).sum(dim=1)

        combined = torch.cat(
            [
                self_h,
                neighbor_agg,
                torch.abs(self_h - neighbor_agg),
                self_h * neighbor_agg,
            ],
            dim=1,
        )
        return self.head(combined).squeeze(-1)


metrics = regression_metrics


def sample_weights_from_train_quantiles(y: np.ndarray, q75: float, q90: float) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    weights[y > q75] = 2.0
    weights[y > q90] = 4.0
    return weights


def select_incoming_neighbors(snapshot: dict, node_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = snapshot["x"].float().numpy().astype(np.float32)
    edge_index = snapshot["edge_index"].long()
    edge_type = snapshot["edge_type"].long()
    edge_weight = snapshot["edge_weight"].float()

    incoming = edge_index[1].eq(int(node_idx))
    if int(incoming.sum()) == 0:
        feature_dim = x.shape[1]
        return (
            np.zeros((MAX_NEIGHBORS, feature_dim), dtype=np.float32),
            np.zeros(MAX_NEIGHBORS, dtype=np.int64),
            np.zeros(MAX_NEIGHBORS, dtype=np.float32),
            np.zeros(MAX_NEIGHBORS, dtype=bool),
        )

    src = edge_index[0, incoming].numpy().astype(np.int64)
    types = edge_type[incoming].numpy().astype(np.int64)
    weights = edge_weight[incoming].numpy().astype(np.float32)
    order = np.argsort(weights)[::-1][:MAX_NEIGHBORS]
    src = src[order]
    types = types[order]
    weights = weights[order]

    feature_dim = x.shape[1]
    neighbor_x = np.zeros((MAX_NEIGHBORS, feature_dim), dtype=np.float32)
    neighbor_type = np.zeros(MAX_NEIGHBORS, dtype=np.int64)
    neighbor_weight = np.zeros(MAX_NEIGHBORS, dtype=np.float32)
    neighbor_mask = np.zeros(MAX_NEIGHBORS, dtype=bool)

    n = len(src)
    neighbor_x[:n] = x[src]
    neighbor_type[:n] = types
    neighbor_weight[:n] = weights
    neighbor_mask[:n] = True
    return neighbor_x, neighbor_type, neighbor_weight, neighbor_mask


def build_samples(snapshots: list[dict], index: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    self_x_rows = []
    neighbor_x_rows = []
    neighbor_type_rows = []
    neighbor_weight_rows = []
    neighbor_mask_rows = []
    y_rows = []

    for snapshot_id, snapshot in enumerate(snapshots):
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        for node_idx in target_nodes:
            neighbor_x, neighbor_type, neighbor_weight, neighbor_mask = select_incoming_neighbors(snapshot, int(node_idx))
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
            self_x_rows.append(snapshot["x"][node_idx].numpy().astype(np.float32))
            neighbor_x_rows.append(neighbor_x)
            neighbor_type_rows.append(neighbor_type)
            neighbor_weight_rows.append(neighbor_weight)
            neighbor_mask_rows.append(neighbor_mask)
            y_rows.append(float(snapshot["y"][node_idx]))

    arrays = {
        "self_x": np.vstack(self_x_rows).astype(np.float32),
        "neighbor_x": np.stack(neighbor_x_rows).astype(np.float32),
        "neighbor_type": np.stack(neighbor_type_rows).astype(np.int64),
        "neighbor_weight": np.stack(neighbor_weight_rows).astype(np.float32),
        "neighbor_mask": np.stack(neighbor_mask_rows).astype(bool),
        "y": np.asarray(y_rows, dtype=np.float32),
    }
    return pd.DataFrame(rows), arrays


def predict_batches(model: HybridMLPGAT, arrays: dict[str, torch.Tensor], indices: np.ndarray) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start : start + BATCH_SIZE]
            pred = model(
                arrays["self_x"][batch],
                arrays["neighbor_x"][batch],
                arrays["neighbor_type"][batch],
                arrays["neighbor_weight"][batch],
                arrays["neighbor_mask"][batch],
            )
            preds.append(pred.numpy())
    return np.concatenate(preds).astype(np.float32)


def train_model(samples: pd.DataFrame, arrays_np: dict[str, np.ndarray], num_edge_types: int) -> tuple[HybridMLPGAT, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    arrays = {
        "self_x": torch.tensor(arrays_np["self_x"], dtype=torch.float32),
        "neighbor_x": torch.tensor(arrays_np["neighbor_x"], dtype=torch.float32),
        "neighbor_type": torch.tensor(arrays_np["neighbor_type"], dtype=torch.long),
        "neighbor_weight": torch.tensor(arrays_np["neighbor_weight"], dtype=torch.float32),
        "neighbor_mask": torch.tensor(arrays_np["neighbor_mask"], dtype=torch.bool),
    }
    y = arrays_np["y"]
    y_t = torch.tensor(y, dtype=torch.float32)

    train_idx = np.where(samples["split"].eq("train").to_numpy())[0]
    val_idx = np.where(samples["split"].eq("validation").to_numpy())[0]
    test_idx = np.where(samples["split"].eq("test").to_numpy())[0]

    q75 = float(np.quantile(y[train_idx], 0.75))
    q90 = float(np.quantile(y[train_idx], 0.90))
    weights = torch.tensor(sample_weights_from_train_quantiles(y, q75, q90), dtype=torch.float32)

    model = HybridMLPGAT(
        feature_dim=arrays_np["self_x"].shape[1],
        num_edge_types=num_edge_types,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log_rows = []
    best_state = None
    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch_np = order[start : start + BATCH_SIZE]
            batch = torch.tensor(batch_np, dtype=torch.long)
            pred = model(
                arrays["self_x"][batch],
                arrays["neighbor_x"][batch],
                arrays["neighbor_type"][batch],
                arrays["neighbor_weight"][batch],
                arrays["neighbor_mask"][batch],
            )
            loss = (weights[batch] * torch.abs(pred - y_t[batch])).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        train_pred = predict_batches(model, arrays, train_idx)
        val_pred = predict_batches(model, arrays, val_idx)
        train_mae, train_rmse = metrics(y[train_idx], train_pred)
        val_mae, val_rmse = metrics(y[val_idx], val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "weighted_mae_q75": q75,
                "weighted_mae_q90": q90,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    prediction_frames = []
    for split_name, split_idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        pred = predict_batches(model, arrays, split_idx)
        frame = samples.iloc[split_idx].copy()
        frame["model"] = "Hybrid MLP-GAT"
        frame["y_pred"] = pred
        prediction_frames.append(frame)

    return model, pd.DataFrame(log_rows), pd.concat(prediction_frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    samples, arrays = build_samples(snapshots, index)
    samples = assign_temporal_split(samples, index)

    model, training_log, predictions = train_model(samples, arrays, num_edge_types=len(edge_type_map))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "Hybrid MLP-GAT",
            "feature_dim": int(arrays["self_x"].shape[1]),
            "max_neighbors": MAX_NEIGHBORS,
            "edge_type_map": edge_type_map,
            "architecture": "self-feature MLP plus edge-type-aware neighbor attention",
        },
        OUT_DIR / "hybrid_mlp_gat_model.pt",
    )

    result_rows = []
    for split_name, group in predictions.groupby("split"):
        mae, rmse = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        result_rows.append(
            {
                "model": "Hybrid MLP-GAT",
                "split": split_name,
                "input": "full node features + graph attention neighbors",
                "graph": "signed top-k + ownership/value-chain/sector/news",
                "mae": mae,
                "rmse": rmse,
                "note": "Hybrid self MLP with edge-type-aware neighbor attention; weighted MAE loss",
            }
        )
    results = pd.DataFrame(result_rows).sort_values(["split", "mae"]).reset_index(drop=True)
    predictions = predictions[
        ["model", "split", "snapshot_id", "article_id", "event_trading_date", "node_idx", "y_true", "y_pred"]
    ]

    training_log.to_csv(OUT_DIR / "hybrid_training_log.csv", index=False)
    predictions.to_csv(OUT_DIR / "hybrid_predictions.csv", index=False)
    results.to_csv(OUT_DIR / "hybrid_results.csv", index=False)

    print("Generated Hybrid MLP-GAT outputs.")
    print(results[results["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
