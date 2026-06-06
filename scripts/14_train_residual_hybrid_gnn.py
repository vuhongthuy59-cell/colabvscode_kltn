from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler

from pipeline_utils import assign_temporal_split, regression_metrics, split_indices
from project_config import local_output

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = local_output("14_residual_hybrid_gnn")

SNAPSHOT_FILE = local_output("05_event_graph_dataset") / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = local_output("05_event_graph_dataset") / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = local_output("05_event_graph_dataset") / "edge_type_map.json"
FEATURE_SCHEMA_FILE = local_output("05_event_graph_dataset") / "node_feature_schema.json"

RANDOM_STATE = 42
EPOCHS = 35
BATCH_SIZE = 256
HIDDEN_DIM = 96
DROPOUT = 0.15
ATTENTION_DROPOUT = 0.45
LR = 7e-4
WEIGHT_DECAY = 2e-4
MAX_NEIGHBORS = 48
RIDGE_ALPHA = 10.0
HISTGBR_MAX_ITER = 100
HISTGBR_MAX_LEAF_NODES = 31
ALPHA_GRID = np.linspace(-1.0, 1.0, 81)
SHRINKAGE_FACTOR = 0.30


class ResidualGraphModel(torch.nn.Module):
    def __init__(self, feature_dim: int, num_edge_types: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout),
        )
        self.neighbor_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout),
        )
        self.edge_embedding = torch.nn.Embedding(num_edge_types, hidden_dim)
        self.edge_weight_encoder = torch.nn.Linear(1, hidden_dim)
        self.attention = torch.nn.Linear(hidden_dim, 1)
        self.attention_dropout = torch.nn.Dropout(ATTENTION_DROPOUT)
        self.neighbor_agg_norm = torch.nn.LayerNorm(hidden_dim)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LeakyReLU(negative_slope=0.01),
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

        scores = self.attention(torch.tanh(self_h.unsqueeze(1) + neighbor_h + edge_h + weight_h)).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, -1e9)

        has_neighbor = neighbor_mask.any(dim=1)
        attention = torch.zeros_like(scores)
        if bool(has_neighbor.any()):
            attention[has_neighbor] = torch.softmax(scores[has_neighbor], dim=1)
            attention[has_neighbor] = self.attention_dropout(attention[has_neighbor])
        neighbor_agg = (neighbor_h * attention.unsqueeze(-1)).sum(dim=1)
        neighbor_agg = self.neighbor_agg_norm(neighbor_agg)

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


def select_incoming_neighbors(snapshot: dict, node_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = snapshot["x"].float().numpy().astype(np.float32)
    edge_index = snapshot["edge_index"].long()
    edge_type = snapshot["edge_type"].long()
    edge_weight = snapshot["edge_weight"].float()

    incoming = edge_index[1].eq(int(node_idx))
    feature_dim = x.shape[1]
    if int(incoming.sum()) == 0:
        return (
            np.zeros((MAX_NEIGHBORS, feature_dim), dtype=np.float32),
            np.zeros(MAX_NEIGHBORS, dtype=np.int64),
            np.zeros(MAX_NEIGHBORS, dtype=np.float32),
            np.zeros(MAX_NEIGHBORS, dtype=bool),
        )

    src = edge_index[0, incoming].numpy().astype(np.int64)
    types = edge_type[incoming].numpy().astype(np.int64)
    weights = edge_weight[incoming].numpy().astype(np.float32)
    order = np.argsort(np.abs(weights))[::-1][:MAX_NEIGHBORS]
    src = src[order]
    types = types[order]
    weights = weights[order]

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


def build_samples(snapshots: list[dict]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
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


def standardize_graph_inputs(arrays: dict[str, np.ndarray], train_idx: np.ndarray) -> tuple[dict[str, np.ndarray], RobustScaler]:
    scaler = RobustScaler()
    scaler.fit(arrays["self_x"][train_idx])

    self_x = scaler.transform(arrays["self_x"]).astype(np.float32)
    flat_neighbor = arrays["neighbor_x"].reshape(-1, arrays["neighbor_x"].shape[-1])
    neighbor_x = scaler.transform(flat_neighbor).reshape(arrays["neighbor_x"].shape).astype(np.float32)

    scaled = dict(arrays)
    scaled["self_x"] = self_x
    scaled["neighbor_x"] = neighbor_x
    return scaled, scaler


def to_torch_arrays(arrays_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "self_x": torch.tensor(arrays_np["self_x"], dtype=torch.float32),
        "neighbor_x": torch.tensor(arrays_np["neighbor_x"], dtype=torch.float32),
        "neighbor_type": torch.tensor(arrays_np["neighbor_type"], dtype=torch.long),
        "neighbor_weight": torch.tensor(arrays_np["neighbor_weight"], dtype=torch.float32),
        "neighbor_mask": torch.tensor(arrays_np["neighbor_mask"], dtype=torch.bool),
    }


def predict_batches(model: ResidualGraphModel, arrays: dict[str, torch.Tensor], indices: np.ndarray) -> np.ndarray:
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


def tune_alpha(y_true: np.ndarray, baseline_pred: np.ndarray, residual_pred: np.ndarray) -> tuple[float, float, float]:
    best_alpha = 0.0
    best_mae = float("inf")
    best_rmse = float("inf")
    for alpha in ALPHA_GRID:
        final_pred = baseline_pred + float(alpha) * residual_pred
        mae, rmse = metrics(y_true, final_pred)
        if mae < best_mae:
            best_alpha = float(alpha)
            best_mae = mae
            best_rmse = rmse
    return best_alpha, best_mae, best_rmse


def tune_alpha_by_r2(y_true: np.ndarray, baseline_pred: np.ndarray, residual_pred: np.ndarray) -> float:
    best_alpha = 0.0
    best_r2 = -float("inf")
    for alpha in ALPHA_GRID:
        final_pred = baseline_pred + float(alpha) * residual_pred
        score = float(r2_score(y_true, final_pred))
        if score > best_r2:
            best_alpha = float(alpha)
            best_r2 = score
    return best_alpha


def sample_weights_from_train_quantiles(y: np.ndarray, q75: float, q90: float) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    weights[y > q75] = 2.0
    weights[y > q90] = 4.0
    return weights


def train_residual_model(
    samples: pd.DataFrame,
    arrays_np: dict[str, np.ndarray],
    baseline_pred: np.ndarray,
    num_edge_types: int,
) -> tuple[ResidualGraphModel, pd.DataFrame, dict[str, np.ndarray], float]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    y = arrays_np["y"]
    residual = y - baseline_pred
    indices = split_indices(samples)
    train_idx = indices["train"]
    val_idx = indices["validation"]
    test_idx = indices["test"]

    arrays_scaled, _ = standardize_graph_inputs(arrays_np, train_idx)
    arrays = to_torch_arrays(arrays_scaled)

    y_scaler = RobustScaler()
    residual_train_scaled = y_scaler.fit_transform(residual[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    residual_scaled = y_scaler.transform(residual.reshape(-1, 1)).reshape(-1).astype(np.float32)
    residual_t = torch.tensor(residual_scaled, dtype=torch.float32)

    q75 = float(np.quantile(y[train_idx], 0.75))
    q90 = float(np.quantile(y[train_idx], 0.90))
    weights = torch.tensor(sample_weights_from_train_quantiles(y, q75, q90), dtype=torch.float32)

    model = ResidualGraphModel(
        feature_dim=arrays_np["self_x"].shape[1],
        num_edge_types=num_edge_types,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5, reduction="none")

    best_state = None
    best_alpha = 0.0
    best_val_mae = float("inf")
    log_rows = []

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
            loss = (weights[batch] * loss_fn(pred, residual_t[batch])).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        val_residual_scaled = predict_batches(model, arrays, val_idx)
        val_residual_pred = y_scaler.inverse_transform(val_residual_scaled.reshape(-1, 1)).reshape(-1)
        alpha, val_mae, val_rmse = tune_alpha(y[val_idx], baseline_pred[val_idx], val_residual_pred)

        train_residual_scaled_pred = predict_batches(model, arrays, train_idx)
        train_residual_pred = y_scaler.inverse_transform(train_residual_scaled_pred.reshape(-1, 1)).reshape(-1)
        train_final_pred = baseline_pred[train_idx] + alpha * train_residual_pred
        train_mae, train_rmse = metrics(y[train_idx], train_final_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
                "alpha": alpha,
                "residual_train_std": float(residual_train_scaled.std()),
                "weighted_mae_q75": q75,
                "weighted_mae_q90": q90,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    preds = {}
    for split_name, idx in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        residual_scaled_pred = predict_batches(model, arrays, idx)
        residual_pred = y_scaler.inverse_transform(residual_scaled_pred.reshape(-1, 1)).reshape(-1)
        preds[split_name] = {
            "idx": idx,
            "baseline_pred": baseline_pred[idx],
            "residual_pred": residual_pred,
        }

    return model, pd.DataFrame(log_rows), preds, best_alpha


def load_baseline_feature_indices() -> tuple[np.ndarray, list[str]]:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    preferred_names = (
        schema.get("price_feature_names", [])
        + schema.get("micro_feature_names", [])
        + schema.get("macro_feature_names", [])
        + [
            "direct_news_sentiment",
            "news_relevance_score",
            "mention_count",
            "return_shock",
            "volatility_shock",
            "volume_shock",
            "negative_news_count",
            "sector_shock",
        ]
    )
    feature_names = schema["feature_names"]
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    selected_names = [name for name in preferred_names if name in name_to_idx]
    selected_indices = np.asarray([name_to_idx[name] for name in selected_names], dtype=np.int64)
    if selected_indices.size == 0:
        raise ValueError("No valid baseline features were selected for Ridge baseline.")
    return selected_indices, selected_names


def fit_tabular_baseline(
    baseline_kind: str,
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    baseline_cols, baseline_feature_names = load_baseline_feature_indices()
    scaler = RobustScaler()
    x_selected = x[:, baseline_cols]
    x_train = scaler.fit_transform(x_selected[train_mask])
    x_all = scaler.transform(x_selected)

    if baseline_kind == "ridge":
        baseline = Ridge(alpha=RIDGE_ALPHA)
    elif baseline_kind == "histgbr":
        baseline = HistGradientBoostingRegressor(
            max_iter=HISTGBR_MAX_ITER,
            max_leaf_nodes=HISTGBR_MAX_LEAF_NODES,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown baseline kind: {baseline_kind}")

    baseline.fit(x_train, y[train_mask])
    baseline_pred = baseline.predict(x_all).astype(np.float32)
    return baseline_pred, baseline_cols, baseline_feature_names


def append_baseline_results(
    model_name: str,
    input_name: str,
    note: str,
    samples: pd.DataFrame,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    prediction_frames: list[pd.DataFrame],
    result_rows: list[dict],
) -> None:
    for split_name in ["train", "validation", "test"]:
        idx = np.where(samples["split"].eq(split_name).to_numpy())[0]
        frame = samples.iloc[idx].copy()
        frame["model"] = model_name
        frame["baseline_pred"] = baseline_pred[idx]
        frame["graph_residual_pred"] = 0.0
        frame["residual_alpha"] = 0.0
        frame["y_pred"] = baseline_pred[idx]
        prediction_frames.append(frame)

        mae, rmse = metrics(y[idx], baseline_pred[idx])
        result_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "input": input_name,
                "graph": "none",
                "mae": mae,
                "rmse": rmse,
                "baseline_mae": mae,
                "baseline_rmse": rmse,
                "residual_alpha": 0.0,
                "note": note,
            }
        )


def append_residual_results(
    model_prefix: str,
    input_name: str,
    note_prefix: str,
    samples: pd.DataFrame,
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    best_alpha: float,
    prediction_frames: list[pd.DataFrame],
    result_rows: list[dict],
) -> tuple[float, float]:
    val_values = preds["validation"]
    alpha_r2 = tune_alpha_by_r2(
        y[val_values["idx"]],
        val_values["baseline_pred"],
        val_values["residual_pred"],
    )
    alpha_shrink = best_alpha * SHRINKAGE_FACTOR
    policies = [
        (
            f"{model_prefix} (MAE tuned)",
            best_alpha,
            f"{note_prefix}; alpha tuned on validation MAE",
        ),
        (
            f"{model_prefix} (R2 tuned)",
            alpha_r2,
            f"{note_prefix}; alpha tuned on validation R2",
        ),
        (
            f"{model_prefix} (MAE shrinkage)",
            alpha_shrink,
            f"{note_prefix}; validation-MAE alpha regularized by fixed shrinkage factor",
        ),
    ]

    for model_name, alpha, note in policies:
        for split_name, values in preds.items():
            idx = values["idx"]
            frame = samples.iloc[idx].copy()
            frame["model"] = model_name
            frame["baseline_pred"] = values["baseline_pred"]
            frame["graph_residual_pred"] = values["residual_pred"]
            frame["residual_alpha"] = alpha
            frame["y_pred"] = values["baseline_pred"] + alpha * values["residual_pred"]
            prediction_frames.append(frame)

            mae, rmse = metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
            baseline_mae, baseline_rmse = metrics(frame["y_true"].to_numpy(), frame["baseline_pred"].to_numpy())
            result_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "input": input_name,
                    "graph": "signed top-k + ownership/value-chain/sector/news",
                    "mae": mae,
                    "rmse": rmse,
                    "baseline_mae": baseline_mae,
                    "baseline_rmse": baseline_rmse,
                    "residual_alpha": alpha,
                    "note": note,
                }
            )
    return alpha_r2, alpha_shrink


def run_experiment_branch(
    branch: dict,
    samples: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    train_mask: np.ndarray,
    num_edge_types: int,
    prediction_frames: list[pd.DataFrame],
    result_rows: list[dict],
) -> tuple[dict, pd.DataFrame]:
    y = arrays["y"]
    baseline_pred, baseline_cols, baseline_feature_names = fit_tabular_baseline(
        branch["baseline_kind"],
        arrays["self_x"],
        y,
        train_mask,
    )

    append_baseline_results(
        model_name=branch["baseline_model_name"],
        input_name=branch["baseline_input"],
        note=branch["baseline_note"],
        samples=samples,
        y=y,
        baseline_pred=baseline_pred,
        prediction_frames=prediction_frames,
        result_rows=result_rows,
    )

    model, training_log, preds, best_alpha = train_residual_model(
        samples,
        arrays,
        baseline_pred,
        num_edge_types,
    )
    alpha_r2, alpha_shrink = append_residual_results(
        model_prefix=branch["hybrid_model_prefix"],
        input_name=branch["hybrid_input"],
        note_prefix=branch["hybrid_note"],
        samples=samples,
        y=y,
        preds=preds,
        best_alpha=best_alpha,
        prediction_frames=prediction_frames,
        result_rows=result_rows,
    )

    artifact = {
        "key": branch["key"],
        "baseline_kind": branch["baseline_kind"],
        "model_state_dict": model.state_dict(),
        "alpha_mae": best_alpha,
        "alpha_r2": alpha_r2,
        "alpha_shrinkage": alpha_shrink,
        "baseline_feature_count": int(len(baseline_cols)),
        "baseline_feature_names": baseline_feature_names,
    }
    return artifact, training_log.assign(branch=f"{branch['key']}_residual_gnn")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    samples, arrays = build_samples(snapshots)
    samples = assign_temporal_split(samples, index)

    y = arrays["y"]
    train_mask = samples["split"].eq("train").to_numpy()

    num_edge_types = int(max(edge_type_map.values())) + 1

    prediction_frames = []
    result_rows = []
    branches = [
        {
            "key": "ridge",
            "baseline_kind": "ridge",
            "baseline_model_name": "Ridge Regression",
            "baseline_input": "ridge baseline selected features",
            "baseline_note": "Two-stage control branch: Ridge baseline only",
            "hybrid_model_prefix": "Ridge Residual Hybrid GNN",
            "hybrid_input": "ridge baseline selected features + graph residual",
            "hybrid_note": "Ridge baseline plus edge-type-aware graph residual attention",
        },
        {
            "key": "histgbr",
            "baseline_kind": "histgbr",
            "baseline_model_name": "HistGBR",
            "baseline_input": "hist gradient boosting selected tabular features",
            "baseline_note": "Strong tabular branch: HistGradientBoostingRegressor baseline only",
            "hybrid_model_prefix": "HistGBR Residual Hybrid GNN",
            "hybrid_input": "hist gradient boosting selected tabular features + graph residual",
            "hybrid_note": "HistGBR baseline plus edge-type-aware graph residual attention",
        },
    ]

    artifacts = []
    training_logs = []
    for branch in branches:
        artifact, training_log = run_experiment_branch(
            branch=branch,
            samples=samples,
            arrays=arrays,
            train_mask=train_mask,
            num_edge_types=num_edge_types,
            prediction_frames=prediction_frames,
            result_rows=result_rows,
        )
        artifacts.append(artifact)
        training_logs.append(training_log)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    results = pd.DataFrame(result_rows).sort_values(["split", "mae"]).reset_index(drop=True)
    artifact_by_key = {artifact["key"]: artifact for artifact in artifacts}

    torch.save(
        {
            "ridge_model_state_dict": artifact_by_key["ridge"]["model_state_dict"],
            "histgbr_model_state_dict": artifact_by_key["histgbr"]["model_state_dict"],
            "model_name": "Residual Hybrid GNN experimental branches",
            "feature_dim": int(arrays["self_x"].shape[1]),
            "max_neighbors": MAX_NEIGHBORS,
            "edge_type_map": edge_type_map,
            "ridge_residual_alpha_mae": artifact_by_key["ridge"]["alpha_mae"],
            "ridge_residual_alpha_r2": artifact_by_key["ridge"]["alpha_r2"],
            "ridge_residual_alpha_shrinkage": artifact_by_key["ridge"]["alpha_shrinkage"],
            "histgbr_residual_alpha_mae": artifact_by_key["histgbr"]["alpha_mae"],
            "histgbr_residual_alpha_r2": artifact_by_key["histgbr"]["alpha_r2"],
            "histgbr_residual_alpha_shrinkage": artifact_by_key["histgbr"]["alpha_shrinkage"],
            "baseline_models": ["Ridge", "HistGradientBoostingRegressor"],
            "baseline_alpha": RIDGE_ALPHA,
            "histgbr_max_iter": HISTGBR_MAX_ITER,
            "histgbr_max_leaf_nodes": HISTGBR_MAX_LEAF_NODES,
            "baseline_feature_count": artifact_by_key["ridge"]["baseline_feature_count"],
            "baseline_feature_names": artifact_by_key["ridge"]["baseline_feature_names"],
            "attention_dropout": ATTENTION_DROPOUT,
            "activation": "LeakyReLU(negative_slope=0.01)",
            "hidden_dim": HIDDEN_DIM,
            "graph_scaler": "RobustScaler",
            "residual_scaler": "RobustScaler",
            "architecture": "two-stage tabular baseline plus edge-type-aware graph residual attention",
        },
        OUT_DIR / "residual_hybrid_gnn_model.pt",
    )
    pd.concat(training_logs, ignore_index=True).to_csv(OUT_DIR / "residual_training_log.csv", index=False)
    predictions[
        [
            "model",
            "split",
            "snapshot_id",
            "article_id",
            "event_trading_date",
            "node_idx",
            "y_true",
            "baseline_pred",
            "graph_residual_pred",
            "residual_alpha",
            "y_pred",
        ]
    ].to_csv(OUT_DIR / "residual_predictions.csv", index=False)
    results.to_csv(OUT_DIR / "residual_results.csv", index=False)

    print("Generated Residual Hybrid GNN outputs.")
    print(results[results["split"].eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
