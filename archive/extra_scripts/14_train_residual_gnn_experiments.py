from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "14_residual_gnn_experiments"

SNAPSHOT_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "edge_type_map.json"
FEATURE_SCHEMA_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "node_feature_schema.json"
RF_IMPORTANCE_FILE = ROOT / "outputs" / "05_train_baselines" / "rf_feature_importance.csv"

RANDOM_STATE = 42
BATCH_SIZE = 512
MAX_EPOCHS = 90
PATIENCE = 12
TOP_K = 20
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25]


class ResidualGraphMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.GELU(),
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


def load_feature_schema() -> dict:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rf_feature_cols(top_n: int) -> list[int]:
    schema = load_feature_schema()
    name_to_idx = {name: idx for idx, name in enumerate(schema["feature_names"])}
    importance = pd.read_csv(RF_IMPORTANCE_FILE)
    selected = [name for name in importance["feature"].astype(str).tolist() if name in name_to_idx]
    return [name_to_idx[name] for name in selected[:top_n]]


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
    rows = []
    selected_count = 0

    for node_idx in target_nodes:
        incoming = corr_edges[1].eq(int(node_idx))
        if int(incoming.sum()) == 0:
            rows.append(np.zeros(len(cols), dtype=np.float32))
            continue
        src = corr_edges[0, incoming]
        weights = corr_weights[incoming]
        keep = min(top_k, int(weights.numel()))
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
    top_k: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    valid_snapshot_ids = set(index["snapshot_id"].astype(int))
    rows = []
    tabular_rows = []
    graph_rows = []
    y_rows = []
    selected_edges = []

    for snapshot_id, snapshot in enumerate(snapshots):
        if snapshot_id not in valid_snapshot_ids:
            continue
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        if not target_nodes:
            continue
        neighbor_features, selected_count = aggregate_topk_corr_for_targets(snapshot, target_nodes, cols, corr_type_id, top_k)
        selected_edges.append(selected_count / max(len(target_nodes), 1))
        for row_pos, node_idx in enumerate(target_nodes):
            node_all = snapshot["x"][node_idx].numpy().astype(np.float32)
            node_selected = node_all[cols]
            neighbor = neighbor_features[row_pos]
            graph_features = np.concatenate(
                [
                    node_selected,
                    neighbor,
                    node_selected - neighbor,
                    node_selected * neighbor,
                ]
            ).astype(np.float32)

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
            tabular_rows.append(node_all)
            graph_rows.append(graph_features)
            y_rows.append(float(snapshot["y"][node_idx]))

    return (
        pd.DataFrame(rows),
        np.vstack(tabular_rows).astype(np.float32),
        np.vstack(graph_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        float(np.mean(selected_edges)) if selected_edges else 0.0,
    )


def fit_linear_baseline(
    x_tabular: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    model = LinearRegression()
    model.fit(x_tabular[train_mask], y[train_mask])
    return {
        "train": model.predict(x_tabular[train_mask]).astype(np.float32),
        "validation": model.predict(x_tabular[val_mask]).astype(np.float32),
        "test": model.predict(x_tabular[test_mask]).astype(np.float32),
        "all": model.predict(x_tabular).astype(np.float32),
    }


def train_residual_variant(
    variant: dict,
    x_graph: np.ndarray,
    residual_target: np.ndarray,
    y: np.ndarray,
    base_pred_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], float]:
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)

    x_scaler = StandardScaler()
    r_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_graph[train_mask]).astype(np.float32)
    x_val = x_scaler.transform(x_graph[val_mask]).astype(np.float32)
    x_test = x_scaler.transform(x_graph[test_mask]).astype(np.float32)
    r_train = r_scaler.fit_transform(residual_target[train_mask].reshape(-1, 1)).reshape(-1).astype(np.float32)

    model = ResidualGraphMLP(input_dim=x_graph.shape[1], hidden_dim=variant["hidden_dim"], dropout=variant["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant["lr"], weight_decay=variant["weight_decay"])
    loss_fn = torch.nn.SmoothL1Loss(beta=variant["huber_beta"])

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    r_train_t = torch.tensor(r_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    y_val = y[val_mask]
    base_val = base_pred_all[val_mask]
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
            loss = loss_fn(pred, r_train_t[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            val_resid_scaled = model(x_val_t).numpy()
        val_resid = r_scaler.inverse_transform(val_resid_scaled.reshape(-1, 1)).reshape(-1)
        val_pred = base_val + val_resid
        val_mae, val_rmse = metrics(y_val, val_pred)
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
                "val_mae_full_prediction": val_mae,
                "val_rmse_full_prediction": val_rmse,
            }
        )
        if stale_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_resid_scaled = model(x_val_t).numpy()
        test_resid_scaled = model(x_test_t).numpy()
    val_resid = r_scaler.inverse_transform(val_resid_scaled.reshape(-1, 1)).reshape(-1)
    test_resid = r_scaler.inverse_transform(test_resid_scaled.reshape(-1, 1)).reshape(-1)

    best_alpha = 0.0
    best_alpha_mae = float("inf")
    for alpha in ALPHA_GRID:
        val_pred = base_pred_all[val_mask] + alpha * val_resid
        val_mae, _ = metrics(y[val_mask], val_pred)
        if val_mae < best_alpha_mae:
            best_alpha_mae = val_mae
            best_alpha = alpha

    preds = {
        "validation": base_pred_all[val_mask] + best_alpha * val_resid,
        "test": base_pred_all[test_mask] + best_alpha * test_resid,
        "validation_residual": val_resid,
        "test_residual": test_resid,
    }
    return pd.DataFrame(log_rows), preds, float(best_alpha)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        corr_type_id = int(json.load(f)["price_correlation"])

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    variants = [
        {"name": "top30_h64_d000", "top_features": 30, "hidden_dim": 64, "dropout": 0.00, "lr": 8e-4, "weight_decay": 1e-4, "huber_beta": 0.50},
        {"name": "top30_h64_d010", "top_features": 30, "hidden_dim": 64, "dropout": 0.10, "lr": 8e-4, "weight_decay": 1e-4, "huber_beta": 0.50},
        {"name": "top40_h96_d010", "top_features": 40, "hidden_dim": 96, "dropout": 0.10, "lr": 6e-4, "weight_decay": 2e-4, "huber_beta": 0.50},
        {"name": "top50_h128_d015", "top_features": 50, "hidden_dim": 128, "dropout": 0.15, "lr": 5e-4, "weight_decay": 3e-4, "huber_beta": 0.50},
    ]

    result_rows = []
    prediction_frames = []
    log_frames = []

    cached_samples = {}
    for variant in variants:
        top_features = int(variant["top_features"])
        if top_features not in cached_samples:
            cols = load_rf_feature_cols(top_features)
            cached_samples[top_features] = build_samples(snapshots, index, cols, corr_type_id, TOP_K) + (len(cols),)
        samples, x_tabular, x_graph, y, mean_edges, feature_count = cached_samples[top_features]

        samples = samples.copy()
        samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
        samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
        samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
        train_mask = samples["split"].eq("train").to_numpy()
        val_mask = samples["split"].eq("validation").to_numpy()
        test_mask = samples["split"].eq("test").to_numpy()

        base_pred = fit_linear_baseline(x_tabular, y, train_mask, val_mask, test_mask)
        base_pred_all = base_pred["all"]
        residual_target = y - base_pred_all

        if not result_rows:
            for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
                mae, rmse = metrics(y[mask], base_pred_all[mask])
                result_rows.append(
                    {
                        "model": "Linear Regression",
                        "variant": "baseline",
                        "split": split_name,
                        "feature_count": x_tabular.shape[1],
                        "input_dim": x_tabular.shape[1],
                        "top_k_corr_neighbors": 0,
                        "mean_topk_edges_per_target": 0.0,
                        "alpha": 0.0,
                        "mae": mae,
                        "rmse": rmse,
                        "note": "Train-only linear baseline, included for direct comparison",
                    }
                )
                frame = samples.loc[mask].copy()
                frame["model"] = "Linear Regression"
                frame["variant"] = "baseline"
                frame["y_pred"] = base_pred_all[mask]
                prediction_frames.append(frame)

        print(f"Training residual GNN {variant['name']} ...")
        log_df, preds, alpha = train_residual_variant(
            variant,
            x_graph,
            residual_target,
            y,
            base_pred_all,
            train_mask,
            val_mask,
            test_mask,
        )
        log_frames.append(log_df)

        for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
            pred = preds[split_name]
            mae, rmse = metrics(y[mask], pred)
            result_rows.append(
                {
                    "model": "Residual TopK Graph MLP",
                    "variant": variant["name"],
                    "split": split_name,
                    "feature_count": feature_count,
                    "input_dim": int(x_graph.shape[1]),
                    "top_k_corr_neighbors": TOP_K,
                    "mean_topk_edges_per_target": mean_edges,
                    "alpha": alpha,
                    "mae": mae,
                    "rmse": rmse,
                    "note": "Final prediction = Linear baseline + alpha * GNN residual; alpha selected on validation MAE",
                }
            )
            frame = samples.loc[mask].copy()
            frame["model"] = "Residual TopK Graph MLP"
            frame["variant"] = variant["name"]
            frame["y_pred"] = pred
            prediction_frames.append(frame)

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "model", "variant"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[
        ["model", "variant", "split", "snapshot_id", "article_id", "event_trading_date", "node_idx", "y_true", "y_pred"]
    ]

    results.to_csv(OUT_DIR / "residual_gnn_results.csv", index=False)
    predictions.to_csv(OUT_DIR / "residual_gnn_predictions.csv", index=False)
    if log_frames:
        pd.concat(log_frames, ignore_index=True).to_csv(OUT_DIR / "residual_gnn_training_log.csv", index=False)

    print("Generated residual GNN results:")
    print((OUT_DIR / "residual_gnn_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
