from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

SNAPSHOT_FILE = OUT_DIR / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = OUT_DIR / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = OUT_DIR / "edge_type_map.json"

RANDOM_STATE = 42
GNN_EPOCHS = 30
GNN_BATCH_SIZE = 512
GNN_HIDDEN_DIM = 64
GNN_LR = 1e-3


def feature_names(categories: list[str]) -> list[str]:
    return (
        [f"log_return_lag_{lag}" for lag in range(20, 0, -1)]
        + [
            "rolling_vol_20_t_minus_1",
            "volume_ratio_20_t_minus_1",
            "direct_news_sentiment",
            "news_relevance_score",
            "is_primary",
            "mention_count",
        ]
        + [f"category_{category}" for category in categories]
        + ["related_news_exposure"]
    )


class CorrOnlyGNN(torch.nn.Module):
    """One-hop correlation message passing baseline for target-node regression."""

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
    train_ids = set(ordered.iloc[:train_end]["snapshot_id"].astype(int))
    val_ids = set(ordered.iloc[train_end:val_end]["snapshot_id"].astype(int))
    test_ids = set(ordered.iloc[val_end:]["snapshot_id"].astype(int))
    return train_ids, val_ids, test_ids


def extract_corr_neighbor_feature(snapshot: dict, node_idx: int, price_feature_dim: int, corr_type_id: int) -> np.ndarray:
    x_price = snapshot["x"][:, :price_feature_dim].float()
    edge_type = snapshot["edge_type"]
    edge_index = snapshot["edge_index"]
    edge_weight = snapshot["edge_weight"].float()

    corr_mask = edge_type == corr_type_id
    if int(corr_mask.sum()) == 0:
        return np.zeros(price_feature_dim, dtype=np.float32)

    corr_edges = edge_index[:, corr_mask].long()
    corr_weights = edge_weight[corr_mask]
    target_mask = corr_edges[1] == int(node_idx)
    if int(target_mask.sum()) == 0:
        return np.zeros(price_feature_dim, dtype=np.float32)

    src = corr_edges[0, target_mask]
    weights = corr_weights[target_mask]
    denom = weights.sum().clamp_min(1e-8)
    agg = (x_price[src] * weights.unsqueeze(1)).sum(dim=0) / denom
    return agg.numpy().astype(np.float32)


def build_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    edge_type_map: dict[str, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    categories = sorted(index["category"].dropna().astype(str).unique().tolist())
    names = feature_names(categories)
    if len(names) != snapshots[0]["x"].shape[1]:
        raise ValueError(f"Feature-name count {len(names)} does not match x dim {snapshots[0]['x'].shape[1]}")

    index_by_snapshot = index.set_index("snapshot_id").to_dict(orient="index")
    rows = []
    x_rows = []
    y_rows = []
    rolling_preds = []
    gnn_rows = []

    price_feature_dim = 22
    corr_type_id = int(edge_type_map["price_correlation"])

    for snapshot_id, snapshot in enumerate(snapshots):
        meta = index_by_snapshot.get(snapshot_id)
        if meta is None:
            continue
        target_indices = torch.where(snapshot["target_mask"])[0].tolist()
        for node_idx in target_indices:
            x_node = snapshot["x"][node_idx].numpy().astype(np.float32)
            y_value = float(snapshot["y"][node_idx])
            neighbor_feature = extract_corr_neighbor_feature(snapshot, node_idx, price_feature_dim, corr_type_id)
            gnn_feature = np.concatenate([x_node[:price_feature_dim], neighbor_feature]).astype(np.float32)

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "article_id": snapshot["article_id"],
                    "event_trading_date": snapshot["event_trading_date"],
                    "node_idx": int(node_idx),
                    "y_true": y_value,
                    "split": "",
                }
            )
            x_rows.append(x_node)
            y_rows.append(y_value)
            rolling_preds.append(float(x_node[20]))
            gnn_rows.append(gnn_feature)

    samples = pd.DataFrame(rows)
    return (
        samples,
        np.vstack(x_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
        np.asarray(rolling_preds, dtype=np.float32),
        np.vstack(gnn_rows).astype(np.float32),
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return float(mae), float(rmse)


def train_corr_gnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, CorrOnlyGNN, pd.DataFrame]:
    torch.manual_seed(RANDOM_STATE)
    model = CorrOnlyGNN(input_dim=x_train.shape[1], hidden_dim=GNN_HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    loss_fn = torch.nn.MSELoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)

    log_rows = []
    best_state = None
    best_val_mae = float("inf")
    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(1, GNN_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        batch_losses = []
        for start in range(0, len(order), GNN_BATCH_SIZE):
            batch_idx = order[start : start + GNN_BATCH_SIZE]
            pred = model(x_train_t[batch_idx])
            loss = loss_fn(pred, y_train_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train_t).numpy()
            val_pred = model(x_val_t).numpy()
        train_mae, train_rmse = metrics(y_train, train_pred)
        val_mae, val_rmse = metrics(y_val, val_pred)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
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
        val_pred = model(x_val_t).numpy()
        test_pred = model(x_test_t).numpy()
    return val_pred, test_pred, model, pd.DataFrame(log_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)

    print("Loading graph_snapshots.pt ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    train_ids, val_ids, test_ids = split_snapshot_ids(index)
    samples, x, y, rolling_pred, gnn_x = build_samples(snapshots, index, edge_type_map)
    samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
    samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
    samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"

    train_mask = samples["split"].eq("train").to_numpy()
    val_mask = samples["split"].eq("validation").to_numpy()
    test_mask = samples["split"].eq("test").to_numpy()

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    prediction_frames = []
    result_rows = []

    for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
        mae, rmse = metrics(y[mask], rolling_pred[mask])
        result_rows.append(
            {
                "model": "Rolling Volatility",
                "split": split_name,
                "input": "historical volatility only",
                "graph": "no",
                "mae": mae,
                "rmse": rmse,
                "note": "prediction = rolling_vol_20 at T-1",
            }
        )
        frame = samples.loc[mask].copy()
        frame["model"] = "Rolling Volatility"
        frame["y_pred"] = rolling_pred[mask]
        prediction_frames.append(frame)

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)

    model_specs = [
        ("Linear Regression", linear, "tabular price/news features", "baseline linear tabular model"),
        ("Random Forest", rf, "tabular price/news features", "nonlinear tabular ML baseline"),
    ]
    for model_name, model, input_desc, note in model_specs:
        for split_name, mask, x_split, y_split in [
            ("validation", val_mask, x_val, y_val),
            ("test", test_mask, x_test, y_test),
        ]:
            pred = model.predict(x_split).astype(np.float32)
            mae, rmse = metrics(y_split, pred)
            result_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "input": input_desc,
                    "graph": "no",
                    "mae": mae,
                    "rmse": rmse,
                    "note": note,
                }
            )
            frame = samples.loc[mask].copy()
            frame["model"] = model_name
            frame["y_pred"] = pred
            prediction_frames.append(frame)

    gnn_val_pred, gnn_test_pred, gnn_model, gnn_log = train_corr_gnn(
        gnn_x[train_mask],
        y_train,
        gnn_x[val_mask],
        y_val,
        gnn_x[test_mask],
    )
    for split_name, mask, y_split, pred in [
        ("validation", val_mask, y_val, gnn_val_pred),
        ("test", test_mask, y_test, gnn_test_pred),
    ]:
        mae, rmse = metrics(y_split, pred)
        result_rows.append(
            {
                "model": "GNN Correlation Only",
                "split": split_name,
                "input": "price features only",
                "graph": "price_correlation only",
                "mae": mae,
                "rmse": rmse,
                "note": "one-hop weighted correlation message passing; no news/relationship edges",
            }
        )
        frame = samples.loc[mask].copy()
        frame["model"] = "GNN Correlation Only"
        frame["y_pred"] = pred
        prediction_frames.append(frame)

    names = feature_names(sorted(index["category"].dropna().astype(str).unique().tolist()))
    importance = pd.DataFrame(
        {
            "feature": names,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "model"]).reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[
        ["model", "split", "snapshot_id", "article_id", "event_trading_date", "node_idx", "y_true", "y_pred"]
    ]

    results.to_csv(OUT_DIR / "baseline_results.csv", index=False)
    predictions.to_csv(OUT_DIR / "baseline_predictions.csv", index=False)
    importance.to_csv(OUT_DIR / "rf_feature_importance.csv", index=False)
    gnn_log.to_csv(OUT_DIR / "gnn_corr_training_log.csv", index=False)
    torch.save(
        {
            "model_state_dict": gnn_model.state_dict(),
            "input_dim": gnn_x.shape[1],
            "hidden_dim": GNN_HIDDEN_DIM,
            "price_feature_dim": 22,
            "edge_type": "price_correlation",
            "architecture": "one-hop weighted correlation message passing MLP",
        },
        OUT_DIR / "gnn_corr_model.pt",
    )

    print("Generated week 5 baseline outputs in data/processed/")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
