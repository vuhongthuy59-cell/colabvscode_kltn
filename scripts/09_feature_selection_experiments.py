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
OUT_DIR = ROOT / "outputs" / "09_feature_selection_experiments"

SNAPSHOT_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "graph_snapshots.pt"
SNAPSHOT_INDEX_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "snapshot_index.csv"
EDGE_TYPE_MAP_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "edge_type_map.json"
FEATURE_SCHEMA_FILE = ROOT / "outputs" / "04_build_graph_snapshots" / "node_feature_schema.json"
RF_IMPORTANCE_FILE = ROOT / "outputs" / "05_train_baselines" / "rf_feature_importance.csv"

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


def load_schema() -> dict:
    with open(FEATURE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def feature_indices(schema: dict, names: list[str]) -> list[int]:
    name_to_idx = {name: idx for idx, name in enumerate(schema["feature_names"])}
    missing = [name for name in names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown feature names: {missing}")
    return [name_to_idx[name] for name in names]


def ordered_rf_features(schema: dict) -> list[str]:
    if RF_IMPORTANCE_FILE.exists():
        importance = pd.read_csv(RF_IMPORTANCE_FILE)
        ranked = importance["feature"].astype(str).tolist()
        return [name for name in ranked if name in set(schema["feature_names"])]
    return schema["feature_names"]


def build_feature_configs(schema: dict) -> dict[str, list[int]]:
    ranked = ordered_rf_features(schema)
    price = schema["price_feature_names"]
    micro = schema["micro_feature_names"]
    macro = schema["macro_feature_names"]
    news = schema["news_feature_names"]
    industry = schema["industry_feature_names"]
    exposure = schema["exposure_feature_names"]

    configs = {
        "full_59": schema["feature_names"],
        "top_10_rf": ranked[:10],
        "top_20_rf": ranked[:20],
        "top_30_rf": ranked[:30],
        "price_only": price,
        "price_micro": price + micro,
        "price_macro": price + macro,
        "price_micro_macro": price + micro + macro,
        "price_industry_news": price + industry + news + exposure,
    }
    return {name: feature_indices(schema, values) for name, values in configs.items()}


def aggregate_for_targets(
    snapshot: dict,
    target_nodes: list[int],
    cols: list[int],
    edge_type_ids: set[int],
) -> np.ndarray:
    x = snapshot["x"][:, cols].float()
    edge_type = snapshot["edge_type"].long()
    edge_index = snapshot["edge_index"].long()
    edge_weight = snapshot["edge_weight"].float()

    selected = torch.zeros(edge_type.shape[0], dtype=torch.bool)
    for type_id in edge_type_ids:
        selected |= edge_type.eq(int(type_id))
    if int(selected.sum()) == 0:
        return np.zeros((len(target_nodes), len(cols)), dtype=np.float32)

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
    return np.vstack(result).astype(np.float32)


def build_samples(
    snapshots: list[dict],
    index: pd.DataFrame,
    cols: list[int],
    edge_type_ids: set[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    valid_snapshot_ids = set(index["snapshot_id"].astype(int))
    rows = []
    node_rows = []
    graph_rows = []
    y_rows = []

    for snapshot_id, snapshot in enumerate(snapshots):
        if snapshot_id not in valid_snapshot_ids:
            continue
        target_nodes = torch.where(snapshot["target_mask"])[0].tolist()
        if not target_nodes:
            continue
        neighbor_features = aggregate_for_targets(snapshot, target_nodes, cols, edge_type_ids)
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
            node_rows.append(node_features)
            graph_rows.append(np.concatenate([node_features, neighbor_features[row_pos]]).astype(np.float32))
            y_rows.append(float(snapshot["y"][node_idx]))

    return (
        pd.DataFrame(rows),
        np.vstack(node_rows).astype(np.float32),
        np.vstack(graph_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.float32),
    )


def train_onehop_mlp(
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


def add_tabular_results(
    rows: list[dict],
    experiment: str,
    feature_count: int,
    x_node: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    models = [
        ("Linear Regression", LinearRegression()),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
    for model_name, model in models:
        model.fit(x_node[train_mask], y[train_mask])
        for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
            pred = model.predict(x_node[mask]).astype(np.float32)
            mae, rmse = metrics(y[mask], pred)
            rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature_count": feature_count,
                    "input_dim": int(x_node.shape[1]),
                    "split": split_name,
                    "mae": mae,
                    "rmse": rmse,
                }
            )


def add_graph_results(
    rows: list[dict],
    experiment: str,
    feature_count: int,
    x_graph: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    preds = train_onehop_mlp(x_graph, y, train_mask, val_mask, test_mask)
    for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
        mae, rmse = metrics(y[mask], preds[split_name])
        rows.append(
            {
                "experiment": experiment,
                "model": "Selected One-Hop Graph MLP",
                "feature_count": feature_count,
                "input_dim": int(x_graph.shape[1]),
                "split": split_name,
                "mae": mae,
                "rmse": rmse,
            }
        )


def main() -> None:
    schema = load_schema()
    with open(EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        edge_type_map = json.load(f)
    all_edge_types = set(edge_type_map.values())

    index = pd.read_csv(SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = split_snapshot_ids(index)

    print("Loading graph snapshots ...")
    snapshots = torch.load(SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    result_rows = []
    config_rows = []
    for experiment, cols in build_feature_configs(schema).items():
        feature_names = [schema["feature_names"][idx] for idx in cols]
        print(f"Running {experiment}: {len(cols)} features")
        samples, x_node, x_graph, y = build_samples(snapshots, index, cols, all_edge_types)
        samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
        samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
        samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
        train_mask = samples["split"].eq("train").to_numpy()
        val_mask = samples["split"].eq("validation").to_numpy()
        test_mask = samples["split"].eq("test").to_numpy()

        add_tabular_results(result_rows, experiment, len(cols), x_node, y, train_mask, val_mask, test_mask)
        add_graph_results(result_rows, experiment, len(cols), x_graph, y, train_mask, val_mask, test_mask)
        config_rows.append(
            {
                "experiment": experiment,
                "feature_count": len(cols),
                "features": ";".join(feature_names),
            }
        )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "experiment", "model"]).reset_index(drop=True)
    configs = pd.DataFrame(config_rows).sort_values(["feature_count", "experiment"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "feature_selection_results.csv", index=False)
    configs.to_csv(OUT_DIR / "feature_selection_configs.csv", index=False, encoding="utf-8-sig")

    print("Generated feature selection outputs:")
    print((OUT_DIR / "feature_selection_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").to_string(index=False))


if __name__ == "__main__":
    main()
