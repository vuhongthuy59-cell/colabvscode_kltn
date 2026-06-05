from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "17_graph_regularized_baselines"


def load_anchor_module():
    script_path = ROOT / "scripts" / "16_train_anchored_gnn_experiments.py"
    spec = importlib.util.spec_from_file_location("anchored_gnn_helpers", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(mean_squared_error(y_true, y_pred, squared=False)),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    helper = load_anchor_module()

    index = pd.read_csv(helper.SNAPSHOT_INDEX_FILE)
    train_ids, val_ids, test_ids = helper.split_snapshot_ids(index)
    with open(helper.EDGE_TYPE_MAP_FILE, "r", encoding="utf-8") as f:
        import json

        corr_type_id = int(json.load(f)["price_correlation"])

    print("Loading graph snapshots ...")
    snapshots = helper.torch.load(helper.SNAPSHOT_FILE, map_location="cpu")
    print(f"Loaded {len(snapshots)} snapshots")

    configs = [
        ("top30_graph", helper.load_cols(30)),
        ("top50_graph", helper.load_cols(50)),
        ("all66_graph", helper.load_cols(None)),
    ]
    models = [
        ("Ridge_a0.1", make_pipeline(StandardScaler(), Ridge(alpha=0.1))),
        ("Ridge_a1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge_a10", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    ]

    result_rows = []
    for config_name, cols in configs:
        samples, x_tabular, x_graph, y, mean_edges = helper.build_samples(snapshots, index, cols, corr_type_id)
        samples.loc[samples["snapshot_id"].isin(train_ids), "split"] = "train"
        samples.loc[samples["snapshot_id"].isin(val_ids), "split"] = "validation"
        samples.loc[samples["snapshot_id"].isin(test_ids), "split"] = "test"
        train_mask = samples["split"].eq("train").to_numpy()
        val_mask = samples["split"].eq("validation").to_numpy()
        test_mask = samples["split"].eq("test").to_numpy()

        x_full = np.column_stack([x_tabular, x_graph]).astype(np.float32)
        for model_name, model in models:
            print(f"Training {config_name} {model_name} ...")
            model.fit(x_full[train_mask], y[train_mask])
            for split_name, mask in [("validation", val_mask), ("test", test_mask)]:
                pred = model.predict(x_full[mask]).astype(np.float32)
                mae, rmse = metrics(y[mask], pred)
                result_rows.append(
                    {
                        "model": model_name,
                        "config": config_name,
                        "split": split_name,
                        "feature_count": len(cols),
                        "input_dim": x_full.shape[1],
                        "mean_topk_edges_per_target": mean_edges,
                        "mae": mae,
                        "rmse": rmse,
                        "note": "Regularized linear model using tabular self features plus top-k graph interaction features",
                    }
                )

    results = pd.DataFrame(result_rows).sort_values(["split", "mae", "model", "config"]).reset_index(drop=True)
    results.to_csv(OUT_DIR / "graph_regularized_results.csv", index=False)
    print("Generated graph regularized results:")
    print((OUT_DIR / "graph_regularized_results.csv").as_posix())
    print(results[results["split"].eq("test")].sort_values("mae").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
