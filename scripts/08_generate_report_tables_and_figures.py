from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = PROCESSED_DIR / "report_outputs"


def save_csv(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / name, index=False)


def save_fig(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=180)
    plt.close()


def plot_model_metrics(df: pd.DataFrame) -> None:
    test = df[df["split"].eq("test")].copy().sort_values("mae")
    x = range(len(test))
    width = 0.38
    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], test["mae"], width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], test["rmse"], width=width, label="RMSE")
    plt.xticks(list(x), test["model"], rotation=25, ha="right")
    plt.ylabel("Error")
    plt.title("Test-set model comparison")
    plt.legend()
    save_fig("figure_model_comparison_test.png")


def plot_rf_importance() -> None:
    importance = pd.read_csv(PROCESSED_DIR / "rf_feature_importance.csv").head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(importance["feature"], importance["importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance")
    save_fig("figure_rf_feature_importance_top15.png")


def plot_training_curves() -> None:
    log = pd.read_csv(PROCESSED_DIR / "full_model_training_log.csv")
    plt.figure(figsize=(10, 5))
    for model, group in log.groupby("model"):
        plt.plot(group["epoch"], group["val_mae"], label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.title("GNN ablation validation curves")
    plt.legend()
    save_fig("figure_gnn_validation_mae.png")


def plot_category_error() -> None:
    category = pd.read_csv(PROCESSED_DIR / "model_error_by_category.csv")
    best_model = (
        pd.read_csv(PROCESSED_DIR / "model_comparison_test.csv")
        .sort_values("mae")
        .iloc[0]["model"]
    )
    data = category[category["model"].eq(best_model)].sort_values("mae")
    plt.figure(figsize=(10, 5))
    plt.bar(data["category"], data["mae"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE")
    plt.title(f"Test MAE by news category: {best_model}")
    save_fig("figure_best_model_category_mae.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(PROCESSED_DIR / "baseline_results.csv")
    ablation = pd.read_csv(PROCESSED_DIR / "ablation_results.csv")
    comparison = pd.concat([baseline, ablation], ignore_index=True, sort=False)
    comparison = comparison.sort_values(["split", "mae", "model"]).reset_index(drop=True)

    test_table = comparison[comparison["split"].eq("test")].copy()
    test_table = test_table[
        [
            "model",
            "input",
            "graph",
            "price_features",
            "news_features",
            "relationship_edges",
            "co_mention_edges",
            "mae",
            "rmse",
            "note",
        ]
    ]
    for col in ["input", "graph", "price_features", "news_features", "relationship_edges", "co_mention_edges", "note"]:
        if col in test_table.columns:
            test_table[col] = test_table[col].fillna("")

    case_studies = pd.read_csv(PROCESSED_DIR / "case_study_results.csv")
    category_metrics = pd.read_csv(PROCESSED_DIR / "model_error_by_category.csv")
    ticker_metrics = pd.read_csv(PROCESSED_DIR / "model_error_by_ticker.csv")

    save_csv(comparison, "table_all_model_metrics.csv")
    save_csv(test_table, "table_model_comparison_test.csv")
    save_csv(category_metrics, "table_error_by_category.csv")
    save_csv(ticker_metrics, "table_error_by_ticker.csv")
    save_csv(case_studies, "table_case_studies.csv")

    plot_model_metrics(comparison)
    plot_rf_importance()
    plot_training_curves()
    plot_category_error()

    print("Generated report tables and figures.")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
