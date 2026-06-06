# Local Scientific Pipeline

Run all commands from the project root.

## Core idea

This repository is organized as a local forecasting research pipeline:

```text
data cleaning
-> feature engineering
-> target and graph construction
-> baseline models
-> GNN models
-> residual hybrid GNN
-> metrics, case studies, report assets
```

The official thesis pipeline writes model outputs to:

```text
outputs/local/
```

and report outputs to:

```text
outputs/report/
```

The old Colab branch was removed from the clean thesis pipeline. All official scripts now run locally.

## Step table

| Step | Script | Main output |
|---:|---|---|
| 00 | `scripts/00_run_local_pipeline.py` | local pipeline runner |
| 01 | `scripts/01_prepare_price_data.py` | `outputs/local/01_price_data/` |
| 02 | `scripts/02_prepare_news_data.py` | `outputs/local/02_news_data/` |
| 03 | `scripts/03_train_news_labeler.py` | `outputs/local/03_news_labeler/` |
| 04 | `scripts/04_prepare_company_relationships.py` | `outputs/local/04_company_relationships/` |
| 05 | `scripts/05_build_event_graph_dataset.py` | `outputs/local/05_event_graph_dataset/` |
| 06 | `scripts/06_train_baseline_models.py` | `outputs/local/06_baseline_models/` |
| 07 | `scripts/07_train_gnn_ablation_models.py` | `outputs/local/07_gnn_ablation_models/` |
| 08 | `scripts/08_tune_selected_gnn.py` | `outputs/local/08_tuned_gnn/` |
| 12 | `scripts/12_train_hybrid_mlp_gat.py` | `outputs/local/12_hybrid_mlp_gat/` |
| 14 | `scripts/14_train_residual_hybrid_gnn.py` | `outputs/local/14_residual_hybrid_gnn/` |
| 09 | `scripts/09_evaluate_and_report.py` | `outputs/report/09_model_evaluation/`, `outputs/report/10_report_assets/`, `outputs/report/11_regression_metrics/` |

Shared utilities:

```text
scripts/pipeline_utils.py
```

This file keeps common temporal split and regression metric helpers in one place. It does not change model logic.

## Recommended commands

Run everything:

```powershell
python scripts\00_run_local_pipeline.py
```

Run from graph construction onward:

```powershell
python scripts\00_run_local_pipeline.py --from-step 05
```

Refresh only final metrics and report assets:

```powershell
python scripts\00_run_local_pipeline.py --from-step 09
```

Skip heavy GNN scripts:

```powershell
python scripts\00_run_local_pipeline.py --skip-heavy
```

## Scientific interpretation

Use Linear Regression and Random Forest as tabular baselines.

Use GNN ablation to test whether graph, news and relationship layers add useful information.

Use Residual Hybrid GNN as the main defensible GNN direction:

```text
final_prediction = linear_baseline_prediction + alpha * graph_residual_prediction
```

This directly tests whether graph structure explains the part of stock volatility that the tabular baseline misses.
