# Clean Forecasting Pipeline

Run from the project root.

## Logic

The project is organized as a standard forecasting workflow:

```text
config
-> price data preparation
-> news data preparation
-> news labeler validation
-> company relationship preparation
-> event graph dataset construction
-> baseline model training
-> GNN ablation training
-> selected GNN tuning
-> model evaluation and case studies
-> report asset generation
-> regression metric computation
```

## Scripts

| Step | Script | Main output |
|---:|---|---|
| 00 | `project_config.py` | Shared data-scope configuration |
| 01 | `01_prepare_price_data.py` | `outputs/01_price_data/` |
| 02 | `02_prepare_news_data.py` | `outputs/02_news_data/` |
| 03 | `03_train_news_labeler.py` | `outputs/03_news_labeler/` |
| 04 | `04_prepare_company_relationships.py` | `outputs/04_company_relationships/`; builds curated multilayer relationship edges for GNN |
| 05 | `05_build_event_graph_dataset.py` | `outputs/05_event_graph_dataset/` |
| 06 | `06_train_baseline_models.py` | `outputs/06_baseline_models/` |
| 07 | `07_train_gnn_ablation_models.py` | `outputs/07_gnn_ablation_models/` |
| 08 | `08_tune_selected_gnn.py` | `outputs/08_tuned_gnn/` |
| 09 | `09_evaluate_models_and_cases.py` | `outputs/09_model_evaluation/` |
| 10 | `10_generate_report_assets.py` | `outputs/10_report_assets/` |
| 11 | `11_compute_regression_metrics.py` | `outputs/11_regression_metrics/` |

## Recommended Run Order

```powershell
python scripts\01_prepare_price_data.py
python scripts\02_prepare_news_data.py
python scripts\03_train_news_labeler.py
python scripts\04_prepare_company_relationships.py
python scripts\05_build_event_graph_dataset.py
python scripts\06_train_baseline_models.py
python scripts\07_train_gnn_ablation_models.py
python scripts\08_tune_selected_gnn.py
python scripts\09_evaluate_models_and_cases.py
python scripts\10_generate_report_assets.py
python scripts\11_compute_regression_metrics.py
```

## Archived Experiments

Older feature-selection, edge-ablation, residual-GNN, blending, anchored-GNN, and graph-Ridge experiments are archived under:

```text
archive/extra_scripts/
archive/extra_outputs/
archive/reports/
```

They are kept for traceability but are not part of the clean thesis pipeline.
