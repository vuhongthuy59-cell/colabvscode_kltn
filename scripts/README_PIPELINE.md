# Pipeline Scripts

Run scripts in this order from the project root.

Default data policy for the thesis experiments:

- Main experiment data: `2022-01-01` to `2025-12-31`.
- 2026 append data: reserved for web demo or robustness checks, not for the main comparison tables.
- By default, `01_build_price_dataset.py` and `02_build_news_dataset.py` ignore the 2026 append files even if they exist.
- To intentionally build an extended demo/robustness dataset through `2026-04-30`, run PowerShell with:

```powershell
$env:KLTN_INCLUDE_2026_APPEND = "1"
```

Clear that variable before generating the final thesis tables:

```powershell
Remove-Item Env:\KLTN_INCLUDE_2026_APPEND
```

| Script | Purpose | Main outputs |
|---|---|---|
| `01_build_price_dataset.py` | Process stock prices and market features | `stock_features.csv`, `master_log_return.csv`, `master_close.csv` |
| `02_build_news_dataset.py` | Process headlines, category, sentiment, ticker mapping | `news_articles.csv`, `news_mentions.csv`, `ticker_aliases.csv` |
| `03_build_company_relationships.py` | Generate company relationship edges | `company_relationships.csv`, `same_group_edges.csv`, `same_industry_edges.csv` |
| `04_build_graph_snapshots.py` | Build graph snapshots for each news event | `graph_snapshots.pt`, `snapshot_index.csv` |
| `05_train_baselines.py` | Train Rolling Volatility, tabular baselines, Corr-GNN | `baseline_results.csv`, `gnn_corr_model.pt` |
| `06_train_gnn_ablation_models.py` | Train GNN + News, GNN + Relationship, Full Model | `ablation_results.csv`, `full_gnn_model.pt` |
| `07_evaluate_results_and_case_studies.py` | Analyze metrics and case studies | `case_study_results.csv`, grouped error tables |
| `08_generate_report_tables_and_figures.py` | Generate report tables and figures | `.csv` and `.png` files under `report_outputs/` |
| `09_feature_selection_experiments.py` | Compare full features, top-k RF features, and feature-group subsets | `feature_selection_results.csv`, `feature_selection_configs.csv` |
| `10_edge_ablation_experiments.py` | Compare correlation thresholds and edge-type subsets for one-hop graph models | `edge_ablation_results.csv` |

Legacy unnumbered scripts are kept as references with the same core logic:
`build_market_outputs.py`, `build_news_outputs.py`, `build_relationship_outputs.py`,
`build_graph_snapshots.py`, `run_week5_baselines.py`, and `run_week6_ablation.py`.
