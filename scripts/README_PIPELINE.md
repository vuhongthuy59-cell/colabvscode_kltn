# Clean Forecasting Pipeline

Run from the project root.

## Logic

The project is organized as a standard forecasting workflow:

```text
config
-> price data preparation
-> news data preparation
-> optional PhoBERT title embedding on Colab
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

| Step | Environment | Script | Main output |
|---:|---|---|---|
| 00 | VSCode | `project_config.py` | Shared path/data-scope configuration |
| 01 | VSCode/local | `01_prepare_price_data.py` | `outputs/local/01_price_data/` |
| 02 | VSCode/local | `02_prepare_news_data.py` | `outputs/local/02_news_data/` |
| 06B | Google Colab/GPU | `colab/06_phobert_embedding_colab.ipynb` | `outputs/local/02_news_data/news_title_embedding_pca.csv` |
| 03 | VSCode/local | `03_train_news_labeler.py` | `outputs/local/03_news_labeler/` |
| 04 | VSCode/local | `04_prepare_company_relationships.py` | `outputs/local/04_company_relationships/` |
| 05 | VSCode/local | `05_build_event_graph_dataset.py` | `outputs/local/05_event_graph_dataset/` |
| 06 | VSCode/local | `06_train_baseline_models.py` | `outputs/local/06_baseline_models/` |
| 07 | Google Colab/GPU | `07_train_gnn_ablation_models.py` | `outputs/colab/07_gnn_ablation_models/` |
| 08 | Google Colab/GPU | `08_tune_selected_gnn.py` | `outputs/colab/08_tuned_gnn/` |
| 09 | VSCode/report | `09_evaluate_models_and_cases.py` | `outputs/report/09_model_evaluation/` |
| 10 | VSCode/report | `10_generate_report_assets.py` | `outputs/report/10_report_assets/` |
| 11 | VSCode/report | `11_compute_regression_metrics.py` | `outputs/report/11_regression_metrics/` |
| 12 | Google Colab/GPU | `12_train_hybrid_mlp_gat.py` | `outputs/colab/12_hybrid_mlp_gat/` |
| 13 | VSCode/report | `13_plot_outputs_and_volatility.py` | `outputs/report/10_report_assets/` |

## Output Layout

```text
outputs/
  local/
    01_price_data/
    02_news_data/
      news_title_embedding_pca.csv  # optional PhoBERT PCA input for graph builder
    03_news_labeler/
    04_company_relationships/
    05_event_graph_dataset/
    06_baseline_models/
  colab/
    07_gnn_ablation_models/
    08_tuned_gnn/
    12_hybrid_mlp_gat/
  report/
    09_model_evaluation/
    10_report_assets/
    11_regression_metrics/
```

## Recommended Run Order

```powershell
# VSCode/local: data correctness and light baselines
python scripts\01_prepare_price_data.py
python scripts\02_prepare_news_data.py

# Google Colab/GPU optional: create PhoBERT PCA title embeddings
# Open and run: colab/06_phobert_embedding_colab.ipynb
# Then copy Drive output back to:
# outputs/local/02_news_data/news_title_embedding_pca.csv

python scripts\03_train_news_labeler.py
python scripts\04_prepare_company_relationships.py
python scripts\05_build_event_graph_dataset.py
python scripts\06_train_baseline_models.py

# Google Colab/GPU: heavy GNN training
python scripts\07_train_gnn_ablation_models.py
python scripts\08_tune_selected_gnn.py
python scripts\12_train_hybrid_mlp_gat.py

# VSCode/report: collect, evaluate, plot, write thesis assets
python scripts\09_evaluate_models_and_cases.py
python scripts\10_generate_report_assets.py
python scripts\11_compute_regression_metrics.py
python scripts\13_plot_outputs_and_volatility.py
```

## Archived Experiments

Older feature-selection, edge-ablation, residual-GNN, blending, anchored-GNN, and graph-Ridge experiments are archived under:

```text
archive/extra_scripts/
archive/extra_outputs/
archive/reports/
```

They are kept for traceability but are not part of the clean thesis pipeline.
