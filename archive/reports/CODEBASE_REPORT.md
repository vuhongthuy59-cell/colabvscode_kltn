# Codebase Report - Stock News Volatility Forecasting BDA Thesis

## 1. Project Summary

This project is a BDA thesis pipeline for forecasting short-term stock volatility after financial news events in the Vietnamese stock market.

Core research question:

> When a news article appears about a listed company, can we forecast the realized volatility of the related stock over the next 5 trading days using price history, news signals, company relationships, micro features, macro features, and graph-based relationships?

Main prediction target:

```text
5-day realized volatility after the news event
```

The project does not primarily predict stock price direction. It predicts how strongly the stock moves after news.

Main experiment scope:

```text
2022-01-01 to 2025-12-31
```

2026 appended data is reserved for web demo or robustness checks, not for main thesis comparison tables.

## 2. High-Level Logic

The project assumes that stock volatility after news depends on multiple information sources:

1. Historical price movement of the stock.
2. Liquidity and size characteristics of the stock.
3. General market and macro context.
4. News content, category, and sentiment.
5. Industry and company relationships.
6. Graph relationships between stocks.

Therefore, each news article is converted into a graph snapshot:

```text
Node = stock ticker
Edge = relationship between two stocks
Node feature = price/micro/macro/industry/news/exposure features
Label = 5-day realized volatility after event date
```

Each graph has 118 stock nodes.

## 3. Directory Structure

```text
mainforecast_KLTN/
  data/
    data_origial/
      universe.csv
      Stock_Price_2022-2025.csv
      Stock_Price_2026_append.csv
      News_2022_2025_2.xlsx
      Vietstock_News_2025_2026_append.xlsx
      relationships.xlsx
    processed/
      generated CSV/PT/JSON outputs
      report_outputs/
        generated tables and figures
  scripts/
    numbered thesis pipeline scripts
    legacy/helper scripts
  web_demo/
    static dashboard demo
```

Note: the folder is named `data_origial` in the current project.

## 4. Main Data Statistics

From current processed outputs:

| Item | Value |
|---|---:|
| Stock universe size | 118 |
| Firm-specific news articles | 22,019 |
| Saved graph snapshots | 16,754 |
| Skipped articles | 5,265 |
| Nodes per graph | 118 |
| Node features | 59 |
| Edge types | 6 |
| Mean edges per snapshot | 13,650.21 |
| Min edges per snapshot | 12,692 |
| Max edges per snapshot | 14,868 |
| Mean target stocks per article | 1.056 |
| Unique event dates | 740 |
| Label horizon | 5 trading days |
| Return lookback | 20 trading days |
| Correlation lookback | 252 trading days |

Skipped snapshot reasons:

| Reason | Count |
|---|---:|
| Insufficient 252-day correlation history | 4,606 |
| Insufficient 20-day feature history | 565 |
| Insufficient 5-day future label | 91 |
| Insufficient target future returns | 3 |

## 5. Node Features

Each stock node currently has 59 features:

| Feature group | Count | Meaning |
|---|---:|---|
| Price features | 22 | historical returns, volatility, volume |
| Micro features | 2 | stock-level liquidity and market cap |
| Industry features | 12 | one-hot industry group |
| Macro features | 8 | market/VNIndex/USD/Oil context |
| News features | 14 | sentiment, relevance, primary flag, category |
| Exposure feature | 1 | related news exposure through company relationships |
| Total | 59 | |

### 5.1 Price Features: 22

```text
log_return_lag_20
log_return_lag_19
...
log_return_lag_1
rolling_vol_20_t_minus_1
volume_ratio_20_t_minus_1
```

Meaning:

- 20 previous daily log returns before the news event.
- Previous 20-day rolling volatility.
- Volume ratio compared with the 20-day average.

### 5.2 Micro Features: 2

```text
trading_value_ratio_20
log_market_cap
```

Meaning:

- `trading_value_ratio_20`: stock-level relative trading value/liquidity.
- `log_market_cap`: log market capitalization, representing firm size.

### 5.3 Industry Features: 12

```text
industry_Banking
industry_ConstructionInfra
industry_ConsumerFoodAgri
industry_EnergyUtilitiesOilGas
industry_HealthcarePharma
industry_ICTTelecom
industry_IndustrialsHoldings
industry_MaterialsChemicalsSteel
industry_RealEstate
industry_RetailDistribution
industry_Securities
industry_TransportLogisticsAviation
```

These are one-hot encoded industry indicators.

### 5.4 Macro Features: 8

```text
universe_market_return
universe_market_roll_vol_20
market_liquidity_ratio_20
vni_return_1d
vni_roll_vol_20
vni_available
usd_vnd_return_1d
oil_return_1d
```

Meaning:

- Overall market return and volatility from the 118-stock universe.
- Market liquidity condition.
- VN-Index return and volatility.
- USD/VND return.
- Oil price return.

### 5.5 News Features: 14

```text
direct_news_sentiment
news_relevance_score
is_primary
mention_count
category_capital_issuance
category_debt_bond
category_dividend
category_earnings
category_leadership
category_legal_regulatory
category_ma_ownership
category_market_industry
category_other
category_project_contract
```

News sentiment and category are currently rule-based.

### 5.6 Exposure Feature: 1

```text
related_news_exposure
```

This captures indirect exposure through company relationship edges when a stock is not directly mentioned but is related to a mentioned stock.

## 6. Edge Types

Current edge type map:

| Edge type | ID | Meaning |
|---|---:|---|
| price_correlation | 0 | price return correlation |
| parent_to_subsidiary | 1 | parent company to subsidiary |
| subsidiary_to_parent | 2 | subsidiary to parent company |
| same_group | 3 | same corporate group/ecosystem |
| same_industry | 4 | same industry |
| news_co_mention | 5 | co-mentioned in the same article |

### 6.1 Price Correlation Edge

Defined in `04_build_graph_snapshots.py`.

Current constants:

```python
CORR_LOOKBACK = 252
CORR_THRESHOLD = 0.15
```

For every event date, the pipeline calculates Pearson correlation between stock returns over the previous 252 trading days.

Edge rule:

```text
If abs(corr) >= 0.15, create a price_correlation edge.
Edge weight = abs(corr).
```

`abs(corr)` means absolute value of Pearson correlation. Both strong positive and strong negative correlations are treated as strong relationships.

### 6.2 Company Relationship Edges

Defined in `03_build_company_relationships.py`.

Current heuristic/default weights:

```python
DEFAULT_SAME_GROUP_WEIGHT = 0.80
DEFAULT_SAME_INDUSTRY_WEIGHT = 0.50
DEFAULT_PARENT_TO_SUB_WEIGHT = 1.00
DEFAULT_SUB_TO_PARENT_WEIGHT = 0.70
```

If ownership percentage is available:

```text
parent_to_subsidiary weight = ownership_pct / 100
subsidiary_to_parent weight = ownership * 0.70
```

If ownership percentage is not available:

```text
parent_to_subsidiary weight = 1.00
subsidiary_to_parent weight = 0.70
```

Same-group and same-industry weights are heuristic.

Heuristic means rule-based or assumption-based values, not learned directly from data.

### 6.3 News Co-Mention Edge

Created when a news article mentions at least two tickers.

Edge rule:

```text
For all mentioned stock pairs in an article, create directed co-mention edges.
Edge weight = min(relevance_i, relevance_j).
```

Relevance score comes from ticker/name/alias matching:

| Match method | Base relevance |
|---|---:|
| ticker_match | 1.00 |
| company_name_match | 0.95 |
| alias_match | 0.90 |
| source_ticker | 0.80 |

Adjustment:

```text
relevance = min(1.0, base_relevance + 0.05 * (mention_count - 1))
```

## 7. Pipeline Scripts

## 7.1 `scripts/data_config.py`

Defines dataset scope and environment switch:

```text
MAIN_START_DATE = 2022-01-01
MAIN_END_DATE = 2025-12-31
DEMO_START_DATE = 2026-01-01
DEMO_END_DATE = 2026-04-30
KLTN_INCLUDE_2026_APPEND
```

Purpose:

- Main thesis experiments use 2022-2025.
- 2026 append files are ignored by default.
- 2026 data is only included when `KLTN_INCLUDE_2026_APPEND=1`.

## 7.2 `scripts/01_build_price_dataset.py`

Purpose:

Build cleaned price and price-based features.

Inputs:

```text
data/data_origial/universe.csv
data/data_origial/Stock_Price_2022-2025.csv
data/data_origial/Stock_Price_2026_append.csv optional
data/data_origial/relationships.xlsx
```

Main techniques:

- ticker/date normalization
- duplicate removal
- log return calculation
- rolling volatility
- rolling volume features
- matrix pivoting

Main outputs:

```text
stock_prices.csv
stock_features.csv
master_log_return.csv
master_rolling_volatility_20.csv
master_rolling_volatility_60.csv
master_close.csv
master_matrix.csv
ticker_metadata.csv
ticker_list.csv
```

## 7.3 `scripts/02_build_news_dataset.py`

Purpose:

Build structured news dataset and ticker mentions.

Inputs:

```text
News_2022_2025_2.xlsx
Vietstock_News_2025_2026_append.xlsx optional
ticker_list.csv
master_close.csv
```

Main techniques:

- text normalization
- manual alias mapping
- ticker/name/alias matching
- rule-based category classification
- rule-based sentiment scoring
- mapping published date to next trading date

Outputs:

```text
news_articles.csv
news_mentions.csv
ticker_aliases.csv
```

Current limitation:

- Category and sentiment are rule-based, not supervised NLP models.
- Vietnamese text in some source/code sections has mojibake/encoding artifacts, though generated outputs appear usable.

## 7.4 `scripts/03_build_company_relationships.py`

Purpose:

Build company relationship edges for graph construction.

Inputs:

```text
ticker_metadata.csv
relationships.xlsx
```

Main techniques:

- graph edge construction
- directed and undirected relationship representation
- heuristic edge weighting
- ownership percentage extraction
- data quality reporting

Main outputs:

```text
company_groups.csv
parent_subsidiary_raw.csv
parent_subsidiary_edges.csv
same_group_edges.csv
same_industry_edges.csv
company_relationships.csv
relationship_quality_report.csv
```

Relationship quality statistics:

| Metric | Value |
|---|---:|
| Metadata tickers | 118 |
| Raw relationship rows | 10 |
| Excluded not in universe | 1 |
| Unverified rows | 1 |
| Company groups | 3 |
| Parent/subsidiary raw rows | 3 |
| Parent/subsidiary edges | 6 |
| Same-group edges | 50 |
| Same-industry edges | 1,342 |
| Company relationship rows | 1,398 |
| Duplicate relationship rows | 0 |

## 7.5 `scripts/build_micro_macro_features.py`

Purpose:

Create additional stock-level micro features and market/macro features.

Main outputs:

```text
stock_micro_features.csv
market_macro_features.csv
micro_macro_quality_report.csv
```

Quality statistics:

| Metric | Value |
|---|---:|
| Micro rows | 117,180 |
| Micro tickers | 118 |
| Market cap failed tickers | 0 |
| Market cap missing rows after fill | 0 |
| Macro rows | 997 |
| VNI available rows | 288 |
| USD/VND nonzero return rows | 916 |
| Oil nonzero return rows | 961 |

## 7.6 `scripts/04_build_graph_snapshots.py`

Purpose:

Build one graph snapshot for each valid firm-specific news article.

Inputs:

```text
ticker_list.csv
stock_features.csv
master_log_return.csv
news_articles.csv
news_mentions.csv
company_relationships.csv
ticker_metadata.csv
stock_micro_features.csv
market_macro_features.csv
```

Main constants:

```python
LOOKBACK_RETURNS = 20
CORR_LOOKBACK = 252
LABEL_HORIZON = 5
CORR_THRESHOLD = 0.15
```

Main techniques:

- event-based graph snapshot construction
- rolling price correlation graph
- relationship graph
- news co-mention graph
- masked node-level regression label
- future realized volatility label

Main outputs:

```text
graph_snapshots.pt
snapshot_index.csv
ticker_to_idx.json
idx_to_ticker.json
edge_type_map.json
node_feature_schema.json
graph_snapshot_quality_report.csv
```

## 7.7 `scripts/05_train_baselines.py`

Purpose:

Train baseline models.

Models:

```text
Rolling Volatility
Linear Regression
Random Forest
GNN Correlation Only
```

Main techniques:

- time-ordered train/validation/test split
- rolling volatility baseline
- tabular regression
- one-hop correlation aggregation
- MAE/RMSE evaluation

Outputs:

```text
baseline_results.csv
baseline_predictions.csv
rf_feature_importance.csv
gnn_corr_training_log.csv
gnn_corr_model.pt
```

## 7.8 `scripts/06_train_gnn_ablation_models.py`

Purpose:

Train graph/news/relationship ablation models.

Models:

```text
GNN Correlation Only
GNN + News
GNN + Relationship
Full Model
```

Important note:

The current graph model is best described as:

```text
one-hop weighted message passing + MLP regression
```

It is GNN-inspired but not a full GCN/GAT architecture.

Outputs:

```text
ablation_results.csv
ablation_predictions.csv
model_comparison_test.csv
full_model_training_log.csv
gnn_corr_ablation_model.pt
gnn_news_model.pt
gnn_relationship_model.pt
full_gnn_model.pt
```

## 7.9 `scripts/07_evaluate_results_and_case_studies.py`

Purpose:

Evaluate model predictions and produce case studies.

Main techniques:

- merge predictions with event/news/ticker context
- grouped error metrics by category/ticker/event date
- best/worst/high-volatility case study selection

Outputs:

```text
all_model_predictions_enriched.csv
model_error_by_category.csv
model_error_by_ticker.csv
model_error_by_event_date.csv
case_study_results.csv
```

## 7.10 `scripts/08_generate_report_tables_and_figures.py`

Purpose:

Generate report-ready tables and figures.

Outputs:

```text
data/processed/report_outputs/table_all_model_metrics.csv
data/processed/report_outputs/table_model_comparison_test.csv
data/processed/report_outputs/table_error_by_category.csv
data/processed/report_outputs/table_error_by_ticker.csv
data/processed/report_outputs/table_case_studies.csv
data/processed/report_outputs/figure_rf_feature_importance_top15.png
data/processed/report_outputs/figure_model_comparison_test.png
data/processed/report_outputs/figure_gnn_validation_mae.png
data/processed/report_outputs/figure_best_model_category_mae.png
```

## 7.11 `scripts/09_feature_selection_experiments.py`

Purpose:

Test whether selected feature subsets improve performance compared with using all 59 node features.

Inputs:

```text
graph_snapshots.pt
snapshot_index.csv
node_feature_schema.json
rf_feature_importance.csv
edge_type_map.json
```

Techniques:

- feature selection
- model-based selection using Random Forest importance
- group-wise feature ablation
- Linear Regression
- Random Forest
- one-hop graph aggregation + MLP

Feature experiments:

| Experiment | Feature count | Meaning |
|---|---:|---|
| full_59 | 59 | all current features |
| top_10_rf | 10 | top 10 RF importance features |
| top_20_rf | 20 | top 20 RF importance features |
| top_30_rf | 30 | top 30 RF importance features |
| price_only | 22 | price features only |
| price_micro | 24 | price + micro |
| price_macro | 30 | price + macro |
| price_micro_macro | 32 | price + micro + macro |
| price_industry_news | 49 | price + industry + news + exposure |

Outputs:

```text
feature_selection_results.csv
feature_selection_configs.csv
```

Best test results from file 09:

| Experiment | Model | Feature count | MAE test | RMSE test |
|---|---|---:|---:|---:|
| price_macro | Random Forest | 30 | 0.008203 | 0.011113 |
| price_industry_news | Linear Regression | 49 | 0.008236 | 0.011224 |
| price_only | Random Forest | 22 | 0.008264 | 0.011200 |
| price_only | Linear Regression | 22 | 0.008266 | 0.011237 |
| price_micro | Linear Regression | 24 | 0.008301 | 0.011257 |
| price_micro | Random Forest | 24 | 0.008314 | 0.011263 |
| full_59 | Random Forest | 59 | 0.008359 | 0.011225 |

Interpretation:

- Full 59 features are not optimal.
- `price_macro` is currently best.
- Feature selection improves the best model from roughly MAE 0.00835 to 0.00820.
- Macro features appear more useful than micro features in the current setup.

## 7.12 `scripts/10_edge_ablation_experiments.py`

Purpose:

Test which graph edge subsets perform best and whether full graph is necessary.

Inputs:

```text
graph_snapshots.pt
snapshot_index.csv
node_feature_schema.json
edge_type_map.json
```

Techniques:

- edge ablation
- correlation threshold sensitivity
- one-hop weighted message passing
- MLP regression
- MAE/RMSE evaluation

Edge experiments:

| Experiment | Edge setup |
|---|---|
| corr_threshold_0_15 | price correlation only, min weight 0.15 |
| corr_threshold_0_20 | price correlation only, min weight 0.20 |
| corr_threshold_0_30 | price correlation only, min weight 0.30 |
| corr_plus_same_industry | correlation + same industry |
| corr_plus_same_group | correlation + same group |
| corr_plus_parent_subsidiary | correlation + parent/subsidiary |
| corr_plus_news_co_mention | correlation + news co-mention |
| full_graph | all edge types |

Output:

```text
edge_ablation_results.csv
```

Best test results from file 10:

| Experiment | Mean selected edges | MAE test | RMSE test |
|---|---:|---:|---:|
| corr_plus_same_industry | 13,594.07 | 0.011173 | 0.014184 |
| full_graph | 13,650.21 | 0.012597 | 0.015994 |
| corr_plus_news_co_mention | 12,252.21 | 0.013220 | 0.017125 |
| corr_plus_parent_subsidiary | 12,258.07 | 0.013674 | 0.017689 |
| corr_plus_same_group | 12,302.07 | 0.013675 | 0.017690 |
| corr_threshold_0_15 | 12,252.07 | 0.013678 | 0.017695 |
| corr_threshold_0_20 | 11,389.98 | 0.013978 | 0.017706 |
| corr_threshold_0_30 | 8,998.71 | 0.014514 | 0.018219 |

Interpretation:

- Full graph is not the best graph configuration.
- Correlation + same industry performs best among tested graph variants.
- Increasing correlation threshold from 0.15 to 0.20/0.30 makes the graph sparser and worsens performance in this setup.
- Thresholds below 0.15 were not tested because current snapshots were originally built with `CORR_THRESHOLD = 0.15`. Testing 0.10 requires rebuilding graph snapshots.

## 8. Train/Validation/Test Split

The project uses a time-ordered split:

```text
Train: first 70% of snapshots by event_trading_date
Validation: next 15%
Test: final 15%
```

Purpose:

| Split | Role |
|---|---|
| Train | used to fit model parameters |
| Validation | used to select model/epoch/configuration |
| Test | final evaluation for reporting |

Reason for time-ordered split:

Financial data has a natural time direction. Random split may leak future information into training.

## 9. Original Model Results Before Feature Selection

Current main test results before file 09/10:

| Model | MAE test | RMSE test |
|---|---:|---:|
| Random Forest | 0.008350 | 0.011222 |
| Rolling Volatility | 0.008830 | 0.011666 |
| Linear Regression | 0.008853 | 0.011803 |
| GNN Correlation Only | 0.009098 | 0.012592 |
| GNN + Relationship | 0.009165 | 0.012579 |
| Full Model | 0.009254 | 0.012761 |
| GNN + News | 0.009622 | 0.013529 |

Interpretation:

- Random Forest was the strongest original model.
- GNN variants did not beat Random Forest.
- Rolling volatility baseline was competitive.

## 10. New Results After Feature Selection and Edge Ablation

Best new result:

| Experiment | Model | Feature count | MAE test | RMSE test |
|---|---|---:|---:|---:|
| price_macro | Random Forest | 30 | 0.008203 | 0.011113 |

Comparison against old best:

| Version | Model | MAE test | RMSE test |
|---|---|---:|---:|
| Old best | Random Forest full features | 0.008350 | 0.011222 |
| New best | Random Forest price_macro | 0.008203 | 0.011113 |

Approximate improvement:

```text
MAE improvement: about 1.76%
RMSE improvement: about 0.97%
```

Key conclusion:

> Feature selection improved the best tabular model, but graph models still do not beat Random Forest.

## 11. How to Explain the Thesis Contribution

Main contribution should not be framed as "GNN beats all baselines." Current results do not support that.

Better framing:

> This thesis builds an end-to-end BDA pipeline combining stock prices, news, company relationships, micro features, macro features, and graph snapshots to study post-news stock volatility. The results show that feature selection is important, full feature/full graph is not necessarily optimal, and Random Forest with price + macro features currently performs best. Graph-based modeling remains useful as a research testbed for information propagation but requires better feature/edge selection and stronger news processing.

## 12. Recommended Honest Interpretation

1. Random Forest is currently the best model.
2. Feature selection matters; `price_macro` beats full 59 features.
3. Macro features are useful in the current setup.
4. Micro features alone do not clearly improve the best result.
5. Full graph is not the best graph configuration.
6. Correlation + same industry is the best graph edge setup among tested variants.
7. News and relationship features may contain noise because:
   - sentiment/category are rule-based,
   - relationship weights are heuristic,
   - same-industry edges dominate the graph,
   - company ownership relationships are limited.
8. GNN-inspired one-hop aggregation has not beaten tabular Random Forest yet.

## 13. Limitations

Important limitations to disclose:

1. News sentiment and category are rule-based.
2. Company relationship weights are partly heuristic.
3. Parent/subsidiary relationships are sparse.
4. Same-industry edges dominate relationship graph.
5. Current graph model is one-hop message passing + MLP, not a full GCN/GAT.
6. Correlation threshold below 0.15 has not been tested without rebuilding snapshots.
7. Current train/validation/test split is time-ordered by snapshot; a stricter split by unique event dates could be added.
8. Some source text/code has Vietnamese encoding artifacts.

## 14. Suggested Next Steps

Highest priority:

1. Use file 09 results in thesis to justify feature selection.
2. Use file 10 results to justify edge ablation.
3. Present Random Forest `price_macro` as the current best model.
4. Avoid claiming GNN is better than all baselines.

Further improvements:

1. Rebuild snapshots with lower correlation threshold such as 0.10.
2. Add stricter split by unique event dates.
3. Manually validate 100-200 news articles for ticker mapping/category/sentiment accuracy.
4. Test lower same-industry weight such as 0.20 or 0.30.
5. Try top-k correlation edges per node instead of threshold-only graph.
6. Consider a proper graph architecture if time allows, such as GCN/GAT/R-GCN.

## 15. One-Minute Explanation for Non-Technical Audience

This project predicts how volatile a stock will be after news appears. Each stock is treated as a node in a graph, and relationships between stocks are edges. The model uses price history, market context, news information, industry, and company relationships.

Initial results showed that using all information was not best. After feature selection, the best model became Random Forest using price and macro features, with MAE 0.008203. Graph experiments showed that using every edge is also not best; correlation plus same-industry edges worked best among graph variants. The main lesson is that selecting the right features and edges is more important than adding all available information.

## 16. Key Files to Upload or Reference

Most important scripts:

```text
scripts/01_build_price_dataset.py
scripts/02_build_news_dataset.py
scripts/03_build_company_relationships.py
scripts/04_build_graph_snapshots.py
scripts/05_train_baselines.py
scripts/06_train_gnn_ablation_models.py
scripts/07_evaluate_results_and_case_studies.py
scripts/08_generate_report_tables_and_figures.py
scripts/09_feature_selection_experiments.py
scripts/10_edge_ablation_experiments.py
scripts/data_config.py
scripts/README_PIPELINE.md
```

Most important outputs:

```text
data/processed/node_feature_schema.json
data/processed/edge_type_map.json
data/processed/graph_snapshot_quality_report.csv
data/processed/relationship_quality_report.csv
data/processed/micro_macro_quality_report.csv
data/processed/baseline_results.csv
data/processed/ablation_results.csv
data/processed/model_comparison_test.csv
data/processed/feature_selection_results.csv
data/processed/feature_selection_configs.csv
data/processed/edge_ablation_results.csv
data/processed/rf_feature_importance.csv
```

## 17. Glossary

| Term | Meaning |
|---|---|
| Feature | input variable used by a model |
| Feature selection | selecting the most useful subset of features |
| Node | one stock ticker in the graph |
| Edge | relationship between two stocks |
| Edge weight | strength of relationship |
| GNN | graph neural network or graph-based learning model |
| One-hop aggregation | aggregate information from direct neighbors only |
| Heuristic | rule or value based on assumption/experience, not learned directly |
| Ablation | remove/change one component to test its contribution |
| MAE | mean absolute error, lower is better |
| RMSE | root mean squared error, lower is better |
| Validation set | data used for model selection |
| Test set | final evaluation data |
| `abs(corr)` | absolute value of Pearson correlation |

