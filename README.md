# Dự báo biến động cổ phiếu từ tin tức tài chính, dữ liệu giá, quan hệ doanh nghiệp và GNN

## 1. Mục tiêu

Dự án xây dựng pipeline dự báo biến động cổ phiếu tại thị trường Việt Nam bằng cách kết hợp:

- Dữ liệu giá cổ phiếu lịch sử.
- Tin tức tài chính.
- Quan hệ doanh nghiệp.
- Graph Neural Networks.

Bài toán chính:

```text
Dự báo future_realized_volatility_5d
```

## 2. Logic pipeline chuẩn

Pipeline đã được sắp xếp lại theo đúng mạch của một bài toán dự báo:

```text
01 Data preparation
02 News preparation
03 News label validation
04 Relationship preparation
05 Forecasting dataset / graph construction
06 Baseline training
07 GNN training
08 Fine-tuning selected GNN
09 Evaluation and case studies
10 Report tables and figures
11 Final regression metrics
```

## 3. Scripts chính

| Bước | Script | Vai trò |
|---:|---|---|
| 00 | `scripts/project_config.py` | Cấu hình phạm vi dữ liệu |
| 01 | `scripts/01_prepare_price_data.py` | Làm sạch và tạo feature giá |
| 02 | `scripts/02_prepare_news_data.py` | Làm sạch tin tức, ticker mentions, nhãn rule/ML |
| 03 | `scripts/03_train_news_labeler.py` | Kiểm định mô hình gán nhãn tin tức từ nhãn tay |
| 04 | `scripts/04_prepare_company_relationships.py` | Xây dựng cạnh quan hệ doanh nghiệp |
| 05 | `scripts/05_build_event_graph_dataset.py` | Tạo graph snapshot và target dự báo |
| 06 | `scripts/06_train_baseline_models.py` | Train Rolling Volatility, Linear Regression, Random Forest, Corr-GNN |
| 07 | `scripts/07_train_gnn_ablation_models.py` | Train GNN ablation: correlation, news, relationship, full model |
| 08 | `scripts/08_tune_selected_gnn.py` | Fine-tune TopK Graph MLP |
| 09 | `scripts/09_evaluate_models_and_cases.py` | Đánh giá lỗi, ticker/category/date, case studies |
| 10 | `scripts/10_generate_report_assets.py` | Tạo bảng và hình báo cáo |
| 11 | `scripts/11_compute_regression_metrics.py` | Tính MAE, RMSE, R2 |

## 4. Output chính

| Bước | Output |
|---:|---|
| 01 | `outputs/local/01_price_data/` |
| 02 | `outputs/local/02_news_data/` |
| 03 | `outputs/local/03_news_labeler/` |
| 04 | `outputs/local/04_company_relationships/` |
| 05 | `outputs/local/05_event_graph_dataset/` |
| 06 | `outputs/local/06_baseline_models/` |
| 07 | `outputs/colab/07_gnn_ablation_models/` |
| 08 | `outputs/colab/08_tuned_gnn/` |
| 09 | `outputs/report/09_model_evaluation/` |
| 10 | `outputs/report/10_report_assets/` |
| 11 | `outputs/report/11_regression_metrics/` |
| 12 | `outputs/colab/12_hybrid_mlp_gat/` |

## 5. Cách chạy

```powershell
# VSCode/local
python scripts\01_prepare_price_data.py
python scripts\02_prepare_news_data.py
python scripts\03_train_news_labeler.py
python scripts\04_prepare_company_relationships.py
python scripts\05_build_event_graph_dataset.py
python scripts\06_train_baseline_models.py

# Google Colab/GPU
python scripts\07_train_gnn_ablation_models.py
python scripts\08_tune_selected_gnn.py
python scripts\12_train_hybrid_mlp_gat.py

# VSCode/report
python scripts\09_evaluate_models_and_cases.py
python scripts\10_generate_report_assets.py
python scripts\11_compute_regression_metrics.py
python scripts\13_plot_outputs_and_volatility.py
```

## 6. Dataset sau khi gộp ticker-date

Đơn vị mẫu hiện tại đã được sửa thành:

```text
1 sample = 1 ticker + 1 event_trading_date
```

Nếu một mã có nhiều tin trong cùng ngày, các tin được gộp thành một event duy nhất.

| Chỉ tiêu | Giá trị |
|---|---:|
| Ticker-date events đầu vào | 31.650 |
| Graph snapshots hợp lệ | 23.705 |
| Duplicate ticker-date groups | 0 |
| Mean news_count/event | 1.89 |
| Max news_count/event | 31 |

## 7. Kết quả hiện tại

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.008584 | 0.011489 | 0.124 |
| Random Forest | 0.008608 | 0.011484 | 0.125 |
| Tuned TopK Graph MLP | 0.008827 | 0.012185 | Chưa lưu prediction để tính |
| GNN Correlation Only | 0.009101 | 0.012301 | -0.004 |
| GNN + Relationship | 0.009141 | 0.012344 | -0.011 |
| Rolling Volatility | 0.009462 | 0.012256 | 0.004 |

Kết luận hiện tại:

```text
Sau khi gộp ticker-date, dữ liệu đánh giá hợp lý hơn.
Linear Regression vẫn có MAE tốt nhất, còn Random Forest có R2 nhỉnh hơn.
GNN chưa vượt baseline trên target volatility thô.
Relationship graph đã được chuyển từ graph chủ yếu `same_industry` sang graph đa tầng gọn hơn gồm `business_cluster`, `value_chain`, `strategic_ecosystem`, `common_owner`, `same_group` và parent/subsidiary edges.
```

## 8. Ghi chú về archive

Các thí nghiệm không thuộc pipeline chuẩn đã được chuyển vào:

```text
archive/extra_scripts/
archive/extra_outputs/
archive/reports/
```

Chúng được giữ để truy vết, nhưng không nằm trong bản chuẩn dùng để nộp khóa luận.
