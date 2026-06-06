# KLTN GNN Stock Volatility Forecasting

## 1. Muc tieu nghien cuu

De tai xay dung pipeline du bao bien dong co phieu tai thi truong Viet Nam bang:

- Du lieu gia lich su.
- Tin tuc tai chinh.
- Quan he doanh nghiep.
- Graph Neural Networks.

Bai toan chinh la hoi quy:

```text
Du bao log_abnormal_volatility_5d
```

Y nghia target:

```text
future realized volatility 5 ngay
- baseline rolling volatility 20 ngay
-> log transform / chuan hoa de giam nhieu va outlier
```

Day la bai toan du bao bien dong bat thuong sau su kien tin tuc, khong phai du bao gia dong cua.

## 2. Logic khoa hoc cua pipeline

Pipeline chinh duoc sap xep theo mach cua mot bai toan forecasting:

```text
raw data
-> cleaning
-> feature engineering
-> target construction
-> time-based split
-> graph construction
-> baseline models
-> GNN models
-> residual hybrid GNN
-> evaluation
-> report assets
```

Nguyen tac quan trong:

- Split theo ngay de tranh leakage thoi gian.
- Baseline tabular duoc train truoc de co moc so sanh.
- GNN duoc danh gia sau baseline, khong thay the baseline bang cam tinh.
- Residual Hybrid GNN kiem tra graph co bo sung duoc phan tin hieu baseline chua hoc duoc khong.
- Tat ca output cua pipeline chinh chay local duoc ghi vao `outputs/local/` va `outputs/report/`.

## 3. Cau truc script chuan

| Step | Script | Vai tro |
|---:|---|---|
| 00 | `scripts/00_run_local_pipeline.py` | Chay toan bo pipeline local theo thu tu |
| 01 | `scripts/01_prepare_price_data.py` | Lam sach gia va tao feature gia |
| 02 | `scripts/02_prepare_news_data.py` | Lam sach tin tuc, map ticker, tao news events |
| 03 | `scripts/03_train_news_labeler.py` | Kiem dinh/giai thich nhan tin tuc |
| 04 | `scripts/04_prepare_company_relationships.py` | Tao graph quan he doanh nghiep |
| 05 | `scripts/05_build_event_graph_dataset.py` | Tao graph snapshots, target, shock/exposure features |
| 06 | `scripts/06_train_baseline_models.py` | Train Rolling Volatility, Linear Regression, Random Forest, Corr-GNN |
| 07 | `scripts/07_train_gnn_ablation_models.py` | Train GNN ablation theo nhom graph/feature |
| 08 | `scripts/08_tune_selected_gnn.py` | Tune TopK Graph MLP |
| 12 | `scripts/12_train_hybrid_mlp_gat.py` | Train Hybrid MLP-GAT |
| 14 | `scripts/14_train_residual_hybrid_gnn.py` | Train Residual Hybrid GNN |
| 09 | `scripts/09_evaluate_and_report.py` | Tinh metrics, case study, bang va bieu do bao cao |

## 4. Output layout local-only

```text
outputs/
  local/
    01_price_data/
    02_news_data/
    03_news_labeler/
    04_company_relationships/
    05_event_graph_dataset/
    06_baseline_models/
    07_gnn_ablation_models/
    08_tuned_gnn/
    12_hybrid_mlp_gat/
    14_residual_hybrid_gnn/
  report/
    09_model_evaluation/
    10_report_assets/
    11_regression_metrics/
```

Pipeline chinh hien khong con nhanh Colab; toan bo script trong `scripts/` chay local.

## 5. Cach chay local

Chay tu thu muc goc project:

```powershell
python scripts\00_run_local_pipeline.py
```

Chay tu mot buoc bat ky:

```powershell
python scripts\00_run_local_pipeline.py --from-step 05
```

Chi refresh metric/report sau khi da co output model:

```powershell
python scripts\00_run_local_pipeline.py --from-step 09
```

Neu muon bo qua cac model GNN nang va chi refresh data/baseline/report:

```powershell
python scripts\00_run_local_pipeline.py --skip-heavy
```

## 6. Ket qua hien tai can bao cao

Bang ket qua chinh nam o:

```text
outputs/report/11_regression_metrics/r2_metrics_test.csv
```

Sau khi them Residual Hybrid GNN va nhanh HistGBR, ket qua tot nhat hien tai co 2 cach doc:

- Theo MAE: Linear Regression van la moc tot nhat hien tai.
- Theo RMSE/R2: `HistGBR Residual Hybrid GNN (R2 tuned)` dang co RMSE thap hon va R2 cao hon Linear Regression.

Dien giai khoa hoc nen dung:

```text
GNN khong hoc lai toan bo target tu dau.
Ridge/HistGBR hoc tin hieu tabular/gia co ban.
Graph model hoc residual, tuc phan sai so con lai cua baseline.
Neu residual hybrid cai thien RMSE/R2, graph co dong gop thong tin bo sung theo tieu chi giai thich phuong sai.
```

## 7. Luu y khoa hoc

- Khong nen ket luan "GNN thuan tuyet doi tot hon baseline" neu chi so khong ung ho.
- Nen ket luan "Residual Hybrid GNN khai thac thong tin graph de bo sung cho baseline tabular".
- PhoBERT embedding la feature text, khong co metric rieng neu chua train lai model va so sanh voi cau hinh khong PhoBERT.
- Graph nhieu, target kho va news sparse la ba nguon rui ro chinh cua bai toan.
