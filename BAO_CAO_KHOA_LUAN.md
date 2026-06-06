# BAO CAO KHOA LUAN - BAN CAP NHAT PIPELINE LOCAL

## De tai

**Du bao bien dong co phieu tu tin tuc tai chinh bang du lieu gia, quan he doanh nghiep va Graph Neural Networks tai thi truong Viet Nam.**

Sinh vien: `[Ten cua ban]`

Giang vien huong dan: `[Ten giang vien]`

Pham vi du lieu chinh: `2022-01-01` den `2025-12-31`

---

## 1. Tong quan bai toan

Du an xay dung pipeline du bao bien dong bat thuong cua co phieu Viet Nam sau khi xuat hien tin tuc tai chinh. Khac voi bai toan du bao gia dong cua, muc tieu o day la du bao muc bien dong/rui ro trong ngan han.

Du lieu su dung gom:

- Du lieu gia lich su cua 118 ma co phieu.
- Tin tuc tai chinh tu Vietstock, tong cong 55.518 bai viet firm-specific.
- Quan he doanh nghiep gom quan he cung tap doan, chuoi gia tri, co dong chung, cum kinh doanh va quan he sector chon loc.
- Graph snapshots theo tung su kien ticker-date.

Bai toan duoc xac dinh la **hoi quy**:

```text
Target: log_abnormal_volatility_5d
```

Y nghia target:

```text
future_realized_volatility_5d
- target_baseline_rolling_vol_20
-> abnormal_volatility_5d
-> log/chuan hoa thanh log_abnormal_volatility_5d
```

Target nay tra loi cau hoi:

> Sau mot su kien tin tuc, co phieu co bien dong bat thuong trong 5 ngay tiep theo hay khong, va muc do bien dong do lon den dau?

---

## 2. Logic khoa hoc cua pipeline

Pipeline hien tai da duoc sap xep lai theo mot quy trinh forecasting local-only:

```text
raw data
-> data cleaning
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

Nguyen tac khoa hoc chinh:

- Split theo `event_trading_date` de tranh leakage thoi gian.
- Baseline tabular duoc train truoc lam moc so sanh.
- GNN duoc danh gia sau baseline, khong ket luan bang cam tinh.
- Residual Hybrid GNN duoc dung de kiem tra graph co bo sung duoc phan tin hieu baseline chua hoc duoc hay khong.
- Tat ca pipeline chinh hien chay local va ghi output vao `outputs/local/` va `outputs/report/`.

Lenh chay pipeline chinh:

```powershell
python scripts\00_run_local_pipeline.py
```

Lenh chi cap nhat metric/bao cao:

```powershell
python scripts\00_run_local_pipeline.py --from-step 09
```

---

## 3. Cau truc output hien tai

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

Nhanh Colab da duoc loai khoi ban pipeline chinh. Toan bo script chinh hien chay local.

---

## 4. Mo ta cac buoc xu ly du lieu

### 4.1. Du lieu gia

Script: `scripts/01_prepare_price_data.py`

Output: `outputs/local/01_price_data/`

Xu ly chinh:

- Kiem tra cot OHLCV bat buoc.
- Loai dong trung theo `date + ticker`.
- Kiem tra loi gia bat thuong: `close <= 0`, `high < low`, `volume <= 0`.
- Tao cac feature gia nhu `log_return`, `rolling_vol_20`, `rolling_vol_60`, `volume_ratio_20`, `abnormal_volume`.
- Tao bao cao chat luong du lieu gia trong `price_quality_report.csv`.

### 4.2. Du lieu tin tuc

Script: `scripts/02_prepare_news_data.py`

Output: `outputs/local/02_news_data/`

Ket qua chinh:

- 55.518 bai viet firm-specific.
- 57.899 ticker mentions.
- 343 aliases ten doanh nghiep.
- Tin tuc duoc gan voi ma co phieu va ngay giao dich su kien.

Nhom nhan tin tuc:

- `earnings`
- `leadership`
- `ma_ownership`
- `debt_bond`
- `market_industry`
- `capital_issuance`
- `dividend`
- `project_contract`
- `legal_regulatory`
- `other`

Sentiment gom:

- `positive`
- `neutral`
- `negative`

### 4.3. Kiem dinh nhan tin tuc

Script: `scripts/03_train_news_labeler.py`

Mo hinh:

```text
TF-IDF word/character n-grams + Logistic Regression
```

Ket qua holdout:

```text
Category   Accuracy=0.9060  Macro F1=0.8859
Sentiment  Accuracy=0.9662  Macro F1=0.8619
```

Y nghia:

> Nhan tin tuc co chat luong du de dung lam feature giai thich trong mo hinh du bao, nhung khong dam bao rieng tin tuc se du manh de du bao volatility.

### 4.4. Quan he doanh nghiep

Script: `scripts/04_prepare_company_relationships.py`

Output: `outputs/local/04_company_relationships/`

Graph quan he gom cac lop:

- `business_cluster`
- `value_chain`
- `strategic_ecosystem`
- `common_owner`
- `same_group`
- `parent/subsidiary`

Luu y khoa hoc:

> Khong dua toan bo same-industry graph vao graph chinh vi graph qua day co the tao nhieu, lam GNN kho hoc tin hieu that.

---

## 5. Xay dung graph dataset va target

Script: `scripts/05_build_event_graph_dataset.py`

Output: `outputs/local/05_event_graph_dataset/`

Ket qua:

```text
Graph snapshots hop le: 23.705
Don vi mau: 1 ticker + 1 event_trading_date
Target: log_abnormal_volatility_5d
Edge types: 6
```

Xu ly missing/outlier:

- Bo su kien khong du 20 ngay lich su de tao feature ngan han.
- Bo su kien khong du 252 ngay de tinh correlation.
- Bo su kien khong du 5 ngay tuong lai de tao target.
- Thay `NaN/inf` trong feature dau vao bang 0.
- Clip feature dau vao de giam outlier:

```text
log_return lags: [-0.30, 0.30]
rolling_vol_20: [0.00, 0.20]
volume_ratio_20: [0.00, 10.00]
trading_value_ratio_20: [0.00, 10.00]
```

Khong clip target vi volatility cao la tin hieu can du bao.

Edge types trong graph:

- `corr_positive_top10`
- `corr_negative_top5`
- `ownership`
- `value_chain_curated`
- `sector_top5_only`
- `news_co_mention`

Feature quan trong:

- Feature gia ngan han.
- Shock features.
- News category/sentiment features.
- Neighbor exposure features.
- PhoBERT PCA title embedding neu file `news_title_embedding_pca.csv` ton tai.

---

## 6. Split du lieu

Du lieu duoc chia theo `event_trading_date`, khong chia ngau nhien theo dong.

Y nghia:

> Tat ca mau cung ngay chi nam trong mot split, tranh viec thong tin cung ngay bi ro ri giua train va test.

Ket qua split hien tai:

```text
Train:      518 ngay
Validation: 111 ngay
Test:       111 ngay
Test size:  3.638 samples
```

---

## 7. Mo hinh da su dung

### 7.1. Baseline tabular

Script: `scripts/06_train_baseline_models.py`

Mo hinh:

- Rolling Volatility.
- Linear Regression.
- Random Forest.
- GNN Correlation Only baseline.

Vai tro:

> Baseline cho biet neu chi dung feature gia/news dang bang bang thi mo hinh dat duoc den dau. GNN chi co y nghia neu bo sung duoc tin hieu vuot qua baseline nay.

### 7.2. GNN ablation

Script: `scripts/07_train_gnn_ablation_models.py`

Cau hinh:

- `GNN Correlation Only`
- `GNN + News`
- `GNN + Relationship`
- `Full Model`

Muc dich:

> Kiem tra tung nhom input/edge co dong gop the nao.

### 7.3. Tuned TopK Graph MLP

Script: `scripts/08_tune_selected_gnn.py`

Cai tien:

- Top-k correlation neighbors.
- StandardScaler cho X/y.
- SmoothL1 loss.
- AdamW.
- Gradient clipping.
- Early stopping theo validation MAE.
- Interaction features: self, neighbor, self-neighbor, self*neighbor.

### 7.4. Hybrid MLP-GAT

Script: `scripts/12_train_hybrid_mlp_gat.py`

Y tuong:

> Ket hop self node feature voi neighbor attention theo edge type.

Ket qua hien tai cho thay Hybrid MLP-GAT thuan chua vuot baseline.

### 7.5. Residual Hybrid GNN

Script: `scripts/14_train_residual_hybrid_gnn.py`

Day la huong GNN chinh hien tai.

Cong thuc:

```text
y_pred_final = y_pred_tabular_baseline + alpha * graph_residual_pred
```

Trong do:

- Ridge Regression hoac HistGradientBoostingRegressor hoc phan tin hieu tabular/gia co ban.
- GNN hoc phan sai so con lai cua baseline.
- `alpha` duoc chon theo validation de tranh graph residual lam xau du bao.
- Activation trong Custom GNN duoc doi tu ReLU sang LeakyReLU(0.01), giu hidden_dim = 96 va attention_dropout = 0.45 de lam doi chung.

Y nghia khoa hoc:

> Mo hinh nay khong ep GNN hoc lai toan bo target tu dau. Thay vao do, no kiem tra graph co giai thich duoc phan bien dong ma baseline tabular chua giai thich duoc hay khong.

---

## 8. Ket qua thuc nghiem moi nhat

Nguon bang:

```text
outputs/report/11_regression_metrics/r2_metrics_test.csv
```

Ket qua test-set:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| HistGBR Residual Hybrid GNN (R2 tuned) | 0.533599 | **0.646247** | **0.136238** |
| Ridge Residual Hybrid GNN (R2 tuned) | 0.535577 | 0.646839 | 0.134655 |
| Linear Regression | 0.529750 | 0.650514 | 0.124795 |
| Ridge Regression | 0.531378 | 0.651081 | 0.123270 |
| Random Forest | 0.533727 | 0.652021 | 0.120735 |
| HistGBR | 0.530908 | 0.654312 | 0.114546 |
| Ridge Residual Hybrid GNN (MAE shrinkage) | 0.530682 | 0.656074 | 0.109769 |
| HistGBR Residual Hybrid GNN (MAE shrinkage) | 0.531180 | 0.659328 | 0.100918 |
| Rolling Volatility | 0.602609 | 0.703597 | -0.023868 |
| GNN Correlation Only | 0.619792 | 0.747527 | -0.155713 |
| GNN + Relationship | 0.621335 | 0.748015 | -0.157223 |
| GNN Correlation Only (Ablation) | 0.622127 | 0.749514 | -0.161865 |
| Hybrid MLP-GAT | 0.615900 | 0.754033 | -0.175919 |
| GNN + News | 0.645702 | 0.782420 | -0.266124 |
| Full Model | 0.646429 | 0.790028 | -0.290868 |

Nhan xet:

- Neu xet MAE, Linear Regression van tot nhat voi MAE = 0.529750; cac bien the residual hybrid moi chua vuot MAE.
- Neu xet RMSE va R2, `HistGBR Residual Hybrid GNN (R2 tuned)` tot nhat voi RMSE = 0.646247 va R2 = 0.136238, cao hon Linear Regression R2 = 0.124795.
- HistGBR alone khong tot hon Linear Regression ve R2, nhung khi ket hop residual GNN thi cho R2 tot nhat. Dieu nay cho thay graph residual co dong gop nho nhung co that theo tieu chi R2/RMSE.
- GNN thuan va Hybrid MLP-GAT thuan van yeu hon baseline.
- Ket qua khong nen trinh bay la GNN thang tuyet doi tren moi metric; nen trinh bay trung thuc rang GNN cai thien R2/RMSE nhung chua cai thien MAE.

Ket luan bao ve nen dung:

> Residual Hybrid GNN cho thay thong tin graph co dong gop bo sung vao bai toan du bao volatility theo RMSE/R2. Tuy nhien, MAE van chua vuot Linear Regression, nen ket luan can nhan manh vao residual learning va phan tich sai so thay vi khang dinh GNN thang tren moi tieu chi.

---

## 9. Phan tich vi sao Residual Hybrid GNN hop ly

Trong bai toan tai chinh, feature gia ngan han thuong rat manh. Linear Regression co the hoc nhanh cac quan he truc tiep giua feature gia va volatility. Neu bat GNN hoc toan bo target tu dau, GNN de bi:

- Graph nhieu.
- Target kho hoc.
- News sparse.
- Quan he doanh nghiep khong phai luc nao cung anh huong truc tiep den volatility ngan han.

Residual Hybrid GNN giai quyet bang cach:

```text
baseline hoc tin hieu de hoc
GNN hoc phan baseline con sai
```

Do do mo hinh phu hop hon voi cau hoi nghien cuu:

> Quan he doanh nghiep va graph co giup giai thich phan bien dong ma feature tabular/gia chua giai thich duoc hay khong?

---

## 10. Case study va report assets

Da cap nhat cac output:

```text
outputs/report/09_model_evaluation/
outputs/report/10_report_assets/
outputs/report/11_regression_metrics/
```

Noi dung da tao:

- Bang metric test-set.
- Bang sai so theo ticker.
- Bang sai so theo news category.
- Bang sai so theo ngay su kien.
- 45 case studies.
- Bieu do target distribution.
- Bieu do volatility raw vs baseline.
- Bieu do abnormal volatility theo ngay.
- Bieu do so sanh MAE model.
- Scatter plot du bao vs thuc te.

---

## 11. Cac thu nghiem chua nen ket luan chinh

### 11.1. GNN thuan

GNN thuan, GNN + News, GNN + Relationship va Full Model hien chua vuot baseline. Cac ket qua nay van co gia tri vi chung chung minh:

- Graph khong the dua vao mot cach may moc.
- Can lam graph thua va dung residual learning.
- News/title embedding co the them nhieu neu khong co feature selection phu hop.

### 11.2. PhoBERT

PhoBERT hien duoc dung de tao embedding title:

```text
vinai/phobert-base
768 dimensions -> PCA 64 dimensions
```

Tuy nhien PhoBERT khong co metric rieng. Metric chi co y nghia sau khi:

```text
PhoBERT embedding
-> rebuild graph
-> train model
-> compare with no-PhoBERT setting
```

### 11.3. Tuned TopK Graph MLP

Ket qua TopK Graph MLP kha gan baseline, nhung hien chua duoc dua lam ket luan chinh vi output chua co prediction day du de tinh R2 trong bang tong hop.

---

## 12. Ket luan chinh

1. Bai toan cua du an la hoi quy du bao `log_abnormal_volatility_5d`, dai dien cho bien dong bat thuong 5 ngay sau tin tuc.

2. Pipeline hien da duoc chuan hoa thanh local-only, co runner `scripts/00_run_local_pipeline.py`, giup tai lap ket qua ro rang.

3. Baseline tabular, dac biet Linear Regression, la doi thu rat manh vi feature gia ngan han co tin hieu lon.

4. GNN thuan chua vuot baseline, cho thay graph va news neu dua truc tiep vao model co the gay nhieu.

5. Residual Hybrid GNN la huong phu hop nhat voi de tai. Mo hinh nay dung baseline de hoc tin hieu co ban, sau do dung GNN hoc phan residual tu graph.

6. Ket qua moi nhat cho thay Residual Hybrid GNN cai thien RMSE/R2 nhung chua cai thien MAE:

```text
MAE: 0.533599 > Linear Regression 0.529750
R2:  0.136238 > Linear Regression 0.124795
RMSE: 0.646247 < Linear Regression 0.650514
```

7. Ket luan khoa hoc nen trinh bay:

> Graph Neural Networks co dong gop tot nhat khi duoc thiet ke theo huong residual hybrid, bo sung tin hieu quan he doanh nghiep cho baseline tabular, thay vi thay the baseline hoan toan.

---

## 13. Huong phat trien

- Kiem dinh PhoBERT bang ablation co/khong PhoBERT.
- Thu graph thua hon theo tung nganh va tung quan he doanh nghiep.
- Thiet ke target classification cho abnormal volatility jump.
- Them macro features nhu lai suat, CPI, VNIndex volatility.
- Thu R-GCN/GAT co regularization manh hon.
- Bao cao confidence interval hoac bootstrap test cho improvement nho giua baseline va residual hybrid.
