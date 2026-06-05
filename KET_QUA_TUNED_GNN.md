# Ket qua tuned GNN sau khi gop ticker-date

## 1. Thay doi quan trong trong data input

Don vi mau da duoc sua tu:

```text
1 sample = 1 article / ticker mention
```

thanh:

```text
1 sample = 1 ticker + 1 event_trading_date
```

Neu mot ma co nhieu tin trong cung ngay, cac tin duoc gop thanh mot event duy nhat. Dieu nay giup tranh nhan ban target va lam bai toan du bao hop ly hon ve mat tai chinh.

Ket qua sau khi gop:

| Chi tieu | Gia tri |
|---|---:|
| Ticker-date events dau vao | 31.650 |
| Graph snapshots hop le | 23.705 |
| Duplicate ticker-date groups | 0 |
| Mean news_count/event | 1.89 |
| Max news_count/event | 31 |

## 2. Tuned GNN

Tuned GNN van dung cau truc top-k correlation va interaction features:

```text
[self, neighbor, self - neighbor, self * neighbor]
```

Voi `top_30_rf`, input dimension la 120.

## 3. Ket qua baseline moi

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.008584 | 0.011489 | 0.124 |
| Random Forest | 0.008608 | 0.011484 | 0.125 |
| Rolling Volatility | 0.009462 | 0.012256 | 0.004 |
| GNN Correlation Only | 0.009101 | 0.012301 | -0.004 |
| GNN + Relationship | 0.009141 | 0.012344 | -0.011 |
| Full Model | 0.009125 | 0.012440 | -0.027 |
| GNN + News | 0.009240 | 0.012594 | -0.052 |

## 4. Ket qua tuned GNN moi

| Variant | Hidden dim | Dropout | MAE | RMSE |
|---|---:|---:|---:|---:|
| `scaled_h96_d015` | 96 | 0.15 | 0.008827 | 0.012185 |
| `scaled_h64_d010` | 64 | 0.10 | 0.008845 | 0.012076 |
| `scaled_h64_d000` | 64 | 0.00 | 0.008851 | 0.012035 |
| `scaled_h128_d020` | 128 | 0.20 | 0.008888 | 0.012208 |

Cau hinh tuned GNN tot nhat:

```text
variant = scaled_h96_d015
feature_set = top_30_rf
top_k_corr_neighbors = 20
hidden_dim = 96
dropout = 0.15
MAE = 0.008827
RMSE = 0.012185
```

## 5. Ket luan

Sau khi sua logic data input, ket qua danh gia dang tin cay hon vi khong con nhan ban target theo nhieu bai bao cung ngay.

Tuy nhien, tuned GNN van chua vuot baseline tabular:

```text
Linear Regression MAE = 0.008584
Tuned GNN MAE        = 0.008827
```

Ket luan nen trinh bay trong khoa luan:

```text
Gop tin theo ticker-date lam pipeline dung logic tai chinh hon.
GNN co cai thien so voi GNN co ban, nhung target volatility tho van bi dac trung gia ngan han chi phoi manh.
Baseline tuyen tinh/tabular van la moc tot nhat tren test set hien tai.
```
