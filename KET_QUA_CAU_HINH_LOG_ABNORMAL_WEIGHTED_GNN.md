# Kết quả cấu hình log abnormal volatility + graph layers + weighted MAE

Ngày chạy: 2026-06-04

## Cấu hình đã triển khai

### Target

```text
log_abnormal_volatility_5d
= sign(abnormal_volatility_5d) * log1p(100 * abs(abnormal_volatility_5d))

abnormal_volatility_5d
= future_realized_volatility_5d - rolling_vol_20_t_minus_1
```

Lưu ý: MAE/RMSE ở báo cáo này nằm trên thang `log_abnormal_volatility_5d`, không so trực tiếp 1-1 với MAE volatility thô trước đây.

### Graph

Edge layers hiện tại:

| Edge layer | Mean edges/snapshot |
|---|---:|
| `corr_positive_top10` | 1,169.13 |
| `sector_top5_only` | 546.99 |
| `value_chain_curated` | 398.00 |
| `ownership` | 167.25 |
| `corr_negative_top5` | 2.01 |
| `news_co_mention` | 1.39 |

Tổng số cạnh trung bình:

```text
2,284.78 edges/snapshot
```

So với graph rất dày trước đây khoảng `13,777 edges/snapshot`, graph mới giảm khoảng `83.4%`.

### Feature

Đã thêm 5 shock features:

```text
return_shock
volatility_shock
volume_shock
negative_news_count
sector_shock
```

Feature count:

```text
66 -> 71 node features
```

### Loss

GNN hiện dùng weighted MAE:

```text
weight = 1 nếu target bình thường
weight = 2 nếu target > Q75 của train set
weight = 4 nếu target > Q90 của train set
loss = mean(weight * abs(y_pred - y_true))
```

Áp dụng cho:

- `GNN Correlation Only`
- `GNN + News`
- `GNN + Relationship`
- `Full Model`

Baseline sklearn vẫn giữ cách train mặc định để làm mốc so sánh.

## Kết quả test

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.530268 | 0.651394 | 0.122424 |
| Random Forest | 0.533938 | 0.653191 | 0.117576 |
| Rolling Volatility | 0.602609 | 0.703597 | -0.023868 |
| GNN + News | 0.592081 | 0.724390 | -0.085279 |
| Full Model | 0.591559 | 0.730243 | -0.102889 |
| GNN Correlation Only | 0.619792 | 0.747527 | -0.155713 |
| GNN + Relationship | 0.621335 | 0.748015 | -0.157223 |

## Nhận xét

Điểm tốt:

- Graph đã thưa hơn rất nhiều và đúng logic hơn.
- Trong nhóm GNN, `Full Model` và `GNN + News` đã tốt hơn `GNN Correlation Only`.
- Điều này cho thấy shock/news features bắt đầu có tác dụng.

Điểm chưa tốt:

- GNN vẫn chưa thắng Linear Regression/Random Forest.
- `corr_negative_top5` gần như không có nhiều cạnh vì rất ít cặp có tương quan âm mạnh dưới `-0.15`.
- `GNN + Relationship` chưa tốt, nghĩa là ownership/value-chain edge chưa đủ nếu không có neighbor exposure chuyên biệt hơn.

Kết luận:

Cấu hình này là bước đúng hơn về mặt phương pháp luận, nhưng chưa phải cấu hình thắng baseline. Bước tiếp theo nên tập trung vào `neighbor_exposure` đúng nghĩa theo từng layer, ví dụ:

- positive-corr neighbor shock
- negative-corr hedge/anti-corr exposure
- ownership neighbor volatility shock
- value-chain upstream/downstream shock
- sector shock từ top-5 sector neighbors

Sau đó mới nên thử GAT/R-GCN/LSTM-GNN.
