# Kết quả sau khi sửa target và làm graph thưa

Ngày chạy: 2026-06-04

Phạm vi đã làm đúng theo yêu cầu:

- Sửa target: không dự báo volatility thô nữa.
- Làm graph thưa lại: giảm correlation graph bằng top-k.
- Chưa thêm feature shock.
- Chưa thêm neighbor exposure mới.
- Chưa thử model mới như GAT/R-GCN/LSTM-GNN.

## 1. Target mới

Target cũ:

```text
y = future_realized_volatility_5d
```

Target mới:

```text
y = abnormal_volatility_5d
  = future_realized_volatility_5d - rolling_vol_20_t_minus_1
```

Ý nghĩa:

Mô hình không còn học mức volatility thô, mà học phần volatility tương lai cao/thấp bất thường so với volatility gần đây của chính cổ phiếu đó.

Phân phối target mới trên toàn bộ snapshot:

| Metric | Raw future vol | Baseline rolling vol | Abnormal vol |
|---|---:|---:|---:|
| Mean | 0.018087 | 0.019363 | -0.001276 |
| Median | 0.015164 | 0.018039 | -0.002337 |
| Std | 0.011623 | 0.008798 | 0.011738 |
| 95% | 0.040637 | 0.036531 | 0.019282 |
| Max | 0.126447 | 0.063952 | 0.100755 |

## 2. Graph thưa lại

Graph cũ:

```text
Giữ mọi cạnh price_correlation nếu |corr| >= 0.15
Mean edges/snapshot ≈ 13,777
```

Graph mới:

```text
Mỗi node chỉ giữ tối đa top-10 correlation neighbors nếu |corr| >= 0.15
Mean edges/snapshot = 2,666
```

Mức giảm:

```text
13,777 -> 2,666 edges/snapshot
Giảm khoảng 80.6%
```

Phân bố edge mới:

| Edge type | Mean edges/snapshot |
|---|---:|
| price_correlation | 1,169.84 |
| business_cluster | 930.00 |
| value_chain | 398.00 |
| common_owner | 56.00 |
| strategic_ecosystem | 55.94 |
| same_group | 50.00 |
| parent/subsidiary | 5.32 |
| news_co_mention | 1.39 |

## 3. Kết quả model hiện có

Lưu ý: đây vẫn là model cũ, chưa thêm feature shock/neighbor exposure và chưa đổi sang GAT/R-GCN.

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.008584 | 0.011489 | 0.119729 |
| Random Forest | 0.008604 | 0.011510 | 0.116404 |
| GNN Correlation Only | 0.009062 | 0.012141 | 0.016871 |
| GNN + News | 0.009110 | 0.012171 | 0.012011 |
| Full Model | 0.009219 | 0.012431 | -0.030586 |
| GNN + Relationship | 0.009242 | 0.012431 | -0.030619 |
| Rolling Volatility | 0.024312 | 0.028498 | -4.416501 |

Rolling Volatility không còn là baseline hợp lệ cho target mới, vì nó dự báo raw volatility trong khi target mới là abnormal volatility.

## 4. So với trước khi sửa

| Model | Trước | Sau | Nhận xét |
|---|---:|---:|---|
| GNN Correlation Only MAE | 0.009101 | 0.009062 | Cải thiện nhẹ |
| GNN + News MAE | 0.009240 | 0.009110 | Cải thiện rõ hơn |
| GNN + Relationship MAE | 0.009141 | 0.009242 | Kém hơn |
| Full Model MAE | 0.009125 | 0.009219 | Kém hơn |

## 5. Kết luận

Sửa target và làm graph thưa là đúng hướng, nhưng chưa đủ để GNN thắng baseline.

Tín hiệu tích cực:

- Graph nhẹ hơn và ít nhiễu hơn rất nhiều.
- `GNN Correlation Only` cải thiện nhẹ.
- `GNN + News` cải thiện so với trước.

Vấn đề còn lại:

- Relationship/full graph vẫn chưa tốt, nghĩa là edge quan hệ chưa được chuyển thành feature/exposure đủ mạnh.
- Feature hiện tại vẫn chưa có shock feature và neighbor exposure đúng nghĩa.
- Model hiện tại vẫn là one-hop weighted message passing + MLP.

Bước tiếp theo nên làm:

1. Thêm feature shock: return shock, volatility shock, volume shock, negative news shock.
2. Thêm neighbor exposure: neighbor volatility shock, neighbor negative news exposure, ownership/relationship exposure.
3. Sau đó mới thử GAT/R-GCN/LSTM-GNN.
