# Kết quả Hybrid MLP-GAT

Ngày chạy: 2026-06-04

## Cấu hình

Model mới:

```text
Hybrid MLP-GAT
```

Ý tưởng:

- Self branch: MLP học đặc trưng của node mục tiêu.
- Neighbor branch: GAT-style attention trên hàng xóm đi vào node mục tiêu.
- Attention dùng:
  - neighbor features
  - edge type embedding
  - edge weight
- Output head dùng:
  - self embedding
  - neighbor attention embedding
  - |self - neighbor|
  - self * neighbor

Loss:

```text
weighted MAE
weight = 1 nếu target bình thường
weight = 2 nếu target > Q75
weight = 4 nếu target > Q90
```

Target:

```text
log_abnormal_volatility_5d
```

Graph:

```text
corr_positive_top10
corr_negative_top5
ownership
value_chain_curated
sector_top5_only
news_co_mention
```

## Kết quả test

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.529750 | 0.650514 | 0.124795 |
| Random Forest | 0.533727 | 0.652021 | 0.120735 |
| GNN + News | 0.584926 | 0.715488 | -0.058770 |
| Hybrid MLP-GAT | 0.589605 | 0.731103 | -0.105488 |
| Full Model | 0.592417 | 0.724355 | -0.085175 |
| GNN Correlation Only | 0.619792 | 0.747527 | -0.155713 |
| GNN + Relationship | 0.621335 | 0.748015 | -0.157223 |

## Nhận xét

Hybrid MLP-GAT tốt hơn:

- `GNN Correlation Only`
- `GNN + Relationship`
- `Full Model` nếu xét MAE

Nhưng Hybrid MLP-GAT chưa tốt hơn:

- `GNN + News`
- `Linear Regression`
- `Random Forest`

## Kết luận

Hướng Hybrid MLP-GAT là hợp lý về mặt kiến trúc, nhưng kết quả hiện tại chưa thắng baseline.

Điểm quan trọng:

- Attention trên toàn bộ neighbor graph chưa đủ tốt nếu edge relationship vẫn còn nhiễu.
- `GNN + News` vẫn là GNN tốt nhất hiện tại.
- Baseline tabular vẫn thắng, cho thấy bài toán hiện còn bị chi phối bởi feature giá/shock trực tiếp hơn là lan truyền graph.

## Hướng tiếp theo

Không nên tăng độ phức tạp model ngay. Nên thử trước:

1. Hybrid MLP-GAT chỉ dùng `corr_positive_top10 + news_co_mention`, bỏ ownership/value-chain khỏi attention.
2. Hybrid MLP-GAT có edge-type ablation riêng.
3. Chuẩn hóa feature và target trong Hybrid MLP-GAT như tuned GNN trước đó.
4. Chỉ sau đó mới thử R-GCN hoặc temporal LSTM-GNN.
