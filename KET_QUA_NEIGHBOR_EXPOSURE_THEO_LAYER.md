# Kết quả sau khi thêm neighbor exposure theo từng graph layer

Ngày chạy: 2026-06-04

## Thay đổi đã làm

Thay vì chỉ có một biến `neighbor_exposure` chung, dataset hiện có 6 exposure features:

```text
pos_corr_neighbor_exposure
neg_corr_neighbor_exposure
ownership_neighbor_exposure
value_chain_neighbor_exposure
sector_neighbor_exposure
news_neighbor_exposure
```

Ý nghĩa:

- Mỗi node nhận shock trung bình có trọng số từ các node hàng xóm đi vào nó.
- Exposure được tách riêng theo graph layer, không gộp tất cả cạnh vào một biến.
- Target vẫn là `log_abnormal_volatility_5d`.
- GNN vẫn dùng weighted MAE.

## Dataset sau khi rebuild

| Metric | Giá trị |
|---|---:|
| Snapshots | 23,705 |
| Node features | 76 |
| Mean edges/snapshot | 2,284.78 |
| Edge types | 6 |

Exposure features tăng từ 1 lên 6, nên feature count tăng từ `71` lên `76`.

## Kết quả test mới

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.529750 | 0.650514 | 0.124795 |
| Random Forest | 0.533727 | 0.652021 | 0.120735 |
| GNN + News | 0.584926 | 0.715488 | -0.058770 |
| Full Model | 0.592417 | 0.724355 | -0.085175 |
| GNN Correlation Only | 0.619792 | 0.747527 | -0.155713 |
| GNN + Relationship | 0.621335 | 0.748015 | -0.157223 |

## So với trước khi thêm exposure theo layer

| Model | Trước | Sau | Thay đổi |
|---|---:|---:|---:|
| Linear Regression MAE | 0.530268 | 0.529750 | tốt hơn 0.000518 |
| Random Forest MAE | 0.533938 | 0.533727 | tốt hơn 0.000211 |
| GNN + News MAE | 0.592081 | 0.584926 | tốt hơn 0.007155 |
| Full Model MAE | 0.591559 | 0.592417 | kém hơn 0.000858 |
| GNN Correlation Only MAE | 0.619792 | 0.619792 | không đổi |
| GNN + Relationship MAE | 0.621335 | 0.621335 | không đổi |

## Kết luận

Neighbor exposure theo layer có tác dụng rõ nhất với `GNN + News`.

Điểm tích cực:

- `GNN + News` cải thiện MAE từ `0.592081` xuống `0.584926`.
- Đây là cải thiện rõ trong nhóm GNN.
- Điều này chứng minh hướng thêm shock/exposure features là đúng hơn so với chỉ thêm edge.

Điểm chưa tốt:

- GNN vẫn chưa thắng baseline tabular.
- `Full Model` chưa tốt hơn vì relationship edges vẫn có thể thêm nhiễu.
- `GNN + Relationship` không đổi vì cấu hình này vẫn chỉ dùng price columns, chưa dùng exposure/news/shock features.

Hướng tiếp theo:

1. Tạo cấu hình GNN mới: `GNN + News + Exposure`, chỉ dùng corr graph + news/shock/exposure features, không dùng relationship edges.
2. Tạo cấu hình `GNN + Ownership Exposure` riêng, chỉ lấy ownership/value-chain exposure features thay vì toàn bộ relationship edges.
3. Sau khi ablation rõ ràng hơn, mới thử GAT hoặc R-GCN.
