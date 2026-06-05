# Chẩn đoán theo thứ tự: Target -> Graph -> Feature -> Model

Ngày kiểm tra: 2026-06-04

## 1. Target

Target hiện tại là `future_realized_volatility_5d`, tức volatility thực hiện trong 5 ngày sau ngày sự kiện.

Kết quả kiểm tra trên test set:

| Thống kê | Giá trị |
|---|---:|
| Số mẫu test | 3,638 |
| Mean | 0.021697 |
| Median | 0.018992 |
| Std | 0.012280 |
| 95% quantile | 0.045756 |
| Max | 0.073765 |

Nhận xét:

- Target bị lệch phải: phần lớn mẫu có volatility thấp-vừa, một số ít mẫu volatility rất cao.
- Các model đều dự đoán co về vùng trung bình, đặc biệt dự báo kém các ngày volatility cao.
- Với nhóm target cao nhất, `y_true` trung bình là `0.041329`, nhưng `GNN + Relationship` chỉ dự đoán trung bình `0.019731`.
- MAE của Linear Regression ở nhóm target cao nhất là `0.018378`, cao hơn nhiều so với nhóm giữa khoảng `0.004581`.

Kết luận target:

Target volatility thô là bài toán khó học. Lỗi lớn nhất không nằm ở mẫu bình thường mà nằm ở các ngày biến động mạnh. Nếu muốn GNN có cơ hội thắng, nên cân nhắc thêm target phụ như `abnormal_volatility` hoặc `volatility_jump_classification`.

## 2. Graph

Graph hiện tại có:

| Metric | Giá trị |
|---|---:|
| Saved snapshots | 23,705 |
| Nodes/snapshot | 118 |
| Edge types | 10 |
| Mean edges/snapshot | 13,777 |

Phân bố edge trung bình:

| Edge type | Mean edges/snapshot |
|---|---:|
| price_correlation | 12,280.70 |
| business_cluster | 930.00 |
| value_chain | 398.00 |
| common_owner | 56.00 |
| strategic_ecosystem | 55.94 |
| same_group | 50.00 |
| parent/subsidiary | 5.32 |
| news_co_mention | 1.39 |

Tỷ trọng:

- `price_correlation` chiếm khoảng `89.1%` tổng số cạnh.
- Static relationship edges chiếm khoảng `10.8%`.
- `news_co_mention` gần như không đáng kể ở đa số snapshot.

Kết luận graph:

Graph đang quá dày và bị correlation edge áp đảo. Đây là khả năng gây nhiễu chính cho GNN. Việc thêm ownership graph có cải thiện rất nhẹ, nhưng không đủ vì tín hiệu ownership bị chìm trong graph correlation quá dày.

## 3. Feature

Random Forest feature importance theo nhóm:

| Nhóm feature | Importance sum |
|---|---:|
| Volatility/absolute-return features | 0.564380 |
| Price lag features | 0.333595 |
| Volume/liquidity | 0.023424 |
| Industry | 0.021317 |
| News | 0.017618 |
| Macro | 0.000656 |

Feature quan trọng nhất:

| Feature | Importance |
|---|---:|
| rolling_vol_20_t_minus_1 | 0.418215 |
| return_mean_10 | 0.042254 |
| abs_return_mean_10 | 0.040823 |
| abs_return_mean_5 | 0.031561 |
| volume_ratio_20_t_minus_1 | 0.023424 |

Kết luận feature:

Feature giá ngắn hạn và volatility quá khứ đang áp đảo. News, macro và relationship exposure có tín hiệu yếu. Đây là lý do baseline tabular vẫn thắng GNN: bài toán hiện tại gần giống dự báo volatility bằng chính volatility quá khứ.

## 4. Model

Kết quả test hiện tại:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.008584 | 0.011489 | 0.124443 |
| Random Forest | 0.008608 | 0.011484 | 0.125184 |
| Tuned TopK Graph MLP | 0.008827 | 0.012185 | N/A |
| GNN Correlation Only | 0.009101 | 0.012301 | -0.003648 |
| GNN + Relationship | 0.009141 | 0.012344 | -0.010676 |
| Full Model | 0.009125 | 0.012440 | -0.026558 |
| GNN + News | 0.009240 | 0.012594 | -0.052049 |

Training log cho thấy:

- GNN train MAE xuống khoảng `0.0068`.
- Validation MAE quanh `0.0094-0.0098`.
- Có khoảng cách train-validation, tức overfitting nhẹ đến vừa.

Kết luận model:

Không nên kết luận "GNN yếu". Vấn đề chính là target khó, graph quá dày/nhiễu, và feature giá quá mạnh. Kiến trúc GNN hiện tại chỉ là one-hop weighted message passing + MLP, nên khó khai thác graph nếu edge chưa thật sự chọn lọc.

## Kết luận tổng hợp

Thứ tự vấn đề hiện tại:

1. Target: volatility thô khó học, đặc biệt các ngày volatility cao.
2. Graph: quá dày, `price_correlation` chiếm 89.1%, ownership/relationship bị chìm.
3. Feature: volatility quá khứ chi phối, news/relationship yếu.
4. Model: có overfit nhẹ, nhưng không phải nguyên nhân gốc.

Hướng sửa ưu tiên:

1. Thêm bài toán phụ `volatility_jump_classification` hoặc target `abnormal_volatility`.
2. Giảm graph bằng top-k correlation thay vì threshold toàn cục.
3. Tách graph ablation rõ hơn: correlation-only, ownership-only, curated-only, news-only, top-k mixed graph.
4. Tăng feature news theo cửa sổ thời gian: news_count_3d/7d, sentiment_sum_3d/7d, negative_news_count, category intensity.
5. Nếu vẫn dùng regression volatility thô, nên dùng weighted loss hoặc quantile/Huber loss để phạt mạnh hơn ở vùng volatility cao.
