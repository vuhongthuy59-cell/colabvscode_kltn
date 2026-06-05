# Kết quả thử nghiệm mở rộng để cải thiện GNN

## 1. Mục tiêu

Mục tiêu của vòng thử nghiệm này là tiếp tục cải thiện GNN để vượt baseline tabular tốt nhất hiện tại:

| Baseline hiện tại | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 0.008550 | 0.011369 |
| Random Forest | 0.008599 | 0.011469 |

Điều kiện giữ nguyên:

- Không sửa nhãn.
- Không sửa target.
- Không dùng test set để train.
- Vẫn split theo `event_trading_date`.

## 2. Các hướng đã thử

### 2.1 Residual GNN

Script: `scripts/14_train_residual_gnn_experiments.py`

Ý tưởng:

```text
prediction = Linear baseline + alpha * GNN residual
```

Trong đó:

- Linear baseline được train trên train set.
- GNN học phần sai số còn lại: `y_true - y_linear_pred`.
- `alpha` được chọn bằng validation MAE.

Kết quả test tốt nhất:

| Model | MAE | RMSE |
|---|---:|---:|
| Residual TopK Graph MLP `top30_h64_d010` | 0.008744 | 0.011783 |
| Linear Regression baseline | 0.008550 | 0.011369 |

Kết luận: Residual GNN cải thiện trên validation nhưng không tổng quát đủ tốt sang test.

### 2.2 Validation-selected blending

Script: `scripts/15_gnn_blending_experiments.py`

Ý tưởng:

Chọn trọng số ensemble trên validation giữa:

- Linear Regression
- Random Forest
- GNN + Relationship
- Residual GNN top30
- Residual GNN top50

Kết quả test:

| Model | MAE | RMSE |
|---|---:|---:|
| Validation-selected blend | 0.008774 | 0.011790 |
| Linear Regression baseline | 0.008550 | 0.011369 |

Trọng số validation chọn:

```text
linear = 0.0000
rf = 0.0000
gnn_relationship = 0.5800
residual_top30 = 0.0000
residual_top50 = 0.4200
```

Kết luận: Ensemble nghiêng về GNN trên validation nhưng không thắng trên test, cho thấy có lệch phân phối giữa validation và test.

### 2.3 Anchored GNN

Script: `scripts/16_train_anchored_gnn_experiments.py`

Ý tưởng:

Đưa dự báo Linear Regression vào làm feature neo cho Graph MLP:

```text
input = [linear_prediction, self, neighbor, self-neighbor, self*neighbor]
```

Kết quả test tốt nhất:

| Model | MAE | RMSE |
|---|---:|---:|
| Anchored TopK Graph MLP `top50_anchor_h96_d010` | 0.008822 | 0.011950 |
| Linear Regression baseline | 0.008550 | 0.011369 |

Kết luận: Thêm anchor giúp mô hình gần baseline hơn, nhưng neural graph vẫn không vượt được baseline tuyến tính.

### 2.4 Graph regularized baseline

Script: `scripts/17_graph_regularized_baselines.py`

Ý tưởng:

Trước khi tiếp tục làm neural sâu hơn, kiểm tra xem graph interaction feature có tín hiệu tuyến tính tốt hay không:

```text
input = [tabular self features, self, neighbor, self-neighbor, self*neighbor]
model = Ridge
```

Kết quả test tốt nhất:

| Model | MAE | RMSE |
|---|---:|---:|
| Ridge graph `top30_graph`, alpha=10 | 0.008734 | 0.011486 |
| Linear Regression baseline | 0.008550 | 0.011369 |

Kết luận: Graph interaction có tín hiệu nhưng chưa đủ để vượt baseline giá trực tiếp.

## 3. Bảng tổng hợp

| Nhóm mô hình | Best MAE | Best RMSE | Có thắng Linear không? |
|---|---:|---:|---|
| Linear Regression baseline | 0.008550 | 0.011369 | Baseline |
| Random Forest baseline | 0.008599 | 0.011469 | Không |
| Tuned TopK Graph MLP cũ | 0.008818 | 0.011957 | Không |
| Residual GNN | 0.008744 | 0.011783 | Không |
| Validation-selected GNN blend | 0.008774 | 0.011790 | Không |
| Anchored GNN | 0.008822 | 0.011950 | Không |
| Graph Ridge diagnostic | 0.008734 | 0.011486 | Không |

## 4. Nhận xét chuyên môn

Các thử nghiệm cho thấy vấn đề chính không còn nằm ở optimizer hay kiến trúc MLP đơn giản. Baseline tuyến tính đang thắng vì target `5-day realized volatility` bị chi phối rất mạnh bởi các feature giá ngắn hạn, đặc biệt:

- `rolling_vol_20_t_minus_1`
- `abs_return_mean_10`
- `return_mean_10`
- `max_abs_return_5`
- các log-return lag gần sự kiện

Graph/news/relationship có tín hiệu bổ sung nhưng tín hiệu này yếu hơn biến động giá gần nhất. Khi đưa graph vào, mô hình cải thiện so với GNN ban đầu nhưng vẫn chưa đủ để vượt baseline tuyến tính.

## 5. Hướng tiếp theo nếu vẫn muốn GNN thắng

Nếu mục tiêu bắt buộc là làm GNN thắng, hướng nên làm tiếp không phải chỉ tăng hidden layer. Cần thay đổi cách đặt bài toán để graph/news có vai trò rõ hơn:

1. Dự báo `abnormal volatility` thay vì volatility thô:

```text
abnormal_volatility = future_realized_vol_5d - rolling_vol_20_t_minus_1
```

Lý do: volatility thô bị giá quá khứ chi phối. Khi trừ nền volatility lịch sử, phần còn lại có khả năng liên quan đến tin tức và quan hệ doanh nghiệp hơn.

2. Làm graph động theo thời gian:

Hiện correlation edge chủ yếu là quan hệ thống kê quá khứ. Nên thử `lead-lag correlation`, ví dụ cổ phiếu A đi trước cổ phiếu B 1-3 ngày.

3. Thay nhãn tin tức rời rạc bằng embedding văn bản:

Hiện news feature chủ yếu là category/sentiment/relevance. Với đề tài GNN + tin tức, nên dùng embedding từ PhoBERT/FinBERT tiếng Việt hoặc sentence-transformer, sau đó đưa vào node/event representation.

4. Chuyển sang bài toán classification phụ:

Ví dụ:

```text
volatility_jump = 1 nếu future_vol_5d nằm trong top 25% theo từng mã
```

Tin tức và quan hệ doanh nghiệp thường hữu ích hơn trong bài toán phát hiện biến động bất thường so với hồi quy volatility thô.

5. Dùng walk-forward validation nhiều fold:

Validation hiện chọn GNN tốt hơn Linear, nhưng test lại không theo. Điều này cho thấy cần nhiều fold theo thời gian để giảm rủi ro chọn mô hình theo một giai đoạn validation duy nhất.

## 6. Kết luận hiện tại

Chưa có mô hình GNN nào thắng baseline Linear Regression trên test set hiện tại. Kết quả tốt nhất vẫn là:

```text
Linear Regression MAE = 0.008550
Best GNN-family MAE  = 0.008744
```

Tuy nhiên, đây là kết quả tốt cho khóa luận nếu trình bày đúng: GNN đã được cải thiện qua nhiều bước, nhưng bài toán volatility thô ở dữ liệu hiện tại bị đặc trưng giá ngắn hạn chi phối mạnh. Bước nghiên cứu tiếp theo hợp lý nhất là đổi target sang `abnormal volatility` hoặc `volatility jump classification` để kiểm tra vai trò thật sự của tin tức và graph.
