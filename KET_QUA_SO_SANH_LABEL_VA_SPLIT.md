# Kết quả so sánh nhãn tin tức và cách chia dữ liệu

## 1. So sánh rule-based labels và ML labels

Tập nhãn ML được huấn luyện từ 6.062 bài báo gán nhãn thủ công bằng mô hình
TF-IDF + Logistic Regression. Sau đó mô hình được áp dụng cho toàn bộ 55.518
bài báo:

| Chỉ tiêu | Giá trị |
|---|---:|
| Tổng số bài báo | 55.518 |
| Bài giữ nhãn thủ công | 6.062 |
| Bài dùng nhãn mô hình | 49.456 |
| Bài đổi category so với rule-based | 19.771 |
| Bài đổi sentiment so với rule-based | 4.345 |

Kết quả đánh giá bộ gán nhãn trên holdout 20%:

| Task | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Category | 0.9060 | 0.8859 | 0.9049 |
| Sentiment | 0.9662 | 0.8619 | 0.9660 |

So sánh mô hình dự báo volatility trước khi sửa split theo ngày:

| Model | MAE rule-based | MAE ML labels | Thay đổi MAE |
|---|---:|---:|---:|
| Random Forest | 0.008116 | 0.008120 | +0.05% |
| Linear Regression | 0.008121 | 0.008126 | +0.06% |
| Rolling Volatility | 0.008790 | 0.008790 | 0.00% |
| GNN + News | 0.009057 | 0.008944 | -1.24% |
| GNN Correlation Only | 0.008988 | 0.008988 | 0.00% |
| Full Model | 0.008999 | 0.009031 | +0.36% |
| GNN + Relationship | 0.009089 | 0.009089 | 0.00% |

Nhận xét: nhãn ML cải thiện rõ nhất cho `GNN + News`, nhưng chưa đủ để vượt
các baseline tabular.

## 2. So sánh snapshot-level split và day-level split

Ban đầu dữ liệu được chia theo thứ tự snapshot. Cách này có thể làm một ngày
giao dịch xuất hiện ở nhiều split khác nhau. Pipeline hiện đã được sửa sang
chia theo `event_trading_date`, bảo đảm mỗi ngày chỉ thuộc một split:

| Split | Số ngày | Số snapshot | Khoảng thời gian |
|---|---:|---:|---|
| Train | 518 | 28.788 | 2023-01-06 đến 2025-02-07 |
| Validation | 111 | 6.656 | 2025-02-10 đến 2025-07-18 |
| Test | 111 | 7.442 | 2025-07-21 đến 2025-12-24 |

Kết quả test sau khi sửa split theo ngày:

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 0.008730 | 0.011640 |
| Random Forest | 0.008752 | 0.011661 |
| GNN Correlation Only | 0.009222 | 0.012531 |
| GNN + Relationship | 0.009348 | 0.012685 |
| Rolling Volatility | 0.009365 | 0.012090 |
| Full Model | 0.009529 | 0.012824 |
| GNN + News | 0.009624 | 0.013040 |

So với snapshot-level split, MAE tăng khoảng 2.6% đến 7.8% tùy mô hình. Điều
này cho thấy split theo ngày là cách đánh giá chặt hơn và phù hợp hơn với bài
toán dự báo theo thời gian.

## 3. Kết luận nên đưa vào khóa luận

Sau khi chỉnh cách chia dữ liệu theo ngày, kết quả thực nghiệm cho thấy các
baseline tuyến tính/tabular vẫn tốt hơn các biến thể graph hiện tại. Nhãn tin
tức học từ dữ liệu thủ công có chất lượng phân loại tốt, nhưng tín hiệu tin tức
chưa đủ mạnh để cải thiện đáng kể dự báo volatility 5 ngày trong thiết lập mô
hình hiện tại.
