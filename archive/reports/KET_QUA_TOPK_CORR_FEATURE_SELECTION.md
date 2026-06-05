# Kết quả top-k correlation và chọn đặc trưng

## 1. Mục tiêu thí nghiệm

Sau khi sửa split theo ngày, các mô hình graph chưa vượt baseline tabular. Một
nguyên nhân có thể là graph correlation quá dày, làm nhiễu thông tin truyền từ
láng giềng. Thí nghiệm này thử:

- Giữ lại `top-k` cổ phiếu có tương quan mạnh nhất với target node.
- Kết hợp với các nhóm đặc trưng đã được feature selection gợi ý.

Lưu ý: `top-k` hiện được chọn từ các cạnh correlation đã có trong graph snapshot
hiện tại. Các snapshot này được dựng với ngưỡng `CORR_THRESHOLD=0.15`, nên đây
chưa phải top-k từ toàn bộ ma trận tương quan thô.

## 2. Cấu hình thử nghiệm

Feature sets:

| Feature set | Số đặc trưng | Ý nghĩa |
|---|---:|---|
| `price_only` | 22 | Return lag, rolling volatility, volume ratio |
| `price_micro_macro` | 32 | Price + micro + macro |
| `top_30_rf` | 30 | 30 đặc trưng quan trọng nhất theo Random Forest |
| `full_59` | 59 | Toàn bộ đặc trưng node |

Top-k correlation neighbors:

| k | Ý nghĩa |
|---:|---|
| 5 | Chỉ lấy 5 láng giềng tương quan mạnh nhất |
| 10 | Lấy 10 láng giềng |
| 20 | Lấy 20 láng giềng |
| 40 | Lấy 40 láng giềng |

## 3. Kết quả tốt nhất

| Xếp hạng | Cấu hình | MAE | RMSE |
|---:|---|---:|---:|
| 1 | `top_30_rf_corr_top_20` | 0.009567 | 0.012959 |
| 2 | `top_30_rf_corr_top_5` | 0.009624 | 0.013005 |
| 3 | `top_30_rf_corr_top_10` | 0.009625 | 0.013119 |
| 4 | `top_30_rf_corr_top_40` | 0.009640 | 0.013127 |
| 5 | `full_59_corr_top_10` | 0.009756 | 0.013370 |

## 4. Nhận xét học thuật

Kết quả cho thấy chọn đặc trưng có tác động rõ hơn so với chỉ tăng hoặc giảm số
láng giềng. Nhóm `top_30_rf` cho kết quả tốt nhất ở cả bốn mức top-k, còn
`full_59` không tốt hơn dù chứa nhiều thông tin hơn. Điều này cho thấy thêm
nhiều đặc trưng có thể làm mô hình graph nhiễu hơn.

Trong nhóm top-k, `k=20` là cấu hình tốt nhất. `k=5` có thể quá ít thông tin,
còn `k=40` bắt đầu đưa thêm nhiều láng giềng yếu hơn.

So với kết quả graph trước đó:

| Cấu hình | MAE |
|---|---:|
| Graph MLP `top_30_rf` trước top-k | 0.009618 |
| `top_30_rf_corr_top_20` | 0.009567 |

Top-k correlation cải thiện nhẹ cho graph MLP, nhưng vẫn chưa vượt các baseline
tabular:

| Baseline | MAE |
|---|---:|
| Linear Regression | 0.008730 |
| Random Forest | 0.008752 |
| `top_30_rf_corr_top_20` | 0.009567 |

## 5. Kết luận nên dùng trong khóa luận

Thí nghiệm top-k correlation cho thấy việc giảm mật độ graph có thể cải thiện
nhẹ mô hình graph, đặc biệt khi kết hợp với nhóm đặc trưng đã được chọn bởi
Random Forest. Tuy nhiên, cải thiện này chưa đủ để vượt baseline tabular. Điều
này củng cố kết luận rằng trong bài toán dự báo volatility 5 ngày sau tin tức,
tín hiệu giá lịch sử vẫn chi phối mạnh hơn thông tin lan truyền qua graph hiện
tại.
