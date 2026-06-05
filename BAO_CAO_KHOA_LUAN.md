================================================================================
  KHÓA LUẬN TỐT NGHIỆP: DỰ BÁO BIẾN ĐỘNG CỔ PHIẾU 
  BẰNG ĐỒ THỊ TRI THỨC KẾT HỢP TIN TỨC TÀI CHÍNH
================================================================================

  Sinh viên: [Tên của bạn]  
  Giảng viên hướng dẫn: [Tên giảng viên]
  Thời gian: 2022 - 2025


1. TỔNG QUAN ĐỀ TÀI
--------------------------------------------------------------------------------
  Bài toán: Dự báo biến động giá cổ phiếu (volatility) sử dụng:
    • Dữ liệu lịch sử giá (OHLCV) của 118 mã trên sàn HOSE
    • Tin tức tài chính từ Vietstock (55.518 bài viết)
    • Quan hệ doanh nghiệp (1.398 cạnh đồ thị)
    • Mô hình đồ thị tri thức (Graph Neural Networks)

  Ý tưởng chính: Kết hợp thông tin từ nhiều nguồn (giá, tin tức, 
  quan hệ doanh nghiệp) vào một khung đồ thị thống nhất để dự báo.


2. KIẾN TRÚC PIPELINE CHUẨN (11 bước)
--------------------------------------------------------------------------------

  Bước 1 - Chuẩn bị dữ liệu giá (01_prepare_price_data)
  ─────────────────────────────────────────────────────────
    • Input: universe.csv (118 cổ phiếu), Stock_Price_2022-2025.csv
    • Output: 10 files vào outputs/local/01_price_data/
    • Kiểm soát chất lượng dữ liệu:
        - Kiểm tra thiếu cột OHLCV bắt buộc
        - Kiểm tra missing trong open/high/low/close/volume
        - Loại dòng trùng theo date + ticker
        - Báo cáo lỗi giá bất thường: close <= 0, high < low,
          open/close ngoài biên high-low, volume âm/zero
        - Báo cáo outlier: |log_return| > 30%/50%,
          volume_ratio_20 > 5/10
        - Output bổ sung: price_quality_report.csv
    • Kết quả:
        - 117.180 dòng dữ liệu OHLCV
        - 118 mã cổ phiếu thuộc 12 ngành
        - Khoảng thời gian: 04/01/2022 → 31/12/2025
        - Đặc trưng: log_return, rolling_vol_20, rolling_vol_60,
          volume_ratio_20, abnormal_volume, v.v.

  Bước 2 - Chuẩn bị dữ liệu tin tức (02_prepare_news_data)
  ─────────────────────────────────────────────────────────
    • Input: News_2022_2025_2.xlsx, Vietstock_News_2022_2025_crawl.xlsx
    • Xử lý: chuẩn hóa Unicode, loại bỏ crawl trùng (1.723 dòng)
    • Nhãn tin tức hiện tại: manual labels + mô hình học máy TF-IDF
      + Logistic Regression từ 6.062 nhãn tay
    • Output: 3 files vào outputs/local/02_news_data/
    • Kết quả:
        - 55.518 bài viết firm-specific (có gắn với mã CK)
        - 57.899 ticker mentions (ánh xạ công ty → mã)
        - 343 aliases (tên gọi khác nhau của doanh nghiệp)
        - Phân bố thể loại sau khi áp nhãn ML:
            • earnings:           10.996
            • other:              10.710
            • leadership:          8.674
            • ma_ownership:        7.787
            • debt_bond:           5.260
            • market_industry:     4.097
            • capital_issuance:    2.560
            • dividend:            2.320
            • project_contract:    1.701
            • legal_regulatory:    1.413
        - Phân bố sentiment sau khi áp nhãn ML:
            • neutral:            51.101
            • positive:            2.853
            • negative:            1.564


  Bước 3 - Kiểm định mô hình gán nhãn tin tức (03_train_news_labeler)
  ───────────────────────────────────────────────────────────────
    • Input: 6.062 bài báo được gán nhãn tay
    • Mô hình: TF-IDF word/character n-grams + Logistic Regression
    • Kết quả đánh giá trên holdout 20%:
         Category   Accuracy=0.9060  Macro F1=0.8859
         Sentiment  Accuracy=0.9662  Macro F1=0.8619
    • Output: outputs/local/03_news_labeler/


  Bước 4 - Chuẩn bị quan hệ doanh nghiệp (04_prepare_company_relationships)
  ──────────────────────────────────────────────────────────────
    • Input: relationships.xlsx (2 sheet: sector + quan hệ)
    • Output: 8 files vào outputs/local/04_company_relationships/
    • Kết quả:
        - 1.500 cạnh đồ thị quan hệ chính dùng cho GNN
        - Phân bố:
            • business_cluster:          930 - cụm kinh doanh hẹp
            • value_chain:               398 - chuỗi giá trị
            • strategic_ecosystem:        60 - hệ sinh thái chiến lược
            • common_owner:               56 - cổ đông chung từ ownership graph
            • same_group:                 50 - cùng tập đoàn
            • parent/subsidiary:           6 - công ty mẹ/con hai chiều
        - same_industry_edges.csv vẫn được lưu riêng 1.342 cạnh để đối chiếu,
          nhưng không đưa toàn bộ vào graph chính nhằm tránh graph quá dày và nhiễu.


  Bước 5 - Xây dựng event graph dataset (05_build_event_graph_dataset)
  ────────────────────────────────────────────────────────────
    • Kết hợp dữ liệu từ 3 bước trên thành đồ thị
    • Xử lý missing/outlier khi tạo tensor:
        - Bỏ các bài không đủ 20 ngày lịch sử, 252 ngày correlation,
          hoặc 5 ngày tương lai để tạo nhãn
        - Thay NaN/inf trong feature đầu vào bằng 0 để tensor hợp lệ
        - Clip nhẹ feature đầu vào:
            log_return lags trong [-0.30, 0.30]
            rolling_vol_20 trong [0.00, 0.20]
            volume_ratio_20 trong [0.00, 10.00]
            trading_value_ratio_20 trong [0.00, 10.00]
        - Bổ sung 7 feature giá ngắn hạn:
            realized_vol_lag_5, realized_vol_lag_10,
            return_mean_5, return_mean_10,
            abs_return_mean_5, abs_return_mean_10,
            max_abs_return_5
        - Không clip target y vì volatility cao là tín hiệu cần dự báo
        - Output bổ sung: graph_feature_quality_report.csv
    • Output: 8 files vào outputs/local/05_event_graph_dataset/
    • Kết quả:
        - 23.705 snapshot đồ thị
        - Mỗi snapshot = 1 mã cổ phiếu trong 1 ngày sự kiện
        - Tin cùng mã/cùng ngày được gộp để tránh nhân bản target
        - Trung bình 13.696 cạnh/snapshot
        - 66 features/node (giá + giá ngắn hạn + vi mô + ngành + vĩ mô + tin tức)
        - 10 loại cạnh:
            • price_correlation (tương quan giá)
            • parent_to_subsidiary / subsidiary_to_parent
            • same_group / strategic_ecosystem / business_cluster / value_chain / common_owner
            • news_co_mention (đồng xuất hiện trong tin tức)


  Bước 6 - Baseline Models (06_train_baseline_models)
  ─────────────────────────────────────────────
    • Chia dữ liệu theo event_trading_date:
         Train:      518 ngày
         Validation: 111 ngày
         Test:       111 ngày, 3.638 ticker-date samples
      Mỗi ngày chỉ thuộc một split, tránh rò rỉ thông tin cùng ngày.
    • 4 mô hình nền tảng:
         Linear Regression      MAE=0.00858  RMSE=0.01149  ← Best MAE
         Random Forest          MAE=0.00861  RMSE=0.01148  ← Best R2
         GNN Correlation Only   MAE=0.00910  RMSE=0.01230
         Rolling Volatility     MAE=0.00946  RMSE=0.01226

  Bước 7 - GNN Ablation (07_train_gnn_ablation_models)
  ────────────────────────────────────────────────────
    • 4 cấu hình GNN (test-set):
         GNN Correlation Only   MAE=0.00910  RMSE=0.01230
         Full Model             MAE=0.00913  RMSE=0.01244
         GNN + Relationship     MAE=0.00914  RMSE=0.01234
         GNN + News             MAE=0.00924  RMSE=0.01259

    => Nhận xét: Sau khi gộp dữ liệu theo ticker-date, GNN không còn
       hưởng lợi từ việc nhân bản nhiều tin cùng target. GNN vẫn chưa
       vượt baseline tuyến tính/tabular trên target volatility thô.

  Bước 8 - Fine-tune GNN được chọn (08_tune_selected_gnn)
  ─────────────────────────────────────
    • Cấu hình đầu vào:
         top_30_rf + corr_top_20
    • Cải tiến khi train:
         StandardScaler cho X và y
         SmoothL1 loss
         AdamW + weight decay
         gradient clipping
         early stopping theo validation MAE
         interaction features: self, neighbor, self-neighbor, self*neighbor
    • Kết quả tốt nhất:
         hidden_dim=64, dropout=0.00
         MAE=0.00883  RMSE=0.01219


  Bước 9 - Evaluation & Case Studies (09_evaluate_models_and_cases)
  ─────────────────────────────────────────────────────────────────────────
    • 45 case studies được phân tích chi tiết
    • Đánh giá sai số theo:
        - Ngành (category)
        - Mã cổ phiếu (ticker)
        - Ngày sự kiện (event date)

  Bước 10 - Report Tables & Figures (10_generate_report_assets)
  ────────────────────────────────────────────────────────────────────────
    • 4 biểu đồ:
        1) So sánh MAE/RMSE giữa các mô hình
        2) Random Forest feature importance (top 15)
        3) GNN ablation validation curves
        4) Test MAE theo news category
    • 5 bảng số liệu chi tiết

  Bước 11 - Final Regression Metrics (11_compute_regression_metrics)
  ─────────────────────────────────────────────────────────────
    • Tính MAE, RMSE và R2 từ prediction files
    • Output: outputs/report/11_regression_metrics/


3. KẾT LUẬN CHÍNH
--------------------------------------------------------------------------------

  1. Sau khi sửa split theo event_trading_date, bổ sung feature giá
     ngắn hạn và gộp tin theo ticker-date, Linear Regression đạt MAE
     tốt nhất trên test-set (MAE=0.00858, RMSE=0.01149). Random Forest
     đứng rất sát và có R2 tốt nhất trong nhóm baseline.

  2. Việc sửa split theo ngày làm kết quả đánh giá chặt hơn. So với
     snapshot-level split trước đó, MAE tăng khoảng 2.6% đến 7.8%
     tùy mô hình, cho thấy split cũ có thể lạc quan hơn thực tế.

  3. Cạnh price_correlation là loại cạnh quan trọng nhất
     trong đồ thị. Các loại cạnh khác ít tác động.

  4. GNN chưa vượt qua được Random Forest trên bài toán này,
     và cũng chưa vượt baseline tuyến tính sau khi chia dữ liệu theo
     ngày. Gợi ý cần cải tiến biểu diễn tin tức, kiến trúc graph
     learning, hoặc cách xây dựng cạnh.

  5. Nhãn tin tức học từ 6.062 nhãn tay đạt chất lượng phân loại
     tương đối tốt (macro F1 > 0.86), nhưng khi đưa vào mô hình dự báo
     volatility, tín hiệu tin tức vẫn chưa đủ mạnh để vượt baseline giá.

  6. Top-k correlation giúp giảm nhiễu so với dùng toàn bộ graph trong
     một số cấu hình. Khi kết hợp với chuẩn hóa feature/target, feature
     giá ngắn hạn và interaction self-neighbor, Tuned TopK Graph MLP đạt
     MAE=0.00883, tốt hơn các biến thể GNN cơ bản nhưng vẫn kém baseline
     tabular tốt nhất.

  7. Pipeline xử lý dữ liệu hoàn chỉnh, tái sử dụng được,
     output được tổ chức rõ ràng theo từng bước.


4. HƯỚNG PHÁT TRIỂN
--------------------------------------------------------------------------------
  • Thử nghiệm GNN deep hơn (2+ layers, attention mechanisms)
  • Cập nhật dữ liệu real-time qua API
  • Xây dựng web demo trực quan
  • Kết hợp thêm dữ liệu vĩ mô (lãi suất, CPI, GDP)
  • Thử nghiệm mô hình transformer-based cho news encoding

================================================================================
