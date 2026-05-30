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


2. KIẾN TRÚC PIPELINE (10 bước)
--------------------------------------------------------------------------------

  Bước 1 - Xây dựng dữ liệu giá (01_build_price_dataset)
  ─────────────────────────────────────────────────────────
    • Input: universe.csv (118 cổ phiếu), Stock_Price_2022-2025.csv
    • Output: 9 files vào outputs/01_build_price_dataset/
    • Kết quả:
        - 117.180 dòng dữ liệu OHLCV
        - 118 mã cổ phiếu thuộc 12 ngành
        - Khoảng thời gian: 04/01/2022 → 31/12/2025
        - Đặc trưng: log_return, rolling_vol_20, rolling_vol_60,
          volume_ratio_20, abnormal_volume, v.v.

  Bước 2 - Xây dựng dữ liệu tin tức (02_build_news_dataset)
  ─────────────────────────────────────────────────────────
    • Input: News_2022_2025_2.xlsx, Vietstock_News_2022_2025_crawl.xlsx
    • Xử lý: chuẩn hóa Unicode, loại bỏ crawl trùng (1.723 dòng)
    • Output: 3 files vào outputs/02_build_news_dataset/
    • Kết quả:
        - 55.518 bài viết firm-specific (có gắn với mã CK)
        - 57.899 ticker mentions (ánh xạ công ty → mã)
        - 343 aliases (tên gọi khác nhau của doanh nghiệp)
        - Phân bố thể loại:
            • other:              25.535 (46%)
            • leadership:          8.494 (15%)
            • earnings:            8.225 (15%)
            • debt_bond:           4.057 ( 7%)
            • capital_issuance:    3.401 ( 6%)
            • ma_ownership:        2.155 ( 4%)
            • dividend:            1.496 ( 3%)
            • project_contract:      889 ( 2%)
            • legal_regulatory:      697 ( 1%)
            • market_industry:       569 ( 1%)


  Bước 3 - Quan hệ doanh nghiệp (03_build_company_relationships)
  ──────────────────────────────────────────────────────────────
    • Input: relationships.xlsx (2 sheet: sector + quan hệ)
    • Output: 6 files vào outputs/03_build_company_relationships/
    • Kết quả:
        - 1.398 cạnh đồ thị quan hệ
        - Phân bố:
            • same_industry:           1.342 (96%) - cùng ngành
            • same_group:                 50 ( 4%) - cùng tập đoàn
            • parent_to_subsidiary:        3 (  ) - công ty mẹ → con
            • subsidiary_to_parent:        3 (  ) - công ty con → mẹ


  Bước 4 - Xây dựng Graph Snapshots (04_build_graph_snapshots)
  ────────────────────────────────────────────────────────────
    • Kết hợp dữ liệu từ 3 bước trên thành đồ thị
    • Output: 7 files vào outputs/04_build_graph_snapshots/
    • Kết quả:
        - 42.886 snapshot đồ thị (mỗi snapshot = 1 sự kiện tin tức)
        - Trung bình 13.696 cạnh/snapshot
        - 59 features/node (giá + vi mô + ngành + vĩ mô + tin tức)
        - 6 loại cạnh:
            • price_correlation (tương quan giá)
            • parent_to_subsidiary / subsidiary_to_parent
            • same_group / same_industry
            • news_co_mention (đồng xuất hiện trong tin tức)


  Bước 5 - Baseline Models (05_train_baselines)
  ─────────────────────────────────────────────
    • 4 mô hình nền tảng:
         Rolling Volatility     MAE=0.00879  RMSE=0.01118
         Linear Regression      MAE=0.00812  RMSE=0.01070
         Random Forest          MAE=0.00812  RMSE=0.01064  ← Best
         GNN Correlation Only   MAE=0.00899  RMSE=0.01210

  Bước 6 - GNN Ablation (06_train_gnn_ablation_models)
  ────────────────────────────────────────────────────
    • 4 cấu hình GNN (test-set):
         GNN Correlation Only   MAE=0.00899  RMSE=0.01210
         GNN + News             MAE=0.00902  RMSE=0.01200
         Full Model             MAE=0.00903  RMSE=0.01210
         GNN + Relationship     MAE=0.00909  RMSE=0.01215

    => Nhận xét: Thêm tin tức giúp giảm RMSE tốt nhất.
       Thêm quan hệ doanh nghiệp chưa cải thiện rõ rệt.

  Bước 7 - Evaluation & Case Studies (07_evaluate_results_and_case_studies)
  ─────────────────────────────────────────────────────────────────────────
    • 45 case studies được phân tích chi tiết
    • Đánh giá sai số theo:
        - Ngành (category)
        - Mã cổ phiếu (ticker)
        - Ngày sự kiện (event date)

  Bước 8 - Report Tables & Figures (08_generate_report_tables_and_figures)
  ────────────────────────────────────────────────────────────────────────
    • 4 biểu đồ:
        1) So sánh MAE/RMSE giữa các mô hình
        2) Random Forest feature importance (top 15)
        3) GNN ablation validation curves
        4) Test MAE theo news category
    • 5 bảng số liệu chi tiết

  Bước 9 - Feature Selection (09_feature_selection_experiments)
  ─────────────────────────────────────────────────────────────
    • 9 cấu hình feature sets khác nhau (22 → 59 features)
    • Top 3 với Random Forest (test-set):
         price_macro         (30 features)    MAE=0.00802
         price_micro         (24 features)    MAE=0.00802
         price_only          (22 features)    MAE=0.00802

    => Kết luận: Chỉ cần 22 features giá là đủ, thêm features
       vi mô/vĩ mô/ngành/tin tức không cải thiện đáng kể!

  Bước 10 - Edge Ablation (10_edge_ablation_experiments)
  ──────────────────────────────────────────────────────
    • 8 cấu hình edge types:
         corr_threshold_0_30          MAE=0.00925
         corr_threshold_0_15          MAE=0.00925
         corr_plus_parent_subsidiary  MAE=0.00925
         corr_plus_same_industry      MAE=0.00931
         full_graph                   MAE=0.00931
         corr_plus_news_co_mention    MAE=0.00931
         corr_plus_same_group         MAE=0.00933
         corr_threshold_0_20          MAE=0.00933

    => Kết luận: Cạnh tương quan giá là quan trọng nhất.
       Thêm cạnh quan hệ/tin tức không cải thiện đáng kể.


3. KẾT LUẬN CHÍNH
--------------------------------------------------------------------------------

  1. Random Forest đạt kết quả tốt nhất (MAE=0.00812) trong số
     các mô hình baseline, vượt qua cả GNN phức tạp.

  2. Feature price-only (22 features) đã đủ tốt, việc thêm
     news/industry/macro features không cải thiện đáng kể.

  3. Cạnh price_correlation là loại cạnh quan trọng nhất
     trong đồ thị. Các loại cạnh khác ít tác động.

  4. GNN chưa vượt qua được Random Forest trên bài toán này,
     gợi ý cần cải tiến kiến trúc hoặc cách xây dựng đồ thị.

  5. Pipeline xử lý dữ liệu hoàn chỉnh, tái sử dụng được,
     output được tổ chức rõ ràng theo từng bước.


4. HƯỚNG PHÁT TRIỂN
--------------------------------------------------------------------------------
  • Thử nghiệm GNN deep hơn (2+ layers, attention mechanisms)
  • Cập nhật dữ liệu real-time qua API
  • Xây dựng web demo trực quan
  • Kết hợp thêm dữ liệu vĩ mô (lãi suất, CPI, GDP)
  • Thử nghiệm mô hình transformer-based cho news encoding

================================================================================
