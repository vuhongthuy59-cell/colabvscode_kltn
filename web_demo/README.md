# News Impact Web Demo

Static dashboard demo for the stock-news volatility forecasting project.

## Run

From the project root:

```powershell
python web_demo\build_demo_data.py
python -m http.server 8765 --bind 127.0.0.1 -d web_demo
```

Open:

```text
http://127.0.0.1:8765
```

## Data

The app reads `web_demo/data/demo-data.json`, generated from files under `data/processed`.
The checked-in demo JSON can use 2026 records for illustration, while the main thesis
experiments should stay on the stable `2022-01-01` to `2025-12-31` dataset.

To refresh demo data with the 2026 append files, intentionally build the extended
dataset first:

```powershell
$env:KLTN_INCLUDE_2026_APPEND = "1"
python scripts\01_build_price_dataset.py
python scripts\02_build_news_dataset.py
# then rerun the downstream processing/training scripts before:
python web_demo\build_demo_data.py
```

Clear `KLTN_INCLUDE_2026_APPEND` and rerun the numbered pipeline before generating
final thesis tables.
