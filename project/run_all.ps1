# Dừng script khi có lỗi, biến chưa khai báo, hoặc pipeline lỗi
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Cài đặt các thư viện cần thiết
python -m pip install -r requirements.txt

# Chạy các bước ETL và train model
python etl/etl_metrics.py
python models/train_timeseries.py
python models/train_reciprocity_lp.py

Write-Host "Done. Files in data/processed ready for API/dashboard."
