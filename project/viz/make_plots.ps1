$env:PYTHONUTF8=1
python viz/export_ts_plots.py
python viz/export_lp_plots.py
python viz/export_gephi_from_preds.py
python viz/export_closure_lp_plots.py
python viz/export_closure_gephi_from_preds.py

# option: dynamic
python viz/export_closure_dynamic_gexf.py
Write-Host "Done. Check data/figs and data/exports"
