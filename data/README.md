# Data

Default: Kaggle **Store Item Demand Forecasting Challenge** (`train.csv`: date, store, item, sales;
10 stores × 50 items, daily, 2013-01-01 → 2017-12-31). `python scripts/download_data.py` fetches it
with the Kaggle API into `data/raw/` (not committed — competition data terms; accept the rules on
kaggle.com once). `python scripts/prepare.py` aggregates to Monday-anchored weeks and writes
`artifacts/weekly.parquet`, which **is** committed as a derived artifact so the app runs without Kaggle.

`python scripts/prepare.py --synthetic` generates look-alike data (seasonality, weekly pattern, trend,
Poisson noise) for tests, CI and anyone without a Kaggle account. Published results use the real data.
