# demand-forecast

**Weekly sales forecasts per store and product, with an uncertainty band and a plain-language answer to "how much should I order this week?"**

<!-- M5: ![demo](assets/demo.gif) -->

🚀 **Live demo:** _coming in Milestone 5_ · 📄 **How it works:** [docs/DESIGN.md](docs/DESIGN.md)

## The problem

Every week a retailer decides how much of each product to order. Too little and the shelf is empty; too much and cash sits in stock. Most teams guess from last week or last year. This project forecasts the next 4 weeks for every store × product, shows how sure it is, and turns that into an order quantity — and it proves, on a held-out backtest, how much better than the simple rules it is.

## What this does

- Forecasts 4 weeks ahead for 500 store–product series (public Kaggle retail data) with P10–P90 bands
- Compares against the two rules a business would use anyway (same week last year; 8-week average) in a rolling-origin backtest with leakage tests
- Dashboard: history + forecast chart, order recommendation from stock on hand, lead time and service level, model-quality page
- No LLM, no GPU: LightGBM trains in minutes on a laptop; the app reads precomputed artifacts

## Results

| Method | WAPE (backtest) |
|---|---|
| Same week last year | _M3_ |
| 8-week moving average | _M3_ |
| **LightGBM (this project)** | _M3_ |

Improvement vs best baseline: _M3_ · folds won: _M3_/6 · P10–P90 coverage: _M3_

What does not work yet: products with very low volume (WAPE is unstable), promotions/price effects (not in the data).

## Run it yourself

```bash
git clone https://github.com/DavutKursun/demand-forecast && cd demand-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/download_data.py        # Kaggle API; or: python scripts/prepare.py --synthetic
python scripts/prepare.py              # → artifacts/weekly.parquet
python scripts/train.py                # backtest + final forecasts → artifacts/
python scripts/eval.py                 # prints the results table
streamlit run app.py
```

Tests: `pytest` (18 tests, synthetic data, ~10 s).

## Stack

Python · pandas · LightGBM · scikit-learn · Streamlit · Plotly · pytest · GitHub Actions

## License

MIT
