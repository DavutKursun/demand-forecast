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

Rolling-origin backtest on the real data: 6 folds (origins every 4 weeks, July–November 2017), 500 store–product series, horizon 1–4 weeks, models retrained per fold. Numbers from `python scripts/eval.py`.

| Method | WAPE (backtest) |
|---|---|
| Seasonal naive (same week last year) | 6.4% |
| 8-week moving average | 11.9% |
| **LightGBM (this project)** | **4.4%** |

Improvement vs best baseline: **30.6%** · folds won: **6/6** · P10–P90 coverage: **76%** (target 80%) · bias: +0.7 units/week

Where it is weakest: the lowest-volume series. The worst 10 items sell 110–217 units a week against a median of 351, and their WAPE is 9–11%, so there is no catastrophic tail. Not in the data: promotions and prices, and the intermittent near-zero products that real catalogues have. Full findings in [docs/DESIGN.md](docs/DESIGN.md#findings-milestone-3-real-data).

## Run it yourself

```bash
git clone https://github.com/DavutKursun/demand-forecast && cd demand-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# macOS only: LightGBM's wheel links the OpenMP runtime, which Homebrew provides
brew install libomp

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
