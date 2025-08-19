

## Forex ML Project

This folder contains a self-contained Python project to train a model to predict forex market direction/returns using OHLCV data from Yahoo Finance, engineered technical indicators, and a simple backtest.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py --symbol EURUSD=X --interval 1d --start 2015-01-01 --task classification
```

- Saves the trained model under `models/`.
- Prints test metrics and backtest results.
- Saves an equity curve PNG next to the model file.

Key options:
- `--task`: `classification` (predict up/down) or `regression` (predict next return)
- `--horizon`: forecast horizon in bars (default 1)
- `--test-size`: fraction for chronological test split (default 0.2)

### Predict

```bash
python scripts/predict.py --model models/EURUSD=X_1d_h1_classification.joblib --symbol EURUSD=X --interval 1d
```

Outputs either `prob_up` (classification) or `predicted_return` (regression) for the most recent bar.

### Backtest a saved model

```bash
python scripts/backtest.py --model models/EURUSD=X_1d_h1_classification.joblib --symbol EURUSD=X --interval 1d --start 2018-01-01
```

Runs a simple threshold strategy on the model scores and reports total return, Sharpe, max drawdown, win rate, and number of trades, plus an equity curve PNG.

### Notes
- Data is cached under `data/`.
- Models are stored under `models/`.
- Default symbol is `EURUSD=X`. You can use other Yahoo Finance FX tickers (e.g., `GBPUSD=X`, `USDJPY=X`). 
