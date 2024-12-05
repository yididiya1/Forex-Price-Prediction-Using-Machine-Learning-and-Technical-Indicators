#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running as `python scripts/train.py`
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from forex_ml.config import PATHS, Defaults
from forex_ml.data.download import download_ohlcv
from forex_ml.features.indicators import add_features
from forex_ml.modeling.dataset import build_supervised_dataset, time_series_split
from forex_ml.modeling.model import train as train_model, evaluate as evaluate_model
from forex_ml.backtest.backtest import backtest


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Train a forex prediction model")
	p.add_argument("--symbol", type=str, default=Defaults.symbol)
	p.add_argument("--interval", type=str, default=Defaults.interval)
	p.add_argument("--start", type=str, default=Defaults.start)
	p.add_argument("--end", type=str, default=Defaults.end)
	p.add_argument("--horizon", type=int, default=Defaults.forecast_horizon)
	p.add_argument("--task", type=str, choices=["classification", "regression"], default="classification")
	p.add_argument("--test-size", type=float, default=Defaults.test_size)
	p.add_argument("--random-state", type=int, default=Defaults.random_state)
	p.add_argument("--out", type=str, default=None, help="Optional explicit model output path")
	p.add_argument("--no-plot", action="store_true", help="Disable saving plots")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	PATHS.ensure()

	print(f"Downloading {args.symbol} {args.interval} {args.start} -> {args.end or 'latest'}")
	df = download_ohlcv(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval, use_cache=True)

	print("Engineering features...")
	feat = add_features(df)
	X, y = build_supervised_dataset(feat, df["Close"], horizon=args.horizon, task=args.task)
	X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=args.test_size)

	print(f"Training {args.task} model on {len(X_train)} samples; testing on {len(X_test)} samples")
	trained = train_model(X_train, y_train, task=args.task, random_state=args.random_state)
	metrics = evaluate_model(trained, X_test, y_test)
	print("Test metrics:", metrics)

	# Backtest on the test split
	print("Backtesting on test split...")
	if args.task == "regression":
		score = pd.Series(trained.model.predict(X_test), index=X_test.index)
	else:
		proba = trained.model.predict_proba(X_test)[:, 1]
		score = pd.Series(proba, index=X_test.index)
	bt = backtest(close=df["Close"].reindex(score.index), score=score)
	print({
		"total_return": round(bt.total_return, 4),
		"sharpe": round(bt.sharpe, 2),
		"max_drawdown": round(bt.max_drawdown, 4),
		"win_rate": round(bt.win_rate, 3),
		"n_trades": bt.n_trades,
	})

	# Save model
	if args.out:
		model_path = Path(args.out)
	else:
		name = f"{args.symbol.replace('/', '-')}_{args.interval}_h{args.horizon}_{args.task}.joblib"
		model_path = PATHS.models_dir / name
	trained.save(model_path)
	print(f"Saved model to {model_path}")

	# Save plots
	if not args.no_plot:
		fig, ax = plt.subplots(figsize=(10, 5))
		bt.equity_curve.plot(ax=ax, title="Equity Curve (Test)")
		ax.set_ylabel("Equity (relative)")
		ax.grid(True, alpha=0.3)
		png_path = model_path.with_suffix("")
		png_path = png_path.parent / (png_path.name + "_equity.png")
		fig.tight_layout()
		fig.savefig(png_path, dpi=150)
		plt.close(fig)
		print(f"Saved equity plot to {png_path}")


if __name__ == "__main__":
	main()
