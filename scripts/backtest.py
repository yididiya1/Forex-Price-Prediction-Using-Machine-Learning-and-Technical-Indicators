#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow running as `python scripts/backtest.py`
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from forex_ml.config import PATHS, Defaults
from forex_ml.data.download import download_ohlcv
from forex_ml.features.indicators import add_features
from forex_ml.modeling.model import TrainedModel
from forex_ml.backtest.backtest import backtest


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Backtest a saved model over a period")
	p.add_argument("--model", type=str, required=True)
	p.add_argument("--symbol", type=str, default=Defaults.symbol)
	p.add_argument("--interval", type=str, default=Defaults.interval)
	p.add_argument("--start", type=str, default=Defaults.start)
	p.add_argument("--end", type=str, default=None)
	p.add_argument("--no-plot", action="store_true")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	model_path = Path(args.model)
	trained = TrainedModel.load(model_path)

	df = download_ohlcv(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval, use_cache=True)
	feat = add_features(df).dropna()
	X = feat[trained.features].dropna()
	# Align to price index
	score_index = X.index
	if trained.task == "classification":
		proba = trained.model.predict_proba(X)[:, 1]
		score = pd.Series(proba, index=score_index)
	else:
		pred = trained.model.predict(X)
		score = pd.Series(pred, index=score_index)

	bt = backtest(close=df["Close"].reindex(score.index), score=score)
	print({
		"total_return": round(bt.total_return, 4),
		"sharpe": round(bt.sharpe, 2),
		"max_drawdown": round(bt.max_drawdown, 4),
		"win_rate": round(bt.win_rate, 3),
		"n_trades": bt.n_trades,
	})

	if not args.no_plot:
		fig, ax = plt.subplots(figsize=(10, 5))
		bt.equity_curve.plot(ax=ax, title="Equity Curve")
		ax.set_ylabel("Equity (relative)")
		ax.grid(True, alpha=0.3)
		png_path = model_path.with_suffix("")
		png_path = png_path.parent / (png_path.name + "_bt_equity.png")
		fig.tight_layout()
		fig.savefig(png_path, dpi=150)
		plt.close(fig)
		print(f"Saved equity plot to {png_path}")


if __name__ == "__main__":
	main()
