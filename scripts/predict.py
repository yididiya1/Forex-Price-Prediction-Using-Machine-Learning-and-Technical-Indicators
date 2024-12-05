#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/predict.py`
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from forex_ml.config import PATHS, Defaults
from forex_ml.data.download import download_ohlcv
from forex_ml.features.indicators import add_features
from forex_ml.modeling.model import TrainedModel


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run a saved forex model to predict the next move")
	p.add_argument("--model", type=str, required=True, help="Path to saved .joblib model")
	p.add_argument("--symbol", type=str, default=Defaults.symbol)
	p.add_argument("--interval", type=str, default=Defaults.interval)
	p.add_argument("--start", type=str, default="2018-01-01")
	p.add_argument("--end", type=str, default=None)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	model_path = Path(args.model)
	trained = TrainedModel.load(model_path)

	df = download_ohlcv(symbol=args.symbol, start=args.start, end=args.end, interval=args.interval, use_cache=True)
	feat = add_features(df)
	# Align to model features
	missing = [f for f in trained.features if f not in feat.columns]
	if missing:
		raise RuntimeError(f"Missing features in current data: {missing}")
	X = feat[trained.features].dropna()
	if X.empty:
		raise RuntimeError("No rows available after feature generation; try earlier start date.")
	latest = X.iloc[[-1]]

	if trained.task == "classification":
		prob_up = trained.model.predict_proba(latest)[:, 1][0]
		label = int(prob_up > 0.5)
		print({"prob_up": float(prob_up), "direction": int(label)})
	else:
		ret = float(trained.model.predict(latest)[0])
		print({"predicted_return": ret, "direction": int(ret > 0)})


if __name__ == "__main__":
	main()
