from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from forex_ml.config import PATHS, Defaults


def _cache_path(symbol: str, interval: str, start: str, end: Optional[str]) -> Path:
	fname = f"{symbol.replace('/', '-')}_{interval}_{start}_{end or 'latest'}.csv"
	return PATHS.data_dir / fname


def download_ohlcv(
	symbol: str = Defaults.symbol,
	start: str = Defaults.start,
	end: Optional[str] = Defaults.end,
	interval: str = Defaults.interval,
	use_cache: bool = True,
) -> pd.DataFrame:
	"""
	Download OHLCV data for a forex pair from Yahoo Finance.
	"""
	cache_file = _cache_path(symbol, interval, start, end)
	if use_cache and cache_file.exists():
		try:
			df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
		except Exception:
			# Fallback: assume first column is datetime index
			df = pd.read_csv(cache_file)
			first_col = df.columns[0]
			df[first_col] = pd.to_datetime(df[first_col])
			df = df.set_index(first_col)
			df.index.name = "Date"
		return df

	df = yf.download(
		tickers=symbol,
		start=start,
		end=end,
		interval=interval,
		auto_adjust=False,
		progress=False,
	)
	if df.empty:
		raise RuntimeError(f"No data returned for {symbol} {interval} {start} {end}")

	# Normalize column names
	df = df.rename(columns={
		"Open": "Open",
		"High": "High",
		"Low": "Low",
		"Close": "Close",
		"Adj Close": "Adj Close",
		"Volume": "Volume",
	})
	# If multi-level columns (can happen in some yfinance versions), take the first level
	if isinstance(df.columns, pd.MultiIndex):
		df.columns = [c[0] for c in df.columns]
	
	df.index.name = "Date"

	cache_file.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(cache_file)
	return df
