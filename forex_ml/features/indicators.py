from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, window: int) -> pd.Series:
	return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
	return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
	change = series.diff()
	gain = change.clip(lower=0.0)
	loss = -change.clip(upper=0.0)
	avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
	rs = avg_gain / (avg_loss.replace(0, np.nan))
	rsi = 100 - (100 / (1 + rs))
	return rsi


def compute_bbands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series]:
	ma = compute_sma(series, window)
	std = series.rolling(window=window, min_periods=window).std()
	upper = ma + num_std * std
	lower = ma - num_std * std
	return upper, lower


def add_features(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Create a feature dataframe from OHLCV data.
	"""
	features = pd.DataFrame(index=df.index)
	close = df["Close"]
	if isinstance(close, pd.DataFrame):
		# yfinance may return MultiIndex columns; select the first close series
		close = close.iloc[:, 0]

	# Returns and volatility
	features["ret_1"] = close.pct_change()
	features["ret_5"] = close.pct_change(5)
	features["ret_10"] = close.pct_change(10)
	features["vol_5"] = features["ret_1"].rolling(5).std()
	features["vol_10"] = features["ret_1"].rolling(10).std()

	# Moving averages and distances
	for w in [5, 10, 20, 50]:
		ma = compute_sma(close, w)
		ema = compute_ema(close, w)
		features[f"sma_{w}"] = ma
		features[f"ema_{w}"] = ema
		features[f"dist_sma_{w}"] = (close - ma) / ma

	# RSI and Bollinger Bands
	features["rsi_14"] = compute_rsi(close, 14)
	bb_upper, bb_lower = compute_bbands(close, 20, 2.0)
	denom = (bb_upper - bb_lower).replace(0, np.nan)
	features["bb_pos"] = (close - bb_lower) / denom

	# High-low features
	features["hl_range"] = (df["High"] - df["Low"]) / close

	# Drop rows with NaNs introduced by rolling windows
	features = features.replace([np.inf, -np.inf], np.nan).dropna()
	return features
