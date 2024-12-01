from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TaskType = Literal["classification", "regression"]


def create_labels(close: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
	future_return = close.shift(-horizon) / close - 1.0
	binary_direction = (future_return > 0).astype(int)
	return future_return, binary_direction


def build_supervised_dataset(
	features: pd.DataFrame,
	close: pd.Series,
	horizon: int,
	task: TaskType,
) -> tuple[pd.DataFrame, pd.Series]:
	y_reg, y_cls = create_labels(close, horizon)
	# Drop NaNs in labels and align indexes across features and labels
	y_reg = y_reg.dropna()
	y_cls = y_cls.dropna()
	common_index = features.index.intersection(y_reg.index).intersection(y_cls.index)
	features = features.loc[common_index]
	y_reg = y_reg.loc[common_index]
	y_cls = y_cls.loc[common_index]

	if task == "regression":
		X = features
		y = y_reg
	else:
		X = features
		y = y_cls

	mask = (~X.isna().any(axis=1)) & (~y.isna())
	X = X.loc[mask]
	y = y.loc[mask]
	return X, y


def time_series_split(
	X: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Simple chronological split preserving order."""
	n = len(X)
	n_test = int(max(1, round(n * test_size)))
	n_train = n - n_test
	X_train = X.iloc[:n_train]
	X_test = X.iloc[n_train:]
	y_train = y.iloc[:n_train]
	y_test = y.iloc[n_train:]
	return X_train, X_test, y_train, y_test
