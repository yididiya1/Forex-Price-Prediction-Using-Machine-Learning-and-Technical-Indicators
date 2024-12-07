from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
	equity_curve: pd.Series
	total_return: float
	sharpe: float
	max_drawdown: float
	win_rate: float
	n_trades: int


def compute_metrics(pnl: pd.Series) -> tuple[float, float, float, float, int]:
	ret = pnl.add(1.0).cumprod()
	total_return = ret.iloc[-1] - 1.0 if len(ret) else 0.0
	# Use daily Sharpe assuming pnl is daily; scale sqrt(252)
	mean = pnl.mean()
	std = pnl.std(ddof=0)
	sharpe = (mean / std) * np.sqrt(252) if std and std > 0 else 0.0
	# Max drawdown
	roll_max = ret.cummax()
	drawdown = (ret / roll_max - 1.0)
	max_dd = drawdown.min() if len(drawdown) else 0.0
	# Win rate and trades
	wins = (pnl > 0).sum()
	n_trades = (pnl != 0).sum()
	win_rate = float(wins) / float(n_trades) if n_trades > 0 else 0.0
	return total_return, sharpe, max_dd, win_rate, n_trades


def backtest(
	close: pd.Series,
	score: pd.Series,
	long_threshold: float = 0.55,
	short_threshold: Optional[float] = None,
	cost_bps: float = 1.0,
) -> BacktestResult:
	"""
	Simple threshold strategy:
	- If score in [0,1] is a probability of up, convert to signed score in [-1,1]
	- Go long if prob > long_threshold; short if prob < 1 - long_threshold
	- Flat otherwise
	- cost_bps applies on position changes (round-trip cost = 2*cost_bps approx)
	"""
	if short_threshold is None:
		short_threshold = 1.0 - long_threshold

	if score.min() >= 0 and score.max() <= 1:
		prob_up = score
		position = np.where(prob_up > long_threshold, 1, np.where(prob_up < short_threshold, -1, 0))
	else:
		position = np.sign(score).astype(int)

	position = pd.Series(position, index=score.index).astype(float)
	returns = close.pct_change().reindex(position.index)
	# Enter next bar
	shifted_pos = position.shift(1).fillna(0.0)
	gross_pnl = shifted_pos * returns
	# Trading cost when position changes
	turnover = shifted_pos.diff().abs().fillna(abs(shifted_pos))
	cost = turnover * (cost_bps / 1e4)
	pnl = gross_pnl - cost
	equity = (1.0 + pnl.fillna(0.0)).cumprod()
	total_return, sharpe, max_dd, win_rate, n_trades = compute_metrics(pnl.fillna(0.0))
	return BacktestResult(
		equity_curve=equity,
		total_return=float(total_return),
		sharpe=float(sharpe),
		max_drawdown=float(max_dd),
		win_rate=float(win_rate),
		n_trades=int(n_trades),
	)
