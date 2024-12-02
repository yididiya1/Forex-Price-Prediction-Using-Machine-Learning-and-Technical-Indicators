from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProjectPaths:
	root: Path = Path.cwd()
	data_dir: Path = Path.cwd() / "data"
	models_dir: Path = Path.cwd() / "models"

	def ensure(self) -> None:
		self.data_dir.mkdir(parents=True, exist_ok=True)
		self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Defaults:
	symbol: str = "EURUSD=X"  # Yahoo Finance forex ticker
	interval: str = "1d"      # 1d, 1h, 15m supported by yfinance
	start: str = "2010-01-01"
	end: Optional[str] = None
	forecast_horizon: int = 1  # predict 1 step ahead return
	test_size: float = 0.2
	random_state: int = 42


PATHS = ProjectPaths()
PATHS.ensure()
