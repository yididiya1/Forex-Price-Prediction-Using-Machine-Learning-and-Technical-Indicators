from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error


TaskType = Literal["classification", "regression"]


@dataclass
class TrainedModel:
	model: object
	task: TaskType
	features: list[str]

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump({
			"model": self.model,
			"task": self.task,
			"features": self.features,
		}, path)

	@staticmethod
	def load(path: Path) -> "TrainedModel":
		obj = joblib.load(path)
		return TrainedModel(model=obj["model"], task=obj["task"], features=obj["features"])


def create_model(task: TaskType, random_state: int = 42):
	if task == "regression":
		return RandomForestRegressor(
			n_estimators=300,
			max_depth=None,
			random_state=random_state,
			n_jobs=-1,
		)
	else:
		return RandomForestClassifier(
			n_estimators=300,
			max_depth=None,
			random_state=random_state,
			n_jobs=-1,
		)


def train(
	X_train,
	y_train,
	task: TaskType,
	random_state: int = 42,
) -> TrainedModel:
	model = create_model(task=task, random_state=random_state)
	model.fit(X_train, y_train)
	return TrainedModel(model=model, task=task, features=list(X_train.columns))


def evaluate(model: TrainedModel, X_test, y_test) -> dict:
	if model.task == "regression":
		pred = model.model.predict(X_test)
		mse = mean_squared_error(y_test, pred)
		return {"mse": float(mse)}
	else:
		proba = model.model.predict_proba(X_test)[:, 1]
		pred = (proba > 0.5).astype(int)
		acc = accuracy_score(y_test, pred)
		return {"accuracy": float(acc)}
