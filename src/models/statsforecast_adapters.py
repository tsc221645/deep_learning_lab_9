"""Statistical baselines backed by StatsForecast model implementations."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.data.panel import Panel

HOUR = pd.Timedelta(1, unit="h")


class StatsForecastAdapter:
    """Use a StatsForecast model per client and forward new observations safely."""

    def __init__(self, model_name: str, season_length: int | None = None) -> None:
        allowed = {"Naive", "SeasonalNaive", "AutoETS"}
        if model_name not in allowed:
            raise ValueError(f"Unsupported StatsForecast model: {model_name}")
        if model_name == "SeasonalNaive" and season_length not in {24, 168}:
            raise ValueError("SeasonalNaive season_length must be 24 or 168")
        if model_name == "AutoETS" and season_length != 24:
            raise ValueError("AutoETS is configured with daily seasonality (24)")
        self.model_name = model_name
        self.season_length = season_length
        self.name = (
            f"{model_name}{season_length}"
            if season_length is not None
            else model_name
        )
        self.ets_model = "AAA"
        self.models = []
        self.client_ids: tuple[str, ...] = ()
        self.fit_end: pd.Timestamp | None = None

    def fit(
        self,
        train: Panel,
        val: Panel | None,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        from statsforecast.models import AutoETS, Naive, SeasonalNaive

        model_factory = {
            "Naive": Naive,
            "SeasonalNaive": lambda: SeasonalNaive(self.season_length),
            "AutoETS": lambda: AutoETS(
                season_length=24, model=self.ets_model
            ),
        }[self.model_name]
        self.client_ids = train.client_ids
        self.fit_end = train.timestamps[-1] + HOUR
        self.models = []
        for values in train.values:
            model = model_factory()
            model.fit(np.asarray(values, dtype=np.float64))
            self.models.append(model)
        return {
            "implementation": "statsforecast",
            "model": self.name,
            "season_length": self.season_length,
            "ets_model": self.ets_model if self.model_name == "AutoETS" else None,
            "fit_end_exclusive": self.fit_end.isoformat(),
            "validation_used_for_fitting": False,
        }

    def predict(
        self,
        history: Panel,
        future_calendar: pd.DataFrame,
        horizon: int = 24,
        origin: pd.Timestamp | None = None,
    ) -> np.ndarray:
        if not self.models or self.fit_end is None:
            raise RuntimeError("Call fit before predict")
        if history.client_ids != self.client_ids:
            raise ValueError("Prediction client IDs differ from the fitted panel")
        if len(future_calendar) != horizon:
            raise ValueError("Future calendar length differs from horizon")
        if origin is not None:
            if history.timestamps[-1] != origin - HOUR:
                raise ValueError("History must end one hour before the origin")
            if future_calendar.index[0] != origin:
                raise ValueError("Future calendar must start at the forecast origin")

        new_start = int(history.timestamps.searchsorted(self.fit_end))
        new_values = history.values[:, new_start:]
        forecasts = np.empty((len(self.client_ids), horizon), dtype=np.float64)
        for position, model in enumerate(self.models):
            if self.model_name == "AutoETS" and len(new_values[position]):
                result = model.forward(
                    np.asarray(new_values[position], dtype=np.float64), horizon
                )
            elif self.model_name != "AutoETS":
                result = model.forward(
                    np.asarray(history.values[position], dtype=np.float64), horizon
                )
            else:
                result = model.predict(horizon)
            forecasts[position] = np.asarray(result["mean"], dtype=np.float64)
        return forecasts


def naive_adapter() -> StatsForecastAdapter:
    return StatsForecastAdapter("Naive")


def seasonal_naive_24_adapter() -> StatsForecastAdapter:
    return StatsForecastAdapter("SeasonalNaive", season_length=24)


def seasonal_naive_168_adapter() -> StatsForecastAdapter:
    return StatsForecastAdapter("SeasonalNaive", season_length=168)


def auto_ets_adapter() -> StatsForecastAdapter:
    return StatsForecastAdapter("AutoETS", season_length=24)
