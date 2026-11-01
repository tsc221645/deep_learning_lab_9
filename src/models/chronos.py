"""Chronos-Bolt zero-shot adapter for the shared day-ahead backtest."""

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.panel import Panel


class ChronosBoltAdapter:
    name = "Chronos-Bolt"

    def __init__(
        self, model_id: str = "amazon/chronos-bolt-small",
        context_length: int = 168,
        batch_size: int = 32,
        device_map: str = "cuda",
    ) -> None:
        if context_length < 1 or batch_size < 1:
            raise ValueError("context_length and batch_size must be positive")
        self.model_id = model_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.device_map = device_map
        self.pipeline: Any = None
        self.quantiles_by_origin: dict[pd.Timestamp, np.ndarray] = {}
        self.client_ids: tuple[str, ...] | None = None

    def _pipeline(self) -> Any:
        if self.pipeline is None:
            from chronos import BaseChronosPipeline

            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device_map
            )
        return self.pipeline

    def predict(
        self,
        history: Panel,
        future_calendar: pd.DataFrame,
        horizon: int = 24,
        origin: pd.Timestamp | None = None,
    ) -> np.ndarray:
        if origin is None or len(future_calendar) != horizon:
            raise ValueError("Chronos requires a dated, complete forecast horizon")
        if future_calendar.index[0] != origin:
            raise ValueError("Future calendar must start at the origin")
        if history.timestamps[-1] != origin - pd.Timedelta(hours=1):
            raise ValueError("History must end before the origin")
        if len(history.timestamps) < self.context_length:
            raise ValueError("Insufficient Chronos context")
        self.client_ids = history.client_ids
        pipeline = self._pipeline()
        context = history.values[:, -self.context_length :]
        forecasts = []
        quantiles = []
        for start in range(0, len(history.client_ids), self.batch_size):
            block = torch.as_tensor(
                np.ascontiguousarray(context[start:start + self.batch_size]),
                dtype=torch.float32,
            )
            q, _ = pipeline.predict_quantiles(
                block, prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
            q = np.asarray(q.detach().cpu(), dtype=np.float64)
            if q.shape != (len(block), horizon, 3):
                raise ValueError(f"Unexpected Chronos quantile shape: {q.shape}")
            if not np.isfinite(q).all() or (q[:, :, 0] > q[:, :, 2]).any():
                raise ValueError("Chronos returned invalid quantiles")
            forecasts.append(q[:, :, 1])
            quantiles.append(q)
        self.quantiles_by_origin[pd.Timestamp(origin)] = np.concatenate(quantiles)
        return np.concatenate(forecasts)

    def parameter_counts(self) -> tuple[int, int]:
        # Zero-shot weights are not trained by this experiment.
        if self.pipeline is None:
            return (0, 0)
        model = self.pipeline.model
        return (sum(p.numel() for p in model.parameters()), 0)

    def probabilistic_predictions(self, point_predictions: pd.DataFrame) -> pd.DataFrame:
        """Attach P10/P50/P90 by origin and client order after shared backtest."""
        if self.client_ids is None:
            raise RuntimeError("Run the shared backtest before exporting quantiles")
        frames = []
        for origin, group in point_predictions.groupby("origin", sort=True):
            q = self.quantiles_by_origin[pd.Timestamp(origin)]
            timestamps = pd.DatetimeIndex(group["ds"].unique()).sort_values()
            key = pd.MultiIndex.from_product(
                (self.client_ids, timestamps), names=("unique_id", "ds")
            )
            ordered = group.set_index(["unique_id", "ds"]).loc[key].reset_index()
            if len(ordered) != q.shape[0] * q.shape[1]:
                raise ValueError("Quantile cache does not match point predictions")
            for index, level in enumerate((10, 50, 90)):
                ordered[f"p{level}"] = q[:, :, index].reshape(-1)
            frames.append(ordered)
        return pd.concat(frames, ignore_index=True)


def probabilistic_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Mean pinball at P10/P90 and empirical interval coverage in kWh."""
    if (frame["p10"] > frame["p90"]).any():
        raise ValueError("P10 exceeds P90")
    actual = frame["y_true"].to_numpy(dtype=np.float64)
    metrics = {}
    for level, tau in ((10, 0.1), (90, 0.9)):
        error = actual - frame[f"p{level}"].to_numpy(dtype=np.float64)
        metrics[f"pinball_p{level}"] = float(
            np.maximum(tau * error, (tau - 1) * error).mean()
        )
    metrics["coverage_p10_p90"] = float(
        ((actual >= frame["p10"].to_numpy()) &
         (actual <= frame["p90"].to_numpy())).mean()
    )
    return metrics
