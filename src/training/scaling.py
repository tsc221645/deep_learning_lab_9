"""Time-safe target normalization and inverse transforms."""

from typing import Literal

import numpy as np
import pandas as pd

ScalingMode = Literal["none", "global", "per_series"]


class TemporalScaler:
    """Fit target statistics only on an explicitly selected training interval."""

    def __init__(self, mode: ScalingMode, epsilon: float = 1e-8) -> None:
        if mode not in {"none", "global", "per_series"}:
            raise ValueError(f"Unsupported scaling mode: {mode}")
        self.mode = mode
        self.epsilon = epsilon
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None

    def fit(
        self,
        values: np.ndarray,
        timestamps: pd.DatetimeIndex,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> "TemporalScaler":
        values = np.asarray(values, dtype=np.float64)
        timestamps = pd.DatetimeIndex(timestamps)
        if values.ndim != 2 or values.shape[1] != len(timestamps):
            raise ValueError("values must have shape (series, timestamps)")
        if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
            raise ValueError("Training timestamps must be unique and increasing")
        train_start, train_end = pd.Timestamp(train_start), pd.Timestamp(train_end)
        mask = (timestamps >= train_start) & (timestamps < train_end)
        if not mask.any():
            raise ValueError("Training interval contains no observations")
        training_values = values[:, mask]
        if not np.isfinite(training_values).all():
            raise ValueError("Training targets must be finite")

        if self.mode == "none":
            center = np.zeros((1, 1), dtype=np.float64)
            scale = np.ones((1, 1), dtype=np.float64)
        elif self.mode == "global":
            center = np.array([[training_values.mean()]])
            scale = np.array([[training_values.std()]])
        else:
            center = training_values.mean(axis=1, keepdims=True)
            scale = training_values.std(axis=1, keepdims=True)
        scale = np.where(scale < self.epsilon, 1.0, scale)

        self.center_ = center
        self.scale_ = scale
        self.fit_start_ = train_start
        self.fit_end_ = train_end
        return self

    def _check_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Call fit before transforming values")
        return self.center_, self.scale_

    def transform(self, values: np.ndarray) -> np.ndarray:
        center, scale = self._check_fitted()
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or (
            self.mode == "per_series" and values.shape[0] != center.shape[0]
        ):
            raise ValueError("values must have shape (series, timestamps)")
        return ((values - center) / scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        center, scale = self._check_fitted()
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or (
            self.mode == "per_series" and values.shape[0] != center.shape[0]
        ):
            raise ValueError("values must have shape (series, horizon)")
        return values * scale + center


def normalize_windows(
    history: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each window using its history only, and return inverse stats."""
    history = np.asarray(history, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if history.ndim < 1 or targets.shape[:-1] != history.shape[:-1]:
        raise ValueError("History and targets must share their leading dimensions")
    center = history.mean(axis=-1, keepdims=True)
    scale = history.std(axis=-1, keepdims=True)
    scale = np.where(scale < epsilon, 1.0, scale)
    return (
        ((history - center) / scale).astype(np.float32),
        ((targets - center) / scale).astype(np.float32),
        center,
        scale,
    )


def inverse_window_normalization(
    values: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Return window-normalized predictions to their original kWh scale."""
    return np.asarray(values) * scale + center
