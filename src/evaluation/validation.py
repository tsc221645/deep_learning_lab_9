"""Chronological validation scoring without touching the test quarter."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.data.splits import DEFAULT_SPLITS
from src.evaluation.metrics import evaluate_predictions, seasonal_mase_scales


def _slice(panel: Panel, start: pd.Timestamp, end: pd.Timestamp) -> Panel:
    mask = (panel.timestamps >= start) & (panel.timestamps < end)
    return Panel(panel.client_ids, panel.timestamps[mask], panel.values[:, mask])


def score_validation(model: Any, panel: Panel,
                     config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit on train and score Jul-Sep daily origins with only past observations."""
    train_start, train_end = DEFAULT_SPLITS["train"]
    val_start, val_end = DEFAULT_SPLITS["validation"]
    history_months = int(config.get("history_months", 30))
    train_start = max(train_start, train_end - pd.DateOffset(months=history_months))
    train = _slice(panel, train_start, train_end)
    validation = _slice(panel, val_start, val_end)
    metadata = model.fit(train, validation, config)
    scales = seasonal_mase_scales(
        train.values, train.timestamps, train.client_ids,
        train.timestamps[0], train_end,
        seasonality=int(config.get("mase_seasonality", 24)),
    )["mase_scale"].to_numpy()
    horizon = int(config.get("horizon", 24))
    lookback = int(config.get("lookback", 168))
    rows = []
    for origin in pd.date_range(val_start, val_end, freq="D", inclusive="left"):
        position = panel.timestamps.get_loc(origin)
        if position < lookback or position + horizon > len(panel.timestamps):
            raise ValueError(f"Incomplete validation origin: {origin}")
        history = Panel(panel.client_ids, panel.timestamps[:position],
                        panel.values[:, :position])
        times = panel.timestamps[position:position + horizon]
        future = calendar_features(times, encoding=str(config.get(
            "calendar_encoding", "cyclic")))
        prediction = np.asarray(
            model.predict(history, future, horizon=horizon, origin=origin),
            dtype=np.float64,
        )
        if prediction.shape != (len(panel.client_ids), horizon):
            raise ValueError("Validation forecast has incorrect shape")
        actual = panel.values[:, position:position + horizon]
        rows.append(pd.DataFrame({
            "unique_id": np.repeat(panel.client_ids, horizon),
            "ds": np.tile(times.to_numpy(), len(panel.client_ids)),
            "y_hat": prediction.reshape(-1),
            "y_true": actual.reshape(-1),
            "mase_scale": np.repeat(scales, horizon),
            "horizon_step": np.tile(np.arange(1, horizon + 1),
                                    len(panel.client_ids)),
        }))
    predictions = pd.concat(rows, ignore_index=True)
    metrics = evaluate_predictions(predictions).overall
    return {"metrics": metrics, "fit_metadata": metadata,
            "predictions": predictions}
