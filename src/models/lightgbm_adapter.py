"""Global direct multi-horizon LightGBM forecaster."""

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.evaluation.metrics import seasonal_mase_scales
from src.utils.seed import set_seed

LAGS = tuple(range(1, 25)) + (168,)
HOUR = pd.Timedelta(1, unit="h")


class LightGBMDirectAdapter:
    """Train one global LightGBM regressor per forecast horizon step."""

    name = "LightGBM-direct"

    def __init__(self) -> None:
        self.models = []
        self.horizon = 24
        self.client_ids: tuple[str, ...] = ()
        self.calendar_encoding = "cyclic"
        self.calendar_columns: tuple[str, ...] = ()
        self.feature_names: list[str] = []
        self.fit_metadata: dict[str, Any] = {}

    def fit(
        self,
        train: Panel,
        val: Panel | None,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        from lightgbm import LGBMRegressor, early_stopping, record_evaluation

        self.horizon = int(config.get("horizon", 24))
        if self.horizon != 24:
            raise ValueError("LightGBMDirectAdapter implements the required H=24")
        self.calendar_encoding = str(config.get("calendar_encoding", "cyclic"))
        self.client_ids = train.client_ids
        self.calendar_columns = tuple(
            calendar_features(
                train.timestamps[:1], encoding=self.calendar_encoding
            ).columns
        )
        self.feature_names = [f"lag_{lag}" for lag in LAGS] + list(
            self.calendar_columns
        )
        minimum_lookback = max(168, int(config.get("lookback", 168)))
        training_stride = int(config.get("training_stride", 24))
        training_origins = np.arange(
            minimum_lookback,
            len(train.timestamps) - self.horizon + 1,
            training_stride,
            dtype=np.int64,
        )
        training_origins = _sample_origins(
            training_origins, config.get("max_train_windows_per_client", 256)
        )
        if not len(training_origins):
            raise ValueError("Training panel does not contain complete lag windows")
        training_calendar = calendar_features(
            train.timestamps, encoding=self.calendar_encoding
        ).to_numpy(dtype=np.float32)
        train_lags, train_targets = _lag_and_target_matrices(
            train.values,
            training_origins,
            self.horizon,
        )

        validation_lags = validation_targets = None
        validation_origins = np.empty(0, dtype=np.int64)
        validation_calendar = None
        if val is not None:
            if val.client_ids != train.client_ids:
                raise ValueError("Training and validation client IDs must match")
            if len(val.timestamps) and train.timestamps[-1] + HOUR != val.timestamps[0]:
                raise ValueError("Training and validation panels must be contiguous")
            combined_values = np.concatenate((train.values, val.values), axis=1)
            combined_timestamps = train.timestamps.append(val.timestamps)
            validation_origins = np.arange(
                len(train.timestamps),
                len(combined_timestamps) - self.horizon + 1,
                int(config.get("validation_stride", 24)),
                dtype=np.int64,
            )
            if not len(validation_origins):
                raise ValueError("Validation panel does not contain complete windows")
            validation_calendar = calendar_features(
                combined_timestamps, encoding=self.calendar_encoding
            ).to_numpy(dtype=np.float32)
            validation_lags, validation_targets = _lag_and_target_matrices(
                combined_values,
                validation_origins,
                self.horizon,
            )

        scale_table = seasonal_mase_scales(
            train.values,
            train.timestamps,
            train.client_ids,
            train.timestamps[0],
            train.timestamps[-1] + HOUR,
            seasonality=int(config.get("mase_seasonality", 24)),
        )
        mase_scales = scale_table["mase_scale"].to_numpy(dtype=np.float64)
        mase_evaluator = _mase_metric(len(train.client_ids), mase_scales)
        seed = int(config.get("seed", 42))
        set_seed(seed)
        patience = int(config.get("patience", 30))
        maximum_estimators = int(config.get("n_estimators", 500))
        if maximum_estimators < 1:
            raise ValueError("n_estimators must be positive")
        fixed_estimators = config.get("final_epochs", config.get("best_epoch"))
        if val is None and fixed_estimators is not None:
            maximum_estimators = int(fixed_estimators)
        if maximum_estimators < 1:
            raise ValueError("The fixed LightGBM iteration count must be positive")

        self.models = []
        history_rows = []
        best_iterations = []
        for horizon_step in range(self.horizon):
            train_features = _features_for_step(
                train_lags,
                training_origins,
                horizon_step,
                len(train.client_ids),
                training_calendar,
            )
            estimator = LGBMRegressor(
                objective="regression",
                n_estimators=maximum_estimators,
                learning_rate=float(config.get("learning_rate", 0.05)),
                num_leaves=int(config.get("num_leaves", 31)),
                max_depth=int(config.get("max_depth", -1)),
                min_child_samples=int(config.get("min_child_samples", 20)),
                subsample=float(config.get("subsample", 1.0)),
                colsample_bytree=float(config.get("colsample_bytree", 1.0)),
                reg_lambda=float(config.get("reg_lambda", 0.0)),
                random_state=seed + horizon_step,
                n_jobs=int(config.get("n_jobs", -1)),
                verbosity=-1,
            )
            evaluations: dict[str, dict[str, list[float]]] = {}
            fit_kwargs: dict[str, Any] = {
                "eval_metric": ["l1", "l2"]
                + ([mase_evaluator] if mase_evaluator is not None else []),
                "callbacks": [record_evaluation(evaluations)],
            }
            if validation_lags is not None and validation_targets is not None:
                validation_features = _features_for_step(
                    validation_lags,
                    validation_origins,
                    horizon_step,
                    len(train.client_ids),
                    validation_calendar,
                )
                fit_kwargs["eval_X"] = (
                    np.ascontiguousarray(train_features),
                    np.ascontiguousarray(validation_features),
                )
                fit_kwargs["eval_y"] = (
                    np.ascontiguousarray(train_targets[:, horizon_step]),
                    np.ascontiguousarray(validation_targets[:, horizon_step]),
                )
                fit_kwargs["eval_names"] = ["train", "validation"]
                fit_kwargs["callbacks"].append(
                    early_stopping(stopping_rounds=patience, verbose=False)
                )
            estimator.fit(
                np.ascontiguousarray(train_features),
                np.ascontiguousarray(train_targets[:, horizon_step]),
                **fit_kwargs,
            )
            self.models.append(estimator)
            best_iteration = int(estimator.best_iteration_ or maximum_estimators)
            best_iterations.append(best_iteration)
            for split, metrics in evaluations.items():
                for metric_name, values in metrics.items():
                    history_rows.extend(
                        {
                            "horizon_step": horizon_step + 1,
                            "iteration": iteration,
                            "split": split,
                            "metric": metric_name,
                            "value": float(value),
                        }
                        for iteration, value in enumerate(values, start=1)
                    )

        self.fit_metadata = {
            "implementation": "lightgbm",
            "strategy": "direct, one regressor per horizon step",
            "lags": list(LAGS),
            "calendar_features": list(self.calendar_columns),
            "train_windows_per_client": int(len(training_origins)),
            "validation_windows_per_client": int(len(validation_origins)),
            "best_iterations": best_iterations,
            "best_iteration_median": int(np.median(best_iterations)),
            "max_estimators": maximum_estimators,
            "mase_undefined_series_count": int((mase_scales <= 0).sum()),
            "history": pd.DataFrame(history_rows),
        }
        self._save_artifacts(config)
        return {
            key: value
            for key, value in self.fit_metadata.items()
            if key != "history"
        } | self._artifact_metadata

    def predict(
        self,
        history: Panel,
        future_calendar: pd.DataFrame,
        horizon: int = 24,
        origin: pd.Timestamp | None = None,
    ) -> np.ndarray:
        if len(self.models) != self.horizon:
            raise RuntimeError("Call fit before predict")
        if horizon != self.horizon or len(future_calendar) != horizon:
            raise ValueError("Forecast horizon differs from the fitted horizon")
        if history.client_ids != self.client_ids:
            raise ValueError("Prediction client IDs differ from the fitted panel")
        if len(history.timestamps) < max(LAGS):
            raise ValueError("LightGBM requires at least 168 historical observations")
        if origin is not None:
            if history.timestamps[-1] != origin - HOUR:
                raise ValueError("History must end one hour before the origin")
            if future_calendar.index[0] != origin:
                raise ValueError("Future calendar must start at the forecast origin")

        expected_calendar = calendar_features(
            future_calendar.index, encoding=self.calendar_encoding
        )
        if not future_calendar.equals(expected_calendar):
            raise ValueError("Future calendar features do not match their timestamps")
        lag_values = history.values[
            :, len(history.timestamps) - np.asarray(LAGS, dtype=np.int64)
        ].astype(np.float32)
        forecast = np.empty((len(self.client_ids), horizon), dtype=np.float64)
        calendar_values = future_calendar.to_numpy(dtype=np.float32)
        for step, model in enumerate(self.models):
            features = np.concatenate(
                (
                    lag_values,
                    np.broadcast_to(
                        calendar_values[step],
                        (len(self.client_ids), len(self.calendar_columns)),
                    ),
                ),
                axis=1,
            )
            forecast[:, step] = model.predict(features)
        return forecast

    def parameter_counts(self) -> tuple[int | None, int | None]:
        if not self.models:
            return None, None
        leaf_count = sum(
            tree["num_leaves"]
            for model in self.models
            for tree in model.booster_.dump_model()["tree_info"]
        )
        return int(leaf_count), int(leaf_count)

    def _save_artifacts(self, config: Mapping[str, Any]) -> None:
        self._artifact_metadata: dict[str, Any] = {}
        artifact_dir = config.get("artifact_dir")
        if artifact_dir is None:
            return
        config_id = str(config.get("config_id", "unidentified"))
        run_dir = Path(artifact_dir) / "lightgbm-direct" / config_id
        refit_month = config.get("refit_month")
        if refit_month is not None:
            run_dir = run_dir / str(refit_month)[:10]
        run_dir.mkdir(parents=True, exist_ok=True)
        history_path = run_dir / "training_log.csv"
        self.fit_metadata["history"].to_csv(history_path, index=False)
        model_paths = []
        for step, model in enumerate(self.models, start=1):
            model_path = run_dir / f"model_h{step:02d}.txt"
            model.booster_.save_model(str(model_path))
            model_paths.append(str(model_path))
        self._artifact_metadata.update(
            {
                "training_log": str(history_path),
                "model_files": model_paths,
            }
        )


def _lag_and_target_matrices(
    values: np.ndarray,
    origins: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    lag_features = []
    targets = []
    for origin in origins:
        lag_features.append(
            values[:, origin - np.asarray(LAGS, dtype=np.int64)]
        )
        targets.append(values[:, origin : origin + horizon].T)
    if not lag_features:
        raise ValueError("Cannot create LightGBM features without origins")
    lag_matrix = np.concatenate(lag_features, axis=0).astype(np.float32)
    target_matrix = np.concatenate(targets, axis=1).T.astype(np.float32)
    return lag_matrix, target_matrix


def _features_for_step(
    lag_matrix: np.ndarray,
    origins: np.ndarray,
    step: int,
    n_series: int,
    calendar_values: np.ndarray | None,
) -> np.ndarray:
    if calendar_values is None:
        raise ValueError("Calendar features are required for LightGBM")
    future_calendar = np.concatenate(
        [
            np.broadcast_to(
                calendar_values[origin + step],
                (n_series, calendar_values.shape[1]),
            )
            for origin in origins
        ],
        axis=0,
    )
    return np.concatenate((lag_matrix, future_calendar), axis=1)


def _mase_metric(n_series: int, scales: np.ndarray):
    defined_scales = scales > 0
    if not defined_scales.any():
        return None

    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        if len(y_true) % n_series:
            raise ValueError("LightGBM rows do not align with client series")
        series_positions = np.tile(
            np.arange(n_series, dtype=np.int64), len(y_true) // n_series
        )
        absolute_errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        sums = np.bincount(
            series_positions,
            weights=absolute_errors,
            minlength=n_series,
        )
        counts = np.bincount(series_positions, minlength=n_series)
        mean_errors = sums / counts
        mase = float(np.mean(mean_errors[defined_scales] / scales[defined_scales]))
        return "mase", mase, False

    return evaluate


def _sample_origins(
    origins: np.ndarray, maximum: int | None
) -> np.ndarray:
    if maximum is None:
        return origins
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("max_train_windows_per_client must be positive")
    if len(origins) <= maximum:
        return origins
    positions = np.linspace(0, len(origins) - 1, num=maximum, dtype=np.int64)
    return origins[positions]
