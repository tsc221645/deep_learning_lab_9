"""Shared PyTorch implementations for MLP, DLinear, and NLinear models."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.calendars import calendar_features
from src.data.panel import Panel
from src.data.windows import MultiSeriesWindowDataset
from src.evaluation.metrics import seasonal_mase_scales
from src.training.scaling import TemporalScaler
from src.training.trainer import Trainer
from src.utils.seed import set_seed

HOUR = pd.Timedelta(1, unit="h")


class MultiOutputMLP(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        calendar_features_count: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_size = lookback * (calendar_features_count + 1)
        input_size += horizon * calendar_features_count
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(
        self, history: torch.Tensor, future_calendar: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            (history.flatten(start_dim=1), future_calendar.flatten(start_dim=1)),
            dim=1,
        )
        return self.network(combined)


class DLinear(nn.Module):
    def __init__(
        self, lookback: int, horizon: int, moving_avg_window: int = 25
    ) -> None:
        super().__init__()
        if moving_avg_window < 1 or moving_avg_window % 2 == 0:
            raise ValueError("moving_avg_window must be a positive odd integer")
        if moving_avg_window > lookback:
            raise ValueError("moving_avg_window cannot exceed lookback")
        self.lookback = lookback
        self.moving_avg_window = moving_avg_window
        self.seasonal_projection = nn.Linear(lookback, horizon)
        self.trend_projection = nn.Linear(lookback, horizon)

    def forward(
        self, history: torch.Tensor, future_calendar: torch.Tensor
    ) -> torch.Tensor:
        values = history[..., 0]
        padding = self.moving_avg_window // 2
        trend = nn.functional.avg_pool1d(
            nn.functional.pad(
                values.unsqueeze(1), (padding, padding), mode="replicate"
            ),
            kernel_size=self.moving_avg_window,
            stride=1,
        ).squeeze(1)
        seasonal = values - trend
        return self.seasonal_projection(seasonal) + self.trend_projection(trend)


class NLinear(nn.Module):
    def __init__(self, lookback: int, horizon: int) -> None:
        super().__init__()
        self.projection = nn.Linear(lookback, horizon)

    def forward(
        self, history: torch.Tensor, future_calendar: torch.Tensor
    ) -> torch.Tensor:
        values = history[..., 0]
        last_value = values[:, -1:]
        return self.projection(values - last_value) + last_value


class TorchForecastAdapter:
    """Train one shared PyTorch model across the panel and invert its scaling."""

    uses_calendar = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.module: nn.Module | None = None
        self.scaler: TemporalScaler | None = None

    def build_module(
        self,
        lookback: int,
        horizon: int,
        calendar_features_count: int,
        config: Mapping[str, Any],
    ) -> nn.Module:
        raise NotImplementedError

    def fit(
        self,
        train: Panel,
        val: Panel | None,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        lookback = int(config.get("lookback", 168))
        horizon = int(config.get("horizon", 24))
        seed = int(config.get("seed", 42))
        normalization = str(config.get("normalization", "per_series"))
        scaling_mode = {"raw": "none"}.get(normalization, normalization)
        if scaling_mode not in {"none", "global", "per_series"}:
            raise ValueError(
                "Torch models support normalization none, global, or per_series"
            )
        if lookback < 1 or horizon < 1:
            raise ValueError("lookback and horizon must be positive")
        if val is not None and val.client_ids != train.client_ids:
            raise ValueError("Training and validation client IDs must match")
        if val is not None and len(val.timestamps):
            if train.timestamps[-1] + HOUR != val.timestamps[0]:
                raise ValueError("Training and validation panels must be contiguous")

        set_seed(seed)
        self.lookback = lookback
        self.horizon = horizon
        self.calendar_encoding = str(config.get("calendar_encoding", "cyclic"))
        self.client_ids = train.client_ids
        self.client_positions = {
            unique_id: index for index, unique_id in enumerate(self.client_ids)
        }
        train_end = train.timestamps[-1] + HOUR
        self.scaler = TemporalScaler(scaling_mode).fit(
            train.values,
            train.timestamps,
            train_start=train.timestamps[0],
            train_end=train_end,
        )
        scaled_train = self.scaler.transform(train.values)
        train_timestamps = train.timestamps

        if val is None:
            all_timestamps = train_timestamps
            all_values = scaled_train
            validation_origins = None
        else:
            scaled_val = self.scaler.transform(val.values)
            all_timestamps = train.timestamps.append(val.timestamps)
            all_values = np.concatenate((scaled_train, scaled_val), axis=1)
            validation_origins = np.arange(
                len(train.timestamps),
                len(all_timestamps) - horizon + 1,
                int(config.get("validation_stride", 24)),
                dtype=np.int64,
            )

        calendar = calendar_features(
            all_timestamps, encoding=self.calendar_encoding
        )
        train_origins = np.arange(
            lookback,
            len(train_timestamps) - horizon + 1,
            int(config.get("training_stride", 1)),
            dtype=np.int64,
        )
        train_origins = _sample_origins(
            train_origins,
            config.get("max_train_windows_per_client", 256),
        )
        if not len(train_origins):
            raise ValueError("Training panel does not contain a complete window")
        if validation_origins is not None and not len(validation_origins):
            raise ValueError("Validation panel does not contain a complete window")

        train_dataset = MultiSeriesWindowDataset(
            all_values,
            all_timestamps,
            train.client_ids,
            train_origins,
            lookback=lookback,
            horizon=horizon,
            calendar=calendar,
        )
        val_dataset = (
            MultiSeriesWindowDataset(
                all_values,
                all_timestamps,
                train.client_ids,
                validation_origins,
                lookback=lookback,
                horizon=horizon,
                calendar=calendar,
            )
            if validation_origins is not None
            else None
        )

        device = torch.device(
            str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        batch_size = int(config.get("batch_size", 512))
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=int(config.get("num_workers", 0)),
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(config.get("num_workers", 0)),
            )
            if val_dataset is not None
            else None
        )

        module = self.build_module(
            lookback,
            horizon,
            calendar.shape[1],
            config,
        )
        self.module = module
        scale_table = seasonal_mase_scales(
            train.values,
            train.timestamps,
            train.client_ids,
            train.timestamps[0],
            train_end,
            seasonality=int(config.get("mase_seasonality", 24)),
        )
        self.mase_scales = dict(
            zip(scale_table["unique_id"], scale_table["mase_scale"], strict=True)
        )
        trainer = Trainer(device)
        fixed_epochs = None
        if val_loader is None:
            fixed_epochs = config.get("final_epochs", config.get("best_epoch"))
            if fixed_epochs is not None:
                fixed_epochs = int(fixed_epochs)
        result = trainer.fit(
            module,
            train_loader,
            val_loader,
            inverse_targets=self._inverse_batch,
            mase_scales=self.mase_scales,
            learning_rate=float(config.get("learning_rate", 1e-3)),
            max_epochs=int(config.get("max_epochs", 20)),
            patience=int(config.get("patience", 5)),
            fixed_epochs=fixed_epochs,
            min_delta=float(config.get("min_delta", 0.0)),
        )
        self.module.eval()
        self.device = device
        return self._training_metadata(result, train_origins, validation_origins, config)

    def predict(
        self,
        history: Panel,
        future_calendar: pd.DataFrame,
        horizon: int = 24,
        origin: pd.Timestamp | None = None,
    ) -> np.ndarray:
        if self.module is None or self.scaler is None:
            raise RuntimeError("Call fit before predict")
        if history.client_ids != self.client_ids:
            raise ValueError("Prediction client IDs differ from the fitted panel")
        if horizon != self.horizon or len(future_calendar) != horizon:
            raise ValueError("Forecast horizon differs from the fitted horizon")
        if len(history.timestamps) < self.lookback:
            raise ValueError("Not enough history for the requested lookback")
        if origin is not None and history.timestamps[-1] != origin - HOUR:
            raise ValueError("History must end one hour before the forecast origin")
        expected_calendar = calendar_features(
            future_calendar.index, encoding=self.calendar_encoding
        )
        if not future_calendar.equals(expected_calendar):
            raise ValueError("Future calendar features do not match their timestamps")

        timestamps = history.timestamps[-self.lookback :]
        normalized = self.scaler.transform(
            history.values[:, -self.lookback :]
        )
        history_values = normalized
        if self.uses_calendar:
            historical_calendar = calendar_features(
                timestamps, encoding=self.calendar_encoding
            ).to_numpy(dtype=np.float32)
            historical_calendar = np.broadcast_to(
                historical_calendar[None, :, :],
                (len(self.client_ids), self.lookback, historical_calendar.shape[1]),
            )
            model_history = np.concatenate(
                (history_values[:, :, None], historical_calendar), axis=2
            )
            future_values = np.broadcast_to(
                future_calendar.to_numpy(dtype=np.float32)[None, :, :],
                (
                    len(self.client_ids),
                    horizon,
                    future_calendar.shape[1],
                ),
            )
        else:
            model_history = history_values[:, :, None]
            future_values = np.zeros(
                (len(self.client_ids), horizon, future_calendar.shape[1]),
                dtype=np.float32,
            )

        with torch.no_grad():
            forecast = self.module(
                torch.from_numpy(np.ascontiguousarray(model_history)).to(self.device),
                torch.from_numpy(np.ascontiguousarray(future_values)).to(self.device),
            )
        return self.scaler.inverse_transform(
            forecast.detach().cpu().numpy().astype(np.float64)
        )

    def _inverse_batch(
        self, values: torch.Tensor, unique_ids: Sequence[str]
    ) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("The scaler is not fitted")
        normalized = values.detach().cpu().numpy().astype(np.float64)
        if self.scaler.mode == "none":
            return normalized
        if self.scaler.mode == "global":
            return normalized * self.scaler.scale_[0, 0] + self.scaler.center_[0, 0]
        positions = np.asarray(
            [self.client_positions[unique_id] for unique_id in unique_ids],
            dtype=np.int64,
        )
        return (
            normalized * self.scaler.scale_[positions]
            + self.scaler.center_[positions]
        )

    def _training_metadata(
        self,
        result,
        train_origins: np.ndarray,
        validation_origins: np.ndarray | None,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "epochs_trained": len(result.history),
            "train_windows_per_client": int(len(train_origins)),
            "validation_windows_per_client": (
                int(len(validation_origins))
                if validation_origins is not None
                else 0
            ),
            "normalization": self.scaler.mode if self.scaler is not None else None,
            "device": str(self.device),
        }
        artifact_dir = config.get("artifact_dir")
        if artifact_dir is not None:
            config_id = str(config.get("config_id", "unidentified"))
            slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.name).strip("-").lower()
            run_dir = Path(artifact_dir) / slug / config_id
            refit_month = config.get("refit_month")
            if refit_month is not None:
                run_dir = run_dir / str(refit_month)[:10]
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "training_log.csv"
            checkpoint_path = run_dir / "model_state.pt"
            if self.module is None:
                raise RuntimeError("The model is not fitted")
            result.history.to_csv(log_path, index=False)
            torch.save(self.module.state_dict(), checkpoint_path)
            metadata["training_log"] = str(log_path)
            metadata["checkpoint"] = str(checkpoint_path)
        return metadata


class MLPAdapter(TorchForecastAdapter):
    uses_calendar = True

    def __init__(self) -> None:
        super().__init__("MLP")

    def build_module(
        self,
        lookback: int,
        horizon: int,
        calendar_features_count: int,
        config: Mapping[str, Any],
    ) -> nn.Module:
        return MultiOutputMLP(
            lookback,
            horizon,
            calendar_features_count,
            hidden_size=int(config.get("hidden_size", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )


class DLinearAdapter(TorchForecastAdapter):
    def __init__(self) -> None:
        super().__init__("DLinear")

    def build_module(
        self,
        lookback: int,
        horizon: int,
        calendar_features_count: int,
        config: Mapping[str, Any],
    ) -> nn.Module:
        return DLinear(
            lookback,
            horizon,
            moving_avg_window=int(config.get("moving_avg_window", 25)),
        )


class NLinearAdapter(TorchForecastAdapter):
    def __init__(self) -> None:
        super().__init__("NLinear")

    def build_module(
        self,
        lookback: int,
        horizon: int,
        calendar_features_count: int,
        config: Mapping[str, Any],
    ) -> nn.Module:
        return NLinear(lookback, horizon)


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
