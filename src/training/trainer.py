"""Reproducible PyTorch training with validation-only early stopping."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float | None


class Trainer:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        *,
        inverse_targets: Callable[
            [torch.Tensor, Sequence[str]], np.ndarray
        ],
        mase_scales: Mapping[str, float],
        learning_rate: float = 1e-3,
        max_epochs: int = 20,
        patience: int = 5,
        fixed_epochs: int | None = None,
        min_delta: float = 0.0,
    ) -> TrainingResult:
        if not len(train_loader.dataset):
            raise ValueError("Training windows cannot be empty")
        if val_loader is not None and not len(val_loader.dataset):
            raise ValueError("Validation windows cannot be empty")
        if max_epochs < 1 or patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a nonnegative finite number")
        if fixed_epochs is not None and fixed_epochs < 1:
            raise ValueError("fixed_epochs must be positive")
        if val_loader is not None and fixed_epochs is not None:
            raise ValueError("Fixed-epoch refitting cannot also use validation")

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch_limit = fixed_epochs or max_epochs
        best_state = None
        best_epoch = epoch_limit
        best_validation_loss = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        records = []

        for epoch in range(1, epoch_limit + 1):
            train_metrics = self._run_epoch(
                model,
                train_loader,
                inverse_targets,
                mase_scales,
                optimizer,
            )
            validation_metrics = (
                self._run_epoch(
                    model,
                    val_loader,
                    inverse_targets,
                    mase_scales,
                    optimizer=None,
                )
                if val_loader is not None
                else None
            )
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae_kwh": train_metrics["mae"],
                "train_mase": train_metrics["mase"],
                "validation_loss": (
                    validation_metrics["loss"] if validation_metrics else None
                ),
                "validation_mae_kwh": (
                    validation_metrics["mae"] if validation_metrics else None
                ),
                "validation_mase": (
                    validation_metrics["mase"] if validation_metrics else None
                ),
            }
            records.append(record)

            if validation_metrics is None:
                continue
            validation_loss = validation_metrics["loss"]
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_validation_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if val_loader is not None:
            if best_state is None:
                raise RuntimeError("Training did not produce a valid best checkpoint")
            model.load_state_dict(best_state)
        return TrainingResult(
            history=pd.DataFrame(records),
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
        )

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        inverse_targets: Callable[[torch.Tensor, Sequence[str]], np.ndarray],
        mase_scales: Mapping[str, float],
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, float | None]:
        training = optimizer is not None
        model.train(training)
        loss_sum = 0.0
        sample_count = 0
        absolute_error_sum = 0.0
        observation_count = 0
        series_error_sum: dict[str, float] = defaultdict(float)
        series_window_count: dict[str, int] = defaultdict(int)

        for batch in loader:
            inputs = batch["X"].to(self.device)
            future_calendar = batch["future_calendar"].to(self.device)
            target = batch["y"].to(self.device)
            ids = [str(unique_id) for unique_id in batch["unique_id"]]

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                forecast = model(inputs, future_calendar)
                loss = nn.functional.mse_loss(forecast, target)
                if training:
                    loss.backward()
                    optimizer.step()

            batch_size = int(target.shape[0])
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            forecast_original = inverse_targets(forecast.detach(), ids)
            target_original = inverse_targets(target.detach(), ids)
            errors = np.abs(forecast_original - target_original)
            absolute_error_sum += float(errors.sum())
            observation_count += int(errors.size)
            for unique_id, sample_error in zip(
                ids, errors.mean(axis=1), strict=True
            ):
                series_error_sum[unique_id] += float(sample_error)
                series_window_count[unique_id] += 1

        if sample_count == 0 or observation_count == 0:
            raise ValueError("An epoch cannot run with an empty data loader")
        mase_values = [
            (series_error_sum[unique_id] / series_window_count[unique_id])
            / mase_scales[unique_id]
            for unique_id in series_error_sum
            if mase_scales[unique_id] > 0
        ]
        return {
            "loss": loss_sum / sample_count,
            "mae": absolute_error_sum / observation_count,
            "mase": float(np.mean(mase_values)) if mase_values else None,
        }
