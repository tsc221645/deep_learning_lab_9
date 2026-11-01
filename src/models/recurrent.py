"""Three day-ahead LSTM strategies using the shared PyTorch adapter.

The decoder sees ground truth only inside ``forward_train``. Validation and
backtesting always call ``forward`` and feed predictions back recursively.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.torch_forecasters import TorchForecastAdapter


def _lstm(input_size: int, hidden_size: int, layers: int, dropout: float) -> nn.LSTM:
    if hidden_size < 1 or layers < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid LSTM hidden_size, num_layers, or dropout")
    return nn.LSTM(
        input_size,
        hidden_size,
        num_layers=layers,
        dropout=dropout if layers > 1 else 0.0,
        batch_first=True,
    )


class LSTMMultiOutput(nn.Module):
    """Encode observed history once and project its final state to H outputs."""

    def __init__(
        self, lookback: int, horizon: int, calendar_size: int,
        hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = _lstm(calendar_size + 1, hidden_size, num_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + horizon * calendar_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, horizon),
        )

    def forward(self, history: torch.Tensor, future_calendar: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(history)
        features = torch.cat((hidden[-1], future_calendar.flatten(1)), dim=1)
        return self.head(features)


class LSTMRecursive(nn.Module):
    """One-step head with feedback of each predicted value; no future targets."""

    def __init__(
        self, lookback: int, horizon: int, calendar_size: int,
        hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
        teacher_forcing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= teacher_forcing <= 1:
            raise ValueError("teacher_forcing must be between zero and one")
        self.encoder = _lstm(1, hidden_size, num_layers, dropout)
        self.decoder = nn.LSTMCell(1, hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        self.teacher_forcing = teacher_forcing

    def _decode(self, history: torch.Tensor, steps: int,
                target: torch.Tensor | None = None) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(history[..., :1])
        state, memory = hidden[-1], cell[-1]
        previous = history[:, -1, :1]
        outputs = []
        for step in range(steps):
            state, memory = self.decoder(previous, (state, memory))
            prediction = self.head(state)
            outputs.append(prediction)
            previous = prediction
            if target is not None and step + 1 < steps and self.teacher_forcing:
                mask = torch.rand_like(previous) < self.teacher_forcing
                previous = torch.where(mask, target[:, step:step + 1], previous)
        return torch.cat(outputs, dim=1)

    def forward(self, history: torch.Tensor, future_calendar: torch.Tensor) -> torch.Tensor:
        return self._decode(history, future_calendar.shape[1])

    def forward_train(self, history: torch.Tensor, future_calendar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        return self._decode(history, future_calendar.shape[1], target)


class LSTMSeq2Seq(nn.Module):
    """Encoder-decoder with known future calendar and optional training forcing."""

    def __init__(
        self, lookback: int, horizon: int, calendar_size: int,
        hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
        teacher_forcing: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0 <= teacher_forcing <= 1:
            raise ValueError("teacher_forcing must be between zero and one")
        self.encoder = _lstm(calendar_size + 1, hidden_size, num_layers, dropout)
        self.decoder = _lstm(calendar_size + 1, hidden_size, num_layers, dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.teacher_forcing = teacher_forcing

    def _decode(self, history: torch.Tensor, future_calendar: torch.Tensor,
                target: torch.Tensor | None = None) -> torch.Tensor:
        _, state = self.encoder(history)
        previous = history[:, -1, :1]
        outputs = []
        for step in range(future_calendar.shape[1]):
            decoder_input = torch.cat(
                (previous, future_calendar[:, step, :]), dim=1
            ).unsqueeze(1)
            decoder_output, state = self.decoder(decoder_input, state)
            prediction = self.head(decoder_output[:, 0, :])
            outputs.append(prediction)
            previous = prediction
            if target is not None and step + 1 < future_calendar.shape[1] and self.teacher_forcing:
                mask = torch.rand_like(previous) < self.teacher_forcing
                previous = torch.where(mask, target[:, step:step + 1], previous)
        return torch.cat(outputs, dim=1)

    def forward(self, history: torch.Tensor, future_calendar: torch.Tensor) -> torch.Tensor:
        return self._decode(history, future_calendar)

    def forward_train(self, history: torch.Tensor, future_calendar: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        return self._decode(history, future_calendar, target)


class _RecurrentAdapter(TorchForecastAdapter):
    model_class: type[nn.Module]
    uses_calendar = True

    def build_module(self, lookback: int, horizon: int, calendar_features_count: int,
                     config: Mapping[str, Any]) -> nn.Module:
        return self.model_class(
            lookback, horizon, calendar_features_count,
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            **({"teacher_forcing": float(config.get("teacher_forcing", 0.5))}
               if self.model_class is not LSTMMultiOutput else {}),
        )


class LSTMRecursiveAdapter(_RecurrentAdapter):
    model_class = LSTMRecursive

    def __init__(self) -> None:
        super().__init__("LSTM-Recursive")


class LSTMMultiOutputAdapter(_RecurrentAdapter):
    model_class = LSTMMultiOutput

    def __init__(self) -> None:
        super().__init__("LSTM-MultiOutput")


class LSTMSeq2SeqAdapter(_RecurrentAdapter):
    model_class = LSTMSeq2Seq

    def __init__(self) -> None:
        super().__init__("LSTM-Seq2Seq")
