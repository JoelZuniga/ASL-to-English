from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    COORDINATE_COUNT,
    LANDMARK_COUNT,
    LSTM_DROPOUT,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_CLASSES,
    LSTM_NUM_LAYERS,
)

FEATURE_SIZE = LANDMARK_COUNT * COORDINATE_COUNT


class MotionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = LSTM_NUM_CLASSES,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        lstm_out_size = hidden_size * directions

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits: torch.Tensor = self.classifier(last)
        return logits
