"""1D CNN classifying a ride from its raw IMU readings."""
import numpy as np
import torch
from torch import nn

from rides.data import CLASSES, SENSORS


class RidesClassifier(nn.Module):
    """Raw readings (batch, T, C) -> class probabilities (batch, 5).

    Normalisation lives inside the model, so the API sends raw sensor values.
    Global pooling lets it read a 10 s window or a whole ride (T >= 10).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.features = nn.Sequential(
            self._block(len(SENSORS), channels),
            self._block(channels, channels),
            self._block(channels, channels * 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels * 2, len(CLASSES)))

    @staticmethod
    def _block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def logits(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Training entry point: already-normalised windows (batch, T, C)."""
        return self.head(self.features(x_norm.transpose(1, 2)))  # Conv1d wants (batch, C, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Serving entry point: raw readings in, probabilities out (the API contract)."""
        return torch.softmax(self.logits((x - self.mean) / self.std), dim=-1)