# src/models/temporal_lstm.py

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):

    def __init__(
        self,
        input_size=1280,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        direction_multiplier = 2 if bidirectional else 1

        self.output_dim = hidden_size * direction_multiplier

    def forward(self, x):

        """
        x shape:
        (B, T, F)

        Example:
        (1, 180, 1280)
        """

        outputs, (hidden, cell) = self.lstm(x)

        # Last timestep output
        temporal_features = outputs[:, -1, :]

        return temporal_features