# src/models/rppg_cnn.py

import torch
import torch.nn as nn


class RPPGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=5,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):

        # x:
        # (B, T, 5)

        x = x.permute(0, 2, 1)

        # becomes:
        # (B, 5, T)

        x = self.network(x)

        # (B, 64, 1)

        x = x.squeeze(-1)

        # (B, 64)

        return x