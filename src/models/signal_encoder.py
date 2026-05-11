# src/models/signal_encoder.py

import torch
import torch.nn as nn


class SignalEncoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),

            nn.ReLU()
        )

    def forward(self, x):

        return self.encoder(x)