# src/models/fusion_model.py

import torch
import torch.nn as nn

from models.spatial_cnn import SpatialCNN
from models.temporal_lstm import TemporalLSTM
from models.rppg_cnn import RPPGCNN
from models.signal_encoder import SignalEncoder


class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # -----------------------------
        # MODELS
        # -----------------------------

        self.spatial_model = SpatialCNN()

        self.temporal_model = TemporalLSTM()

        self.rppg_model = RPPGCNN()

        self.blink_encoder = SignalEncoder(
            input_dim=180
        )

        self.motion_encoder = SignalEncoder(
            input_dim=180
        )

        # -----------------------------
        # FINAL CLASSIFIER
        # -----------------------------

        total_features = (
            512 +   # temporal
            64 +    # rPPG
            32 +    # blink
            32      # motion
        )

        self.classifier = nn.Sequential(

            nn.Linear(total_features, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(
        self,
        frames,
        rppg,
        blink,
        motion
    ):

        # ---------------------------------
        # SPATIAL FEATURES
        # ---------------------------------

        spatial_features = self.spatial_model.extract_features(
            frames
        )

        # spatial:
        # (B, T, 1280)

        # ---------------------------------
        # TEMPORAL FEATURES
        # ---------------------------------

        temporal_features = self.temporal_model(
            spatial_features
        )

        # (B, 512)

        # ---------------------------------
        # rPPG FEATURES
        # ---------------------------------

        rppg_features = self.rppg_model(
            rppg
        )

        # (B, 64)

        # ---------------------------------
        # BLINK FEATURES
        # ---------------------------------

        blink_features = self.blink_encoder(
            blink
        )

        # (B, 32)

        # ---------------------------------
        # MOTION FEATURES
        # ---------------------------------

        motion_features = self.motion_encoder(
            motion
        )

        # (B, 32)

        # ---------------------------------
        # FUSION
        # ---------------------------------

        fused = torch.cat([

            temporal_features,
            rppg_features,
            blink_features,
            motion_features

        ], dim=1)

        # ---------------------------------
        # CLASSIFICATION
        # ---------------------------------

        output = self.classifier(
            fused
        )
        print("Spatial:", spatial_features.shape)
        print("Temporal:", temporal_features.shape)
        print("rPPG:", rppg.shape)
        print("Blink:", blink.shape)
        print("Motion:", motion.shape)
        return output