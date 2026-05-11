# src/models/spatial_cnn.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np


class SpatialCNN(nn.Module):

    def __init__(self):

        super().__init__()

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # Remove classifier
        self.feature_extractor = model.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, x):

        """
        x:
        (B, 3, 224, 224)
        """

        x = self.feature_extractor(x)

        x = self.pool(x)

        x = x.flatten(1)

        return x

    # --------------------------------
    # EXTRACT FEATURES FROM NUMPY FRAMES
    # --------------------------------
    def extract_features(self, frames):

        """
        Input:
            frames -> [B, T, C, H, W]

        Output:
            features -> [B, T, 1280]
        """

        B, T, C, H, W = frames.shape

        # Flatten temporal dimension
        frames = frames.view(B * T, C, H, W)

        # CNN forward
        features = self.forward(frames)

        # Restore temporal dimension
        features = features.view(B, T, -1)

        return features