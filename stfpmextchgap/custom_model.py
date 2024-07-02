from __future__ import annotations

from typing import Callable

import timm
import torch
import numpy as np
import cv2

from torch import Tensor, nn
from anomalib.models.components import FeatureExtractor
from anomalib.models.stfpmextchgap.attentioncaps.customattention import EXTCHGAP

class EXTCHGAPATTENTION(nn.Module):
    def __init__(
        self,
        layers: list[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
        # pre_trained=False,
        pre_trained=True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = layers
        self.requires_grad = requires_grad

        self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=pre_trained, layers=self.layers, requires_grad=self.requires_grad)

        channels = self.feature_extractor.out_dims
        scales = self.feature_extractor.scale

        self.attentions = nn.ModuleList()
        for channel in channels:
            self.attentions.append(
                EXTCHGAP(channel,channel)
            )
        
        self.norms = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [channel, int(input_size[0] / scale),
                     int(input_size[1] / scale)],
                    elementwise_affine=True,
                )
            )
        

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor] | tuple[dict[str, Tensor]]:
        return_val: Tensor | dict[str, Tensor] | Tensor | list[Tensor] | tuple[list[Tensor]] | tuple[dict[str, Tensor]]

        attn_features = self._get_cnn_features(input_tensor)
        return_val = attn_features
        return return_val
    
    
    def _get_cnn_features(self, images: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        features: dict[str, Tensor] = self.feature_extractor(images)

        attentionfeatures = {f"attention_{i+1}": self.attentions[i](feature) 
                         for i, (key, feature) in enumerate(features.items())}
        
        # Apply normalization to each attention feature and store in a dictionary with new keys
        normfeatures = {f"norm_{i+1}": self.norms[i](attentionfeature)
                        for i, (key, attentionfeature) in enumerate(attentionfeatures.items())}

        
        return (features, normfeatures)
    
