"""Loss function for the STFPM Model Implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class STFPMEXTCHGAPLoss(nn.Module):
    """Feature Pyramid Loss This class implmenents the feature pyramid loss function proposed in STFPM paper.

    Example:
        >>> from anomalib.models.components.feature_extractors import FeatureExtractor
        >>> from anomalib.models.stfpm.loss import STFPMLoss
        >>> from torchvision.models import resnet18

        >>> layers = ['layer1', 'layer2', 'layer3']
        >>> teacher_model = FeatureExtractor(model=resnet18(pretrained=True), layers=layers)
        >>> student_model = FeatureExtractor(model=resnet18(pretrained=False), layers=layers)
        >>> loss = Loss()

        >>> inp = torch.rand((4, 3, 256, 256))
        >>> teacher_features = teacher_model(inp)
        >>> student_features = student_model(inp)
        >>> loss(student_features, teacher_features)
            tensor(51.2015, grad_fn=<SumBackward0>)
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.huber_loss = nn.SmoothL1Loss()


    def compute_layer_loss(self, teacher_feats: Tensor, attn_teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        """Compute layer loss based on Equation (1) in Section 3.2 of the paper.

        Args:
          teacher_feats (Tensor): Teacher features
          student_feats (Tensor): Student features

        Returns:
          L2 distance between teacher and student features.
        """

        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_attn_teacher_features = F.normalize(attn_teacher_feats)
        norm_student_features = F.normalize(student_feats)
        
        ### V_1 with pixel_AUPR 0.91038 pixel_AUROC 0.98808 pixel_F1Score 0.83144 training loss 7.8 with distillation loss ###
        # teacher_attention_loss = (0.5 / (width * height)) * self.mse_loss(norm_attn_teacher_features, norm_teacher_features)
        distillation_loss = (0.5 / (width * height)) * self.mse_loss(norm_student_features, norm_attn_teacher_features)        
        
        
        # teacher_attention_loss = (0.5 / (width * height)) * self.huber_loss(norm_attn_teacher_features, norm_teacher_features)
        # distillation_loss = (0.5 / (width * height)) * self.huber_loss(norm_student_features, norm_attn_teacher_features)


        layer_loss = distillation_loss
        ### if layer_loss is the following training loss 32
        # layer_loss = teacher_attention_loss + distillation_loss
        
        return layer_loss


    def forward(self, teacher_features: dict[str, Tensor], attn_teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]) -> Tensor:
        """Compute the overall loss via the weighted average of the layer losses computed by the cosine similarity.

        Args:
          teacher_features (dict[str, Tensor]): Teacher features
          student_features (dict[str, Tensor]): Student features

        Returns:
          Total loss, which is the weighted average of the layer losses.
        """

        layer_losses: list[Tensor] = []
        # for layer in teacher_features.keys():
        #     loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
        #     layer_losses.append(loss)

        # calculate loss between teacher norm and student layer
        for teacher_key, attn_teacher_key, student_key in zip(teacher_features.keys(), attn_teacher_features.keys(), student_features.keys()):
            loss = self.compute_layer_loss(teacher_features[teacher_key], attn_teacher_features[attn_teacher_key], student_features[student_key])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss
