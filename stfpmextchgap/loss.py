
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class STFPMEXTCHGAPLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.huber_loss = nn.SmoothL1Loss()


    def compute_layer_loss(self, teacher_feats: Tensor, attn_teacher_feats: Tensor, student_feats: Tensor) -> Tensor:

        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_attn_teacher_features = F.normalize(attn_teacher_feats)
        norm_student_features = F.normalize(student_feats)
        
        distillation_loss = (0.5 / (width * height)) * self.mse_loss(norm_student_features, norm_attn_teacher_features)        
        
        layer_loss = distillation_loss
        
        return layer_loss


    def forward(self, teacher_features: dict[str, Tensor], attn_teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]) -> Tensor:
        
        layer_losses: list[Tensor] = []
        # calculate loss between teacher norm and student layer
        for teacher_key, attn_teacher_key, student_key in zip(teacher_features.keys(), attn_teacher_features.keys(), student_features.keys()):
            loss = self.compute_layer_loss(teacher_features[teacher_key], attn_teacher_features[attn_teacher_key], student_features[student_key])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss
