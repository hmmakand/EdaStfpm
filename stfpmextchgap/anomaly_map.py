from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, image_size: ListConfig | tuple) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)

    def compute_layer_map(self, attn_teacher_features: Tensor, student_features: Tensor) -> Tensor:
       
        norm_attn_teacher_features = F.normalize(attn_teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_attn_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
        # layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="nearest")
        return layer_map

    def compute_anomaly_map(
        self, attn_teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]
    ) -> torch.Tensor:
       
        batch_size = list(attn_teacher_features.values())[0].shape[0]
        anomaly_map = torch.ones(batch_size, 1, self.image_size[0], self.image_size[1])
        # for layer in teacher_features.keys():
        #     layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
        #     anomaly_map = anomaly_map.to(layer_map.device)
        #     anomaly_map *= layer_map

        # between teacher norm and student layer
        for attn_teacher_key, student_key in zip(attn_teacher_features.keys(), student_features.keys()):
            layer_map = self.compute_layer_map(attn_teacher_features[attn_teacher_key], student_features[student_key])
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs: dict[str, Tensor]) -> torch.Tensor:
        

        if not ("attn_teacher_features" in kwargs and "student_features" in kwargs):
            raise ValueError(f"Expected keys `attn_teacher_features` and `student_features. Found {kwargs.keys()}")

        attn_teacher_features: dict[str, Tensor] = kwargs["attn_teacher_features"]
        student_features: dict[str, Tensor] = kwargs["student_features"]

        return self.compute_anomaly_map(attn_teacher_features, student_features)
