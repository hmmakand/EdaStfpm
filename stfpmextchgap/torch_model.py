
from __future__ import annotations

from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.stfpmextchgap.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler
from anomalib.models.stfpmextchgap.custom_model import EXTCHGAPATTENTION


class STFPMEXTCHGAPModel(nn.Module):

    def __init__(
        self,
        layers: list[str],
        input_size: tuple[int, int],
        backbone: str = "resnet18",
        # backbone: str = "fcn_resnet50",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = EXTCHGAPATTENTION(backbone=self.backbone, pre_trained=True, layers=layers, input_size=input_size)
        self.student_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=False, layers=layers, requires_grad=True
        )

        
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size)

    def forward(self, images: Tensor) -> Tensor | dict[str, Tensor] | tuple[dict[str, Tensor]]:
        
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features_combined = self.teacher_model(images)
        teacher_features: dict[str, Tensor] = teacher_features_combined[0]
        normattn_teacher_features: dict[str, Tensor] = teacher_features_combined[1]
        student_features: dict[str, Tensor] = self.student_model(images)
        if self.training:
            output = teacher_features, normattn_teacher_features, student_features
        else:
            output = self.anomaly_map_generator(attn_teacher_features=normattn_teacher_features, student_features=student_features)
            if self.tiler:
                output = self.tiler.untile(output)

        return output
