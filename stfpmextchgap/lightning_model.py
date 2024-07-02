from __future__ import annotations

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim

from anomalib.models.components import AnomalyModule
from anomalib.models.stfpmextchgap.loss import STFPMEXTCHGAPLoss
from anomalib.models.stfpmextchgap.torch_model import STFPMEXTCHGAPModel

__all__ = ["StfpmextchgapLightning"]


@MODEL_REGISTRY
class Stfpmextchgap(AnomalyModule):

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        layers: list[str],
    ) -> None:
        super().__init__()

        self.model = STFPMEXTCHGAPModel(
            input_size=input_size,
            backbone=backbone,
            layers=layers,
        )
        self.loss = STFPMEXTCHGAPLoss()

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        
        self.model.teacher_model.eval()
        teacher_features, attn_teacher_features, student_features = self.model.forward(batch["image"])
        loss = self.loss(teacher_features, attn_teacher_features, student_features)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        
        batch["anomaly_maps"] = self.model(batch["image"])

        return batch


class StfpmextchgapLightning(Stfpmextchgap):

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_callbacks(self) -> list[EarlyStopping]:
        
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return optim.SGD(
            params=self.model.student_model.parameters(),
            lr=self.hparams.model.lr,
            momentum=self.hparams.model.momentum,
            weight_decay=self.hparams.model.weight_decay,
        )
