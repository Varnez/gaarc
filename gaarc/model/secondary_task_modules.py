from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import _Loss as Loss

from gaarc.data.arc_data_models import ARCSample


class STM(ABC, nn.Module):
    _name: str
    _loss_function: Loss

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def get_target(self, sample: ARCSample) -> Any:
        pass

    @abstractmethod
    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate_task(self, sample: ARCSample, feature_vector: torch.Tensor) -> Loss:
        target = self.get_target(sample)

        prediction = self.forward(feature_vector)

        loss = self._loss_function(prediction, target)

        return loss


class EntityMassCentre(STM):
    _name: str = "Entity's Mass Center"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        encoder: nn.Module,
        latent_space_size: int,
        hidden_layer_size: int,
        use_batch_norm: bool,
    ):
        self._encoder = encoder
        self._use_batch_norm = use_batch_norm

        self._x_classifier = Linear(hidden_layer_size, 1)
        self._y_classifier = Linear(hidden_layer_size, 1)

        if self._use_batch_norm:
            self.batch_norm = BatchNorm1d(latent_space_size)

        self.to(encoder.device)

    def get_input(self, sample: ARCSample) -> list[torch.Tensor]:
        inputs: list[torch.Tensor[int]] = [
            torch.Tensor(entity.entity, device=self.device)
            for entity in sample.entities
        ]

        return inputs

    def get_target(self, sample: ARCSample) -> list[torch.Tensor]:
        targets: list[torch.Tensor[float]] = [
            torch.Tensor(entity.center_of_mass, device=self.device)
            for entity in sample.entities
        ]

        return targets

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(feature_vector)

        if self._use_batch_norm:
            features = self.batch_norm(features)

        x_features = self._x_classifier(features)
        y_features = self._y_classifier(features)

        features = torch.cat((x_features, y_features))
        return features

    def train_on_task(self, sample: ARCSample) -> Loss:
        inputs: list[torch.Tensor[int]] = self.get_input(sample)
        targets: list[torch.Tensor[float]] = self.get_target(sample)

        losses: list[Loss] = []
        for input_, target in zip(inputs, targets):
            features = self._encoder(input_)
            prediction = self.forward(features)

            loss = self._loss_function(prediction, target)
            losses.append(loss)

        loss = sum(losses)

        return loss
