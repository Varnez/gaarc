from abc import ABC, abstractmethod
from random import randint

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Sequential
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
    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        pass

    def evaluate_task(self, sample: ARCSample, feature_vector: torch.Tensor) -> Loss:
        target = self.get_target(sample)

        prediction = self.forward(feature_vector)

        loss = self._loss_function(prediction, target)

        return loss


class EntityMassCentre(STM):
    """
    Secondary task that will aim to predict the center of mass of a random entity within
    the provided sample.

    The initial intention was to go through all entities within the image, but that lead
    to memory issues, given the big amount of entities a single sample can possibly have.
    """

    _name: str = "Entity's Mass Center"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        encoder: nn.Module,
        latent_space_size: int,
        hidden_layer_size: int,
        use_batch_norm: bool,
    ):
        super().__init__()
        self._encoder = encoder
        self._use_batch_norm = use_batch_norm

        self._x_classifier = Sequential(
            Linear(latent_space_size, hidden_layer_size),
            Linear(hidden_layer_size, 1),
        )
        self._y_classifier = Sequential(
            Linear(latent_space_size, hidden_layer_size),
            Linear(hidden_layer_size, 1),
        )

        if self._use_batch_norm:
            self.batch_norm = BatchNorm1d(latent_space_size)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_input(self, sample: ARCSample, idx: int) -> torch.Tensor:
        entity = sample.entities[idx]

        input_ = (
            torch.tensor(entity.entity, device=self._device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return input_

    def get_target(self, sample: ARCSample, idx: int) -> torch.Tensor:
        entity = sample.entities[idx]

        target = torch.tensor(
            entity.center_of_mass, device=self._device, dtype=torch.float
        ).unsqueeze(1)

        return target

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(feature_vector).unsqueeze(0)

        if self._use_batch_norm:
            features = self.batch_norm(features)

        x_features = self._x_classifier(features)
        y_features = self._y_classifier(features)

        features = torch.cat((x_features, y_features))

        return features

    def train_on_task(self, sample: ARCSample) -> Loss:
        if sample.entities:
            idx = randint(0, len(sample.entities) - 1)

            input_ = self.get_input(sample, idx)
            target = self.get_target(sample, idx)

            features = self._encoder(input_)[0]
            prediction = self.forward(features)

            loss = self._loss_function(prediction, target)

        else:
            loss = self._loss_function(
                torch.tensor([0.0], device=self._device, dtype=torch.float),
                torch.tensor([0.0], device=self._device, dtype=torch.float),
            )

        return loss
