from abc import ABC, abstractmethod
from random import randint

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Sequential
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import _Loss as Loss

from gaarc.data.arc_data_models import ARC_ENTITY_UNIQUE_COLORS, ARCSample


class STM(ABC, nn.Module):
    _name: str
    _loss_function: Loss

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def get_input_features(self, sample: ARCSample, idx: int):
        pass

    @abstractmethod
    def get_target(self, sample: ARCSample, idx: int):
        pass

    @abstractmethod
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def train_on_task(self, samples: torch.Tensor) -> Loss:
        pass


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

        self._flatten = nn.Flatten()

        if self._use_batch_norm:
            self.batch_norm = BatchNorm1d(latent_space_size)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_input_features(self, sample: ARCSample, idx: int) -> torch.Tensor:
        entity = sample.entities[idx]

        input_features = torch.tensor(
            entity.entity, device=self._device, dtype=torch.float
        ).unsqueeze(0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> torch.Tensor:
        entity = sample.entities[idx]

        target = torch.tensor(
            entity.center_of_mass, device=self._device, dtype=torch.float
        )

        return target

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        features = self._encoder(input_features)[0]

        features = self._flatten(features)

        if self._use_batch_norm:
            features = self.batch_norm(features)

        x_features = self._x_classifier(features)
        y_features = self._y_classifier(features)

        features = torch.cat((x_features, y_features), dim=1)

        return features

    def train_on_task(self, samples: torch.Tensor) -> Loss:
        input_features_retrieved = []
        targets_retrieved = []

        for sample in samples:
            sample = ARCSample(sample.detach().cpu().squeeze(0).numpy())

            if sample.entities:
                idx = randint(0, len(sample.entities) - 1)

                input_features_retrieved.append(self.get_input_features(sample, idx))
                targets_retrieved.append(self.get_target(sample, idx))

        input_features = torch.tensor(
            np.array(input_features_retrieved), dtype=torch.float, device=self._device
        )
        targets = torch.tensor(
            np.array(targets_retrieved), dtype=torch.float, device=self._device
        )

        predictions = self.forward(input_features)

        loss = self._loss_function(predictions, targets)

        return loss


class SuperEntityColors(STM):
    _name: str = "Super Entity's Colors"
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

        self._classifier = Sequential(
            Linear(latent_space_size, hidden_layer_size),
            Linear(hidden_layer_size, 1),
        )

        self._flatten = nn.Flatten()

        if self._use_batch_norm:
            self.batch_norm = BatchNorm1d(latent_space_size)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_input_features(self, sample: ARCSample, idx: int) -> torch.Tensor:
        super_entity = sample.super_entities[idx]

        input_ = torch.tensor(
            super_entity.entity, device=self._device, dtype=torch.float
        ).unsqueeze(0)

        return input_

    def get_target(self, sample: ARCSample, idx: int) -> torch.Tensor:
        super_entity = sample.super_entities[idx]

        target = torch.tensor(
            [len(super_entity.colors) / ARC_ENTITY_UNIQUE_COLORS],
            device=self._device,
            dtype=torch.float,
        )

        return target

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        features = self._encoder(input_features)[0]

        features = self._flatten(features)

        if self._use_batch_norm:
            features = self.batch_norm(features)

        features = self._classifier(features)

        return features

    def train_on_task(self, samples: torch.Tensor) -> Loss:
        input_features_retrieved = []
        targets_retrieved = []

        for sample in samples:
            sample = ARCSample(sample.detach().cpu().squeeze(0).numpy())

            if sample.super_entities:
                idx = randint(0, len(sample.super_entities) - 1)

                input_features_retrieved.append(self.get_input_features(sample, idx))
                targets_retrieved.append(self.get_target(sample, idx))

        input_features = torch.tensor(
            np.array(input_features_retrieved), dtype=torch.float, device=self._device
        )
        targets = torch.tensor(
            np.array(targets_retrieved), dtype=torch.float, device=self._device
        )

        predictions = self.forward(input_features)

        loss = self._loss_function(predictions, targets)

        return loss
