from abc import ABC, abstractmethod
from random import randint

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Sequential
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import _Loss as Loss

from gaarc.data.arc_data_models import ARC_ENTITY_UNIQUE_COLORS, ARCSample


class STM(nn.Module, ABC):
    _name: str
    _loss_function: Loss

    def __init__(self):
        super().__init__()

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
    def train_on_task(self, samples: list[ARCSample], encoder: nn.Module) -> Loss:
        pass

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = encoder(input_features)[0]

        features = self._flatten(features)

        return features


class EntitySTM(STM, ABC):

    def __init__(
        self,
        task_loss_weight: float = 1.0,
    ):
        super().__init__()
        self._loss_weight: float = task_loss_weight

        self._flatten: nn.Module = nn.Flatten()

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _retrieve_training_elements(
        self, samples: list[ARCSample]
    ) -> tuple[list, list]:
        input_features_retrieved = []
        targets_retrieved = []

        for sample in samples:
            if sample.entities:
                idx = randint(0, len(sample.entities) - 1)

                input_features_retrieved.append(self.get_input_features(sample, idx))
                targets_retrieved.append(self.get_target(sample, idx))

        return input_features_retrieved, targets_retrieved

    def train_on_task(self, samples: list[ARCSample], encoder: nn.Module) -> Loss:
        input_features_retrieved, targets_retrieved = self._retrieve_training_elements(
            samples
        )

        input_features = torch.tensor(
            np.array(input_features_retrieved), dtype=torch.float, device=self._device
        )
        targets = torch.tensor(
            np.array(targets_retrieved), dtype=torch.float, device=self._device
        )

        predictions = self.forward(input_features, encoder)

        loss = self._loss_function(predictions, targets)
        loss = loss * self._loss_weight / len(input_features_retrieved)

        return loss


class SuperEntitySTM(EntitySTM, ABC):

    def _retrieve_training_elements(
        self, samples: list[ARCSample]
    ) -> tuple[list, list]:
        input_features_retrieved = []
        targets_retrieved = []

        for sample in samples:
            if sample.super_entities:
                idx = randint(0, len(sample.super_entities) - 1)

                input_features_retrieved.append(self.get_input_features(sample, idx))
                targets_retrieved.append(self.get_target(sample, idx))

        return input_features_retrieved, targets_retrieved


class EntityMassCentre(EntitySTM):
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
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._x_classifier: nn.Module
        self._y_classifier: nn.Module

        if use_batch_norm:
            self._x_classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )
            self._y_classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._x_classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )
            self._y_classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        entity = sample.entities[idx]

        input_features = np.array(entity.entity)

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        entity = sample.entities[idx]

        target = np.array(entity.center_of_mass)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        x_features = self._x_classifier(features)
        y_features = self._y_classifier(features)

        features = torch.cat((x_features, y_features), dim=1)

        return features


class SuperEntityMassCentre(SuperEntitySTM):
    """
    Secondary task that will aim to predict the center of mass of a random super entity within
    the provided sample.
    """

    _name: str = "Super Entity's Mass Center"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._x_classifier: nn.Module
        self._y_classifier: nn.Module

        if use_batch_norm:
            self._x_classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )
            self._y_classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._x_classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )
            self._y_classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.super_entities[idx]

        input_features = np.array(super_entity.entity)

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.super_entities[idx]

        target = np.array(super_entity.center_of_mass)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        x_features = self._x_classifier(features)
        y_features = self._y_classifier(features)

        features = torch.cat((x_features, y_features), dim=1)

        return features


class EntitySize(EntitySTM):
    """
    Secondary task that will aim to predict the amount of pixels of a random entity within
    the provided sample.
    """

    _name: str = "Entity's Size"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._classifier: nn.Module

        if use_batch_norm:
            self._classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        entity = sample.entities[idx]

        input_features = np.array(
            entity.entity,
        )

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.entities[idx]

        target = np.array(
            super_entity.size,
        )

        target = np.expand_dims(target, axis=0)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        features = self._classifier(features)

        return features


class SuperEntitySize(SuperEntitySTM):
    """
    Secondary task that will aim to predict the amount of pixels of a random super entity within
    the provided sample.
    """

    _name: str = "Super Entity's Size"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._classifier: nn.Module

        if use_batch_norm:
            self._classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entities = sample.super_entities[idx]

        input_features = np.array(
            super_entities.entity,
        )

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.super_entities[idx]

        target = np.array(
            super_entity.size,
        )

        target = np.expand_dims(target, axis=0)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        features = self._classifier(features)

        return features


class EntityColor(EntitySTM):
    _name: str = "Entity's Color"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._classifier: nn.Module

        if use_batch_norm:
            self._classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        entity = sample.entities[idx]

        input_features = np.array(
            entity.entity,
        )

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        entity = sample.entities[idx]

        target = np.array(
            entity.color / ARC_ENTITY_UNIQUE_COLORS,
        )

        target = np.expand_dims(target, axis=0)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        features = self._classifier(features)

        return features


class SuperEntityColors(SuperEntitySTM):
    _name: str = "Super Entity's Colors"
    _loss_function: Loss = MSELoss()  # type: ignore[assignment]

    def __init__(
        self,
        latent_space_size: int,
        hidden_layer_size: int,
        task_loss_weight: float = 1.0,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            task_loss_weight=task_loss_weight,
        )

        self._classifier: nn.Module

        if use_batch_norm:
            self._classifier = Sequential(
                BatchNorm1d(latent_space_size),
                Linear(latent_space_size, hidden_layer_size),
                BatchNorm1d(hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

        else:
            self._classifier = Sequential(
                Linear(latent_space_size, hidden_layer_size),
                Linear(hidden_layer_size, 1),
            )

    def get_input_features(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.super_entities[idx]

        input_features = np.array(
            super_entity.entity,
        )

        input_features = np.expand_dims(input_features, axis=0)

        return input_features

    def get_target(self, sample: ARCSample, idx: int) -> np.ndarray:
        super_entity = sample.super_entities[idx]

        target = np.array(
            len(super_entity.colors) / ARC_ENTITY_UNIQUE_COLORS,
        )

        target = np.expand_dims(target, axis=0)

        return target

    def forward(self, input_features: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        features = super().forward(input_features, encoder)

        features = self._classifier(features)

        return features
