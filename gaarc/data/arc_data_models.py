import matplotlib.pyplot as plt
import numpy as np


class ARCSample:
    def __init__(self, sample: np.ndarray, visualize_entity_detection: bool = False):
        self._sample: np.ndarray = sample
        self._visualize_entity_detection: bool = visualize_entity_detection
        self._values: dict = {}
        self._entities: list[Entity] | None = None

        values, counts = np.unique(sample, return_counts=True)
        for value, count in zip(values, counts):
            self._values[value] = count

    @property
    def has_background(self) -> bool:
        has_background: bool = 0 in self._values

        return has_background

    @property
    def sample(self) -> np.ndarray:
        return self._sample

    @property
    def shape(self) -> tuple[int, ...]:
        return self._sample.shape

    def detect_entities(self) -> int:
        detection_mask = np.zeros(self._sample.shape).astype(bool)
        entities = []

        for i in range(self._sample.shape[0]):
            for j in range(self._sample.shape[1]):
                if self._sample[i][j] > 0 and detection_mask[i][j] != True:
                    entity_mask = self._flood_entity(i, j)

                    detection_mask = np.logical_or(detection_mask, entity_mask)

                    entity = Entity(self._sample, entity_mask)
                    entities.append(entity)

        self._entities = entities

        return len(entities)

    def __getitem__(self, idx) -> np.ndarray:
        return self._sample[idx]

    def _flood_entity(
        self, x: int, y: int, entity_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if entity_mask is None:
            entity_mask = np.zeros(self._sample.shape).astype(bool)

        if self._sample[x][y] != 0:
            entity_mask[x][y] = True

        if self._visualize_entity_detection:
            with plt.style.context("grayscale"):
                plt.imshow(entity_mask)
                plt.show()

        if x != 0 and self._sample[x - 1][y] != 0 and entity_mask[x - 1][y] != True:
            returned_entity_mask = self._flood_entity(x - 1, y, entity_mask)

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if y != 0 and self._sample[x][y - 1] and entity_mask[x][y - 1] != True:
            returned_entity_mask = self._flood_entity(x, y - 1, entity_mask)

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if (
            x < entity_mask.shape[0] - 1
            and self._sample[x + 1][y]
            and entity_mask[x + 1][y] != True
        ):
            returned_entity_mask = self._flood_entity(x + 1, y, entity_mask)

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if (
            y < entity_mask.shape[1] - 1
            and self._sample[x][y + 1]
            and entity_mask[x][y + 1] != True
        ):
            returned_entity_mask = self._flood_entity(x, y + 1, entity_mask)

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        return entity_mask


class Entity:
    def __init__(self, sample: np.ndarray, entity_mask: np.ndarray):
        self._entity_mask: np.ndarray = entity_mask
        self._entity: np.ndarray = sample * entity_mask
        self._values: dict[int, int] = dict()

        values, counts = np.unique(self._entity, return_counts=True)
        for value, count in zip(values, counts):
            if value != 0:
                self._values[value] = count

    @property
    def entity(self) -> np.ndarray:
        return self._entity

    @property
    def entity_mask(self) -> np.ndarray:
        return self._entity_mask

    @property
    def colors(self) -> list[int]:
        colors = list(self._values.keys())

        return colors

    @property
    def size(self) -> int:
        size = 0

        for amount in self._values.values():
            size += amount

        return size
