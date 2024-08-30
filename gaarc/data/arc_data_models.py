import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from gaarc.visualization.arc_visualization import plot_arc_sample

BORDER_TYPES = ("side", "corner", "point", "isolated")
ARC_ENTITY_UNIQUE_COLORS = 9

# pylint: disable=singleton-comparison


sys.setrecursionlimit(2000)


class BaseEntity(ABC):
    """
    Common parent class to the different kinds of entities.

    It implements most functionality, excluding those which depends on how the inherent class is
    holding certain specific ambits of its information.

    Parameters
    ----------
    sample : np.ndarray
        Original arc sample in which the entity is taking place.
    entity_mask : np.ndarray
        Boolean two dimensional tensor which shares the same shape as the sample.
        It marks the positions from the sample tensor in which the entity takes place with True,
        False elsewhere.
        The generation of this mask is done in the according ARCSample methods to generate entities.
    """

    def __init__(self, sample: np.ndarray, entity_mask: np.ndarray):
        self._entity_mask: np.ndarray = entity_mask
        self._entity: np.ndarray = sample * entity_mask
        self._border_pixels: dict[str, list[tuple[int, int]]] = {
            border_type: [] for border_type in BORDER_TYPES
        }
        self._center_of_mass: tuple[float, float] | None = None
        self._is_square: bool | None = None
        self._is_rectangular: bool | None = None

        self._detect_borders()

    @property
    def entity(self) -> np.ndarray:
        return self._entity

    @property
    def entity_mask(self) -> np.ndarray:
        return self._entity_mask

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    def center_of_mass(self) -> tuple[float, float]:
        if self._center_of_mass is None:
            x_center = 0.0
            y_center = 0.0

            for x in range(self._entity_mask.shape[0] - 1):
                for y in range(self._entity_mask.shape[1] - 1):
                    if self._entity_mask[x][y] == True:
                        x_center += x_center
                        y_center += y_center

            x_center /= self.size
            y_center /= self.size

            self._center_of_mass = (x_center, y_center)

        return self._center_of_mass

    @property
    def is_square(self) -> bool:
        if self._is_square is None:
            if (
                self.size == 1
                or self.is_rectangular
                and (
                    (
                        self._border_pixels["corner"][0][0]
                        - self._border_pixels["corner"][-1][0]
                        == self._border_pixels["corner"][0][1]
                        - self._border_pixels["corner"][-1][1]
                    )
                )
            ):
                self._is_square = True
            else:
                self._is_square = False

        return self._is_square

    @property
    def is_rectangular(self) -> bool:
        if self._is_rectangular is None:
            if (
                len(self._border_pixels["corner"]) == 4
                and len(self._border_pixels["point"]) == 0
            ):
                self._is_rectangular = True
            else:
                self._is_rectangular = False

        return self._is_rectangular

    def _detect_borders(self) -> None:
        for x in range(self._entity_mask.shape[0]):
            for y in range(self._entity_mask.shape[1]):
                if self._entity_mask[x][y] == True:
                    sides_to_the_outside = 0

                    if x == 0 or (x != 0 and self._entity_mask[x - 1][y] != True):
                        sides_to_the_outside += 1

                    if y == 0 or (y != 0 and self._entity_mask[x][y - 1] != True):
                        sides_to_the_outside += 1

                    if x == self._entity_mask.shape[0] - 1 or (
                        x < self._entity_mask.shape[0] - 1
                        and self._entity_mask[x + 1][y] != True
                    ):
                        sides_to_the_outside += 1

                    if y == self._entity_mask.shape[1] - 1 or (
                        y < self._entity_mask.shape[1] - 1
                        and self._entity_mask[x][y + 1] != True
                    ):
                        sides_to_the_outside += 1

                    if sides_to_the_outside == 1:
                        self._border_pixels["side"].append((x, y))
                    elif sides_to_the_outside == 2:
                        self._border_pixels["corner"].append((x, y))
                    elif sides_to_the_outside == 3:
                        self._border_pixels["point"].append((x, y))
                    elif sides_to_the_outside == 4:
                        self._border_pixels["isolated"].append((x, y))

    def plot(self) -> None:
        """
        Plots the entity (against the background of the original image) with the id mapped to their
        corresponding colors as per the ARC canon.
        """
        plot_arc_sample(self.entity)


class Entity(BaseEntity):
    """
    Entity that is composed by a single color, the basic conception of an entity.

    Parameters
    ----------
    sample : np.ndarray
        Original arc sample in which the entity is taking place.
    entity_mask : np.ndarray
        Boolean two dimensional tensor which shares the same shape as the sample.
        It marks the positions from the sample tensor in which the entity takes place with True,
        False elsewhere.
        The generation of this mask is done in the according ARCSample methods to generate entities.
    """

    def __init__(self, sample: np.ndarray, entity_mask: np.ndarray):
        super().__init__(sample, entity_mask)
        self._size: int = np.sum(entity_mask)

        colors = np.unique(self._entity)
        assert len(colors) == 2, (
            f"Multiple colors detected on entity {colors[1:]} (this should be "
            "a super entity)"
        )
        self._color: int = colors[1]

    @property
    def color(self):
        return self._color

    @property
    def size(self) -> int:
        return self._size


class SuperEntity(BaseEntity):
    """
    An extension on the conception of the base Entities, formed by more than one color.

    Thus, this class is able to combine multiple colors taking place subjacent to one another,
    not being isolated by background (positions with value 0).

    This sometimes, for samples which, for example, have a grid covering the whole image, can lead
    to produce a super entity that is equal to the initial sample. So, tread carefully.

    Parameters
    ----------
    sample : np.ndarray
        Original arc sample in which the entity is taking place.
    entity_mask : np.ndarray
        Boolean two dimensional tensor which shares the same shape as the sample.
        It marks the positions from the sample tensor in which the entity takes place with True,
        False elsewhere.
        The generation of this mask is done in the according ARCSample methods to generate entities.
    """

    def __init__(self, sample: np.ndarray, entity_mask: np.ndarray):
        super().__init__(sample, entity_mask)
        self._values: dict[int, int] = dict()

        values, counts = np.unique(self._entity, return_counts=True)
        for value, count in zip(values, counts):
            if value != 0:
                self._values[value] = count

        self._size = 0
        for amount in self._values.values():
            self._size += amount

    @property
    def colors(self) -> list[int]:
        colors = list(self._values.keys())

        return colors

    @property
    def size(self) -> int:
        return self._size


class ARCSample:
    """
    Class to expand our understanding of individual samples within the ARC Dataset.

    It allows for the detection and generation of entities instances from which multiple properties
    can be extracted.

    Also it provides some tool methods such as plotting automatically applying the color mapping
    that changes each id for its canonical color within the ARC.

    Parameters
    ----------
    sample : np.ndarray
        ARC sample read as a numpy array.
    visualize_entity_detection: bool
        This activates the debugging functionality that plots each step of the entity flooding,
        to check how it is taking place. Cool stuff
    """

    def __init__(self, sample: np.ndarray, visualize_entity_detection: bool = False):
        self._sample: np.ndarray = sample
        self._visualize_entity_detection: bool = visualize_entity_detection
        self._values: dict = {}
        self._entities: list[Entity] | None = None
        self._super_entities: list[SuperEntity] | None = None

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

    @property
    def entities(self) -> list[Entity]:  #
        if self._entities is None:
            self.detect_entities()

        return self._entities  # type: ignore[return-value]

    @property
    def super_entities(self) -> list[SuperEntity]:
        if self._super_entities is None:
            self.detect_super_entities()

        return self._super_entities  # type: ignore[return-value]

    def detect_entities(self) -> int:
        detection_mask = np.zeros(self._sample.shape).astype(bool)
        entities = []

        for i in range(self._sample.shape[0]):
            for j in range(self._sample.shape[1]):
                if self._sample[i][j] > 0 and detection_mask[i][j] != True:
                    entity_color = self._sample[i][j]
                    entity_mask = self._flood_entity(i, j, entity_color=entity_color)

                    detection_mask = np.logical_or(detection_mask, entity_mask)

                    entity = Entity(self._sample, entity_mask)
                    entities.append(entity)

        self._entities = entities

        return len(entities)

    def detect_super_entities(self) -> int:
        detection_mask = np.zeros(self._sample.shape).astype(bool)
        entities = []

        for i in range(self._sample.shape[0]):
            for j in range(self._sample.shape[1]):
                if self._sample[i][j] > 0 and detection_mask[i][j] != True:
                    entity_mask = self._flood_entity(i, j)

                    detection_mask = np.logical_or(detection_mask, entity_mask)

                    entity = SuperEntity(self._sample, entity_mask)
                    entities.append(entity)

        self._super_entities = entities

        return len(entities)

    def __getitem__(self, idx) -> np.ndarray:
        return self._sample[idx]

    def _flood_entity(
        self,
        x: int,
        y: int,
        entity_mask: np.ndarray | None = None,
        entity_color: int | None = None,
    ) -> np.ndarray:
        if entity_mask is None:
            entity_mask = np.zeros(self._sample.shape).astype(bool)

        if self._sample[x][y] != 0:
            entity_mask[x][y] = True

        if self._visualize_entity_detection:
            with plt.style.context("grayscale"):
                plt.imshow(entity_mask)
                plt.show()

        if (
            x != 0
            and self._sample[x - 1][y] != 0
            and entity_mask[x - 1][y] != True
            and (entity_color is None or self._sample[x - 1][y] == entity_color)
        ):
            returned_entity_mask = self._flood_entity(
                x - 1, y, entity_mask, entity_color
            )

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if (
            y != 0
            and self._sample[x][y - 1] != 0
            and entity_mask[x][y - 1] != True
            and (entity_color is None or self._sample[x][y - 1] == entity_color)
        ):
            returned_entity_mask = self._flood_entity(
                x, y - 1, entity_mask, entity_color
            )

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if (
            x < entity_mask.shape[0] - 1
            and self._sample[x + 1][y] != 0
            and entity_mask[x + 1][y] != True
            and (entity_color is None or self._sample[x + 1][y] == entity_color)
        ):
            returned_entity_mask = self._flood_entity(
                x + 1, y, entity_mask, entity_color
            )

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        if (
            y < entity_mask.shape[1] - 1
            and self._sample[x][y + 1] != 0
            and entity_mask[x][y + 1] != True
            and (entity_color is None or self._sample[x][y + 1] == entity_color)
        ):
            returned_entity_mask = self._flood_entity(
                x, y + 1, entity_mask, entity_color
            )

            entity_mask = np.logical_or(entity_mask, returned_entity_mask)

        return entity_mask

    def plot(self) -> None:
        """
        Plots the image with the id mapped to their corresponding colors as per the ARC canon.
        """
        plot_arc_sample(self.sample)
