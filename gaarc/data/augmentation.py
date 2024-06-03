import abc
import random
from abc import ABC

import numpy as np

from gaarc.data.transformations import flip_image, rotate_image


class DataAugmentationTransformation(ABC):
    def __init__(self, chance_of_execution: float = 1.0):
        self._chance_of_execution = chance_of_execution

    @abc.abstractmethod
    def transform(self, image: np.ndarray, **kwargs) -> np.ndarray:
        pass


class FlipTransformation(DataAugmentationTransformation):
    def transform(  # pylint: disable=arguments-differ
        self, image: np.ndarray, axis: str | None = None
    ) -> np.ndarray:
        if random.random() <= self._chance_of_execution:
            image = flip_image(image, axis)

        return image


class RotateTransformation(DataAugmentationTransformation):
    def transform(  # pylint: disable=arguments-differ
        self, image: np.ndarray, angle: str | None = None
    ) -> np.ndarray:
        if random.random() <= self._chance_of_execution:
            image = rotate_image(image, angle)

        return image
