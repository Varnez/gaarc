import abc
import random
from abc import ABC
from typing import Callable

import numpy as np

from gaarc.data.transformations import flip_image, rotate_image


class DataAugmentationTransformation(ABC):
    def __init__(self, chance_of_execution: float = 1.0):
        self._chance_of_execution = chance_of_execution
        self._transformation: Callable

    @abc.abstractmethod
    def transform(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        if random.random() <= self._chance_of_execution:
            image = self._transformation(image, *args, **kwargs)

        return image

    @abc.abstractmethod
    def transform_in_bulk(
        self, images: list[np.ndarray], *args, **kwargs
    ) -> list[np.ndarray]:
        transformed_images = []
        if random.random() <= self._chance_of_execution:
            for image in images:
                image = self._transformation(image, *args, **kwargs)

                transformed_images.append(image)

        return transformed_images


class FlipTransformation(DataAugmentationTransformation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._transformation = flip_image


class RotateTransformation(DataAugmentationTransformation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._transformation = rotate_image
