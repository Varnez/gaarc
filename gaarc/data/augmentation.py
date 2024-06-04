import abc
import random
from abc import ABC
from typing import Any, Callable

import numpy as np

from gaarc.data.transformations import flip_image, rotate_image


class DataAugmentationTransformation(ABC):
    _transformation: Callable

    def __init__(self, chance_of_execution: float = 1.0):
        self._chance_of_execution = chance_of_execution

    def transform(self, image: np.ndarray) -> np.ndarray:
        if random.random() <= self._chance_of_execution:
            image = self._transformation(image)

        return image

    def transform_in_bulk(self, images: list[np.ndarray]) -> list[np.ndarray]:
        transformed_images = []
        if random.random() <= self._chance_of_execution:
            parameter: Any | None = None

            for image in images:
                image, parameter = self._transformation(image, parameter)

                transformed_images.append(image)

        return transformed_images


class FlipTransformation(DataAugmentationTransformation):
    _transformation = flip_image


class RotateTransformation(DataAugmentationTransformation):
    _transformation = rotate_image
