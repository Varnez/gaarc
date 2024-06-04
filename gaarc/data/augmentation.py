import abc
import random
from abc import ABC
from typing import Any

import numpy as np

from gaarc.data.transformations import flip_image, rotate_image


class DataAugmentationTransformation(ABC):
    @staticmethod
    @abc.abstractmethod
    def _transformation(
        image: np.ndarray, parameter: Any | None = None
    ) -> tuple[np.ndarray, Any]:
        pass

    def __init__(self, chance_of_execution: float = 1.0):
        self._chance_of_execution = chance_of_execution

    def transform(self, image: np.ndarray) -> np.ndarray:
        if random.random() <= self._chance_of_execution:
            image, _ = self._transformation(image)

        return image

    def transform_in_bulk(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if random.random() <= self._chance_of_execution:
            transformed_images: list[np.ndarray] = []

            parameter: Any | None = None

            for image in images:
                image, parameter = self._transformation(image, parameter)

                transformed_images.append(image)

            images = transformed_images

        return images


class FlipTransformation(DataAugmentationTransformation):
    @staticmethod
    def _transformation(  # pylint: disable=arguments-renamed
        image: np.ndarray, axis: str | None = None
    ) -> tuple[np.ndarray, str]:
        image, axis = flip_image(image, axis)

        return image, axis


class RotateTransformation(DataAugmentationTransformation):
    @staticmethod
    def _transformation(  # pylint: disable=arguments-renamed
        image: np.ndarray, angle: int | None = None
    ) -> tuple[np.ndarray, int]:
        image, angle = rotate_image(image, angle)

        return image, angle
