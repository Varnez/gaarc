import random

import cv2
import numpy as np

# For cv2, which causes a lot of false positives
# pylint: disable=no-member


def rotate_image(image: np.ndarray, angles: int | None = None) -> np.ndarray:
    """
    Rotates the provided image the amount of angles indicated. The rotation only supports angles
    are multiples of 90º.

    If no angles are provided, a random multiple of 90º rotation will be applied.

    Parameters
    ----------
    image : np.ndarray
        Image to rotate, formatted as a numpy array.
    angles : int | None, optional
        Amount of angles, multiple of 90º, to rate the image.
        By default None, which will cause a rotation which is a random multiple of 90º.

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    ROTATIONS = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    if angles is None:
        angles = random.choice(list(ROTATIONS.keys()))

    if angles not in ROTATIONS:
        raise ValueError(
            f"Only angles multiple of 90º are supported ({angles} was provided)"
        )

    rotation = ROTATIONS[angles]

    rotated_image = cv2.rotate(image, rotation)

    return rotated_image


def flip_image(image: np.ndarray, axis: str | None = None) -> np.ndarray:
    """
    Flips an image over the selected axis.

    If None is provided, a random axis (or both) will be selected.

    Parameters
    ----------
    image : np.ndarray
        Image to flip, formatted as a numpy array.
    type : str | None, optional
        Axis over which to perform the flip.
        Available values are:
            - "vertical" or "y" to flip vertically.
            - "horizontal" or "x" to flip horizontally.
            - "both" or "xy" to flip both axis.
        Any other value will raise a ValueError.

    Returns
    -------
    np.ndarray
        Flipped image.
    """
    FLIP_AXIS = {"vertical": 0, "y": 0, "horizontal": 1, "x": 1, "both": -1, "xy": -1}

    if axis is None:
        axis = random.choice(list(FLIP_AXIS.keys()))

    if axis not in FLIP_AXIS:
        raise ValueError(
            (
                f"'{axis}' flip not supported."
                "Available values are:"
                "- 'vertical' or 'y' to flip vertically."
                "- 'horizontal' or 'x' to flip horizontally."
                "- 'both' or 'xy' to flip both axis."
            )
        )

    flip_code = FLIP_AXIS[axis]

    flipped_image = cv2.flip(image, flip_code)

    return flipped_image
