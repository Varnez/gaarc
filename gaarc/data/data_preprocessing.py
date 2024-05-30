import random

import cv2
import numpy as np

# pylint: disable=no-member


def padd_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    padding_value: int = -1,
    frame_size: int = 0,
    frame_value: int = -2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Pads image up to an specific height and width.
    Also, it's possible to also generate a frame with another value around the original image.

    Will return the padded image along with a tuple containing the size of each padding, ordered as
    (top, bottom, left, right)
    Parameters
    ----------
    image : np.ndarray
        Original image.
    target_height : int
        Height to pad the image to.
    target_width : int
        Width to pad the image to.
    padding_value : int, optional
        Value to use for the padding.
        By default -1
    frame_size : int, optional
        Size of the frame around the image, as the amount of positions in each direction.
        By default 0
    frame_value : int, optional
        Value to use for the frame.
        By default -2

    Returns
    -------
    tuple[np.ndarray, tuple[int, int, int, int]]
        Tuple with the padded image and the size of each padding, ordered as (top, bottom, left,
        right)
    """
    original_height, original_width = image.shape

    # pylint: disable-next=no-member
    frammed_image = cv2.copyMakeBorder(  # type: ignore
        image,
        frame_size,
        frame_size,
        frame_size,
        frame_size,
        cv2.BORDER_CONSTANT,  # pylint: disable=no-member
        value=frame_value,
    )

    vertical_pad = target_height - original_height - frame_size * 2
    horizontal_pad = target_width - original_width - frame_size * 2

    top = vertical_pad // 2
    bottom = vertical_pad // 2
    left = horizontal_pad // 2
    right = horizontal_pad // 2

    if vertical_pad % 2 != 0:
        top += 1

    if horizontal_pad % 2 != 0:
        left += 1

    # pylint: disable-next=no-member
    padded_image = cv2.copyMakeBorder(  # type: ignore
        frammed_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,  # pylint: disable=no-member
        value=padding_value,
    )

    return padded_image, (top, bottom, left, right)


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
