import cv2
import numpy as np

# For cv2, which causes a lot of false positives
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

    frammed_image = cv2.copyMakeBorder(  # type: ignore[call-overload]
        image,
        frame_size,
        frame_size,
        frame_size,
        frame_size,
        cv2.BORDER_CONSTANT,
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

    padded_image = cv2.copyMakeBorder(  # type: ignore[call-overload]
        frammed_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_value,
    )

    return padded_image, (top, bottom, left, right)
