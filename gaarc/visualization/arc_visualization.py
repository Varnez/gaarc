import matplotlib.pyplot as plt
import numpy as np

ARC_COLOR_MAP = {
    -1: np.array([242, 242, 242]),  # Grey-ish white
    0: np.array([0, 0, 0]),  # Black
    1: np.array([0, 116, 217]),  # Blue
    2: np.array([255, 65, 5]),  # Red
    3: np.array([46, 204, 64]),  # Green
    4: np.array([255, 220, 0]),  # Yellow
    5: np.array([170, 170, 170]),  # Grey
    6: np.array([240, 18, 19]),  # Fuschia
    7: np.array([255, 133, 27]),  # Orange
    8: np.array([127, 219, 255]),  # Teal
    9: np.array([135, 12, 37]),  # Brown
    "unknown": np.array([255, 255, 255]),
}


def color_map_arc_sample(arc_sample: np.ndarray) -> np.ndarray:
    """
    Transforms a an arc sample 2D tensor, to a 3D tensor with the canonical rgb
    value of each class.

    Parameters
    ----------
    arc_sample : np.ndarray
        2 dimensional tensor containing an ARC sample.

    Returns
    -------
    np.ndarray
        3 dimensional tensor containing the rgb values of each class.
    """
    color_mapped_sample = np.zeros(arc_sample.shape + (3,))

    for h in range(arc_sample.shape[0]):
        for w in range(arc_sample.shape[1]):
            for c in range(3):
                color_mapped_sample[h, w, c] = ARC_COLOR_MAP.get(
                    int(arc_sample[h, w]), ARC_COLOR_MAP["unknown"]
                )[c]

    color_mapped_sample /= 255

    return color_mapped_sample


def plot_arc_sample(arc_sample: np.ndarray) -> None:
    """
    Tool function to automate the plotting of arc samples, making the rgb
    mapping and plotting together.

    Parameters
    ----------
    arc_sample : np.ndarray
            2 dimensional tensor containing an ARC sample.
    """
    color_mapped_sample = color_map_arc_sample(arc_sample)

    plt.imshow(color_mapped_sample)
    plt.show()
