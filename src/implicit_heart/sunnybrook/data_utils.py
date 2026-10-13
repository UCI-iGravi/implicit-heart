# Standard imports


# Third-party imports
import numpy as np


def get_contour(contour_dict: dict, idx: int, verbose: bool = False) -> np.ndarray:
    """Returns the closest contour from the contour dictionary.

    Args:
        contour_dict (dict): [Dictionary containing the contours]
        idx (int): [Contour index]
        verbose (bool, optional): [Whether to print extra information]. Defaults to False.

    Returns:
        np.ndarray: [The closest contour mask to the index]
    """
    contour_indices = list(contour_dict.keys())
    closest = min(contour_indices, key=lambda x: abs(x - idx))
    if verbose:
        print(f"Closest contour index: {closest}")
    return contour_dict[closest]