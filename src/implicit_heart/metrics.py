# 3rd party libraries
import numpy as np


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    
    #if not im1.shape or not im2.shape:
    #    return empty_score

    # Check if shapes are valid before comparing.
    if im1.shape != im2.shape:
        print(str(im1.shape) + ' x ' + str(im2.shape))
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape." + \
                         str(im1.shape) + ' x ' + str(im2.shape))

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def jaccard(img, ground_truth):
    """
    """
    img = np.asarray(img).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    # Check if shapes are valid before comparing.
    if img.shape != ground_truth.shape:
        print(str(img.shape) + ' x ' + str(ground_truth.shape))
        raise ValueError("Shape mismatch: img and ground_truth must have the same shape." + \
                         str(img.shape) + ' x ' + str(ground_truth.shape))
    
    tp = np.logical_and(img, ground_truth).sum()
    fp = np.logical_and(img, ~ground_truth).sum()
    fn = np.logical_and(~img, ground_truth).sum()

    # 0 / 0 case, no labels in this slice
    if tp == 0 and fp == 0 and fn == 0:
        if ground_truth.sum() == 0:
            return 1
        else:
            return 0        
    return tp / (tp + fp + fn)


def precision(img, ground_truth):
    """
    """
    img = np.asarray(img).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    # Check if shapes are valid before comparing.
    if img.shape != ground_truth.shape:
        print(str(img.shape) + ' x ' + str(ground_truth.shape))
        raise ValueError("Shape mismatch: img and ground_truth must have the same shape." + \
                         str(img.shape) + ' x ' + str(ground_truth.shape))
    
    tp = np.logical_and(img, ground_truth).sum()
    fp = np.logical_and(img, ~ground_truth).sum()
    
    # 0 / 0 case, no labels in this slice
    if tp == 0 and fp == 0:
        if ground_truth.sum() == 0:
            return 1
        else:
            return 0        
    return tp / (tp + fp)


def recall(img, ground_truth):
    """
    """
    img = np.asarray(img).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    # Check if shapes are valid before comparing.
    if img.shape != ground_truth.shape:
        print(str(img.shape) + ' x ' + str(ground_truth.shape))
        raise ValueError("Shape mismatch: img and ground_truth must have the same shape." + \
                         str(img.shape) + ' x ' + str(ground_truth.shape))
    
    tp = np.logical_and(img, ground_truth).sum()
    fn = np.logical_and(~img, ground_truth).sum()
    return tp / (tp + fn)


def _polygon_area(pts) -> float:
    """Shoelace area of a polygon given as an ordered (n, 2) array of points."""
    if len(pts) == 0:  # slice without a boundary
        return 0.0
    x, y = np.asarray(pts, dtype=float).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ejection_fraction(volumes: np.ndarray) -> float:
    """
    Ejection fraction (%) from voxel counts of a sequence of masks indexed as volumes[t, ...].
    """
    counts = [np.count_nonzero(v) for v in volumes]
    return (max(counts) - min(counts)) / max(counts) * 100


def ejection_fraction_from_constraints(imp) -> float:
    """
    Ejection fraction (%) of a 4D ImplicitFunction, estimated per time-step by summing
    the polygon areas enclosed by each slice's boundary constraints.
    """
    areas = []
    for t in sorted(imp.data.constraints4D):
        slices = imp.data.constraints4D[t]
        areas.append(sum(_polygon_area([pt[:2] for pt, value in slices[z] if value == 0])
                         for z in sorted(slices)))
    return (max(areas) - min(areas)) / max(areas) * 100
