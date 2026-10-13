# 3rd party libraries
import cv2
import numpy as np

from .data import APPROX, RET_MODE


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


def ejection_fraction(volumes: np.ndarray) -> float:
    """
    Ejection fraction (%) from voxel counts of a sequence of masks indexed as volumes[t, ...].
    """
    counts = [np.count_nonzero(v) for v in volumes]
    return (max(counts) - min(counts)) / max(counts) * 100


def ejection_fraction_surface(imp, ts=None, step: int = 1) -> float:
    """
    Ejection fraction (%) of a 4D ImplicitFunction from the volumes enclosed by its f = 0 surface
    (marching cubes on the sampled field), taking min/max volume over ts as ES/ED (paper Sec. 3.5.4).
    :param ts: time coordinates to evaluate; default: the input frames.
    :param step: pixel stride in x and y when sampling f.
    """
    from skimage.measure import marching_cubes
    from .visualize import sample_field

    ts = [imp.data._coord(i) for i in range(imp.dim[0])] if ts is None else ts
    volumes = []
    for t in ts:
        # Half-gap samples; repeat the end slices half a gap outward so each slice stands for a slab one
        # slice gap thick (as voxel counting assumes), then close the surface sharply.
        f, _, (dx, dy, dz) = sample_field(imp, t, step, z_samples=2)
        f = np.concatenate([f[:1], f, f[-1:]])
        f = np.pad(f, 1, constant_values=-1e9)
        verts, faces, _, _ = marching_cubes(f, 0.0, spacing=(dz, dy, dx))
        v0, v1, v2 = verts[faces].transpose(1, 0, 2)
        volumes.append(abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum()) / 6)
    return (max(volumes) - min(volumes)) / max(volumes) * 100


def ejection_fraction_polygon(volumes: np.ndarray) -> float:
    """
    Ejection fraction (%) of masks indexed as volumes[t, slice, y, x], with each time-step's
    volume estimated as the summed area of the contour polygons that become boundary constraints
    (outer contours minus holes).
    """
    areas = [sum(_contour_area(img) for img in volume) for volume in volumes]
    return (max(areas) - min(areas)) / max(areas) * 100


def _contour_area(img: np.ndarray) -> float:
    contours, hierarchy = cv2.findContours(img.astype('uint8'), RET_MODE, APPROX)
    if not contours:
        return 0.0
    # With RETR_CCOMP, contours with a parent are holes.
    return sum(cv2.contourArea(c) * (1 if parent == -1 else -1)
               for c, parent in zip(contours, hierarchy[0][:, 3]))
