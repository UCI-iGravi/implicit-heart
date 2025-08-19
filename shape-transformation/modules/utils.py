# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt


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
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    
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
    img = np.asarray(img).astype(np.bool)
    ground_truth = np.asarray(ground_truth).astype(np.bool)

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
    img = np.asarray(img).astype(np.bool)
    ground_truth = np.asarray(ground_truth).astype(np.bool)

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
    img = np.asarray(img).astype(np.bool)
    ground_truth = np.asarray(ground_truth).astype(np.bool)

    # Check if shapes are valid before comparing.
    if img.shape != ground_truth.shape:
        print(str(img.shape) + ' x ' + str(ground_truth.shape))
        raise ValueError("Shape mismatch: img and ground_truth must have the same shape." + \
                         str(img.shape) + ' x ' + str(ground_truth.shape))
    
    tp = np.logical_and(img, ground_truth).sum()
    fn = np.logical_and(~img, ground_truth).sum()
    return tp / (tp + fn)


def visualize_constraints(I: np.ndarray, b: np.ndarray, n: np.ndarray):
    """
    Visualize boundary and normal constraints on a 2D image I.
    :param I: some NumPy array representing the object to interpolate
    :param b: NumPy array representing the boundary constraints
    :param n: NumPy array representing the normal constraints
    """
    plt.imshow(I)
    for pt in b:
        plt.plot(pt[0], pt[1], 'ro')
    for pt in n:
        plt.plot(pt[0], pt[1], 'gX')
    plt.show()
    
    
def visualize_block(img: np.ndarray):
    """
    Visualize individual pixel block values of an image.
    :param img: some NumPy image to visualize
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ax.text(j, i, str(img[i, j].round(2)), color='r', ha='center', va='center')
    plt.show()