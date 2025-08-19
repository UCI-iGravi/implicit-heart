# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt


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