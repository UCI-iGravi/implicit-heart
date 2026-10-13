# 3rd party imports
import matplotlib.pyplot as plt
import numpy as np


def show(image: np.ndarray, title: str = "") -> None:
    """Displays an image

    Args:
        image (np.ndarray): [Image to be displayed]
        title (str, optional): [Title of the image]. Defaults to "".
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    
    
def show_contour(image: np.ndarray, icontour: np.ndarray, ocontour: np.ndarray = None, title: str = "") -> None:
    """Displays an image with a contour overlayed

    Args:
        image (np.ndarray): [Image to be displayed]
        icontour (np.ndarray): [Inner contour to be overlayed]
        ocontour (np.ndarray): [Outer contour to be overlayed]. Defaults to None.
        title (str, optional): [Title of the image]. Defaults to "".
    """
    plt.imshow(image, cmap='gray')
    plt.contour(icontour, colors='r')
    plt.plot([], [], color='r', label='Inner contour')
    if ocontour is not None:
        plt.contour(ocontour, colors='b')
        plt.plot([], [], color='b', label='Outer contour')
    plt.title(title)
    plt.legend()
    plt.show()