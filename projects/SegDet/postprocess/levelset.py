import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io


def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


img = io.imread('twoObj.bmp')
img = color.rgb2gray(img)
img = img - np.mean(img)

# Smooth the image to reduce noise and separation between noise and edge becomes clear
img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)

F = stopping_fun(img_smooth)

def default_phi(x):
    # Initialize surface phi at the border (5px from the border) of the image
    # i.e. 1 outside the curve, and -1 inside the curve
    phi = np.ones(x.shape[:2])
    phi[5:-5, 5:-5] = -1.
    return phi


dt = 1.
n_iter = 100
for i in range(n_iter):
    dphi = grad(phi)
    dphi_norm = norm(dphi)

    dphi_t = F * dphi_norm

    phi = phi + dt * dphi_t

    # plot the zero level curve of phi
    plt.contour(phi, 0)
    plt.show()