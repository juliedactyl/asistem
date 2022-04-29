from skimage import morphology
from skimage.feature import canny, corner_harris, corner_peaks
import matplotlib.pyplot as plt


def plot_compute_corner_process(corners, square, coords):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4,2))
    axs[0].matshow(corners, cmap='viridis')
    axs[1].imshow(square, cmap=plt.cm.gray)
    axs[1].plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    plt.show()

def compute_corners(approximate_pattern, k=0.05, sigma=4, plot=False):
    '''
    Computes corners of the a binary image of the approximate pattern.

    approximate_pattern = binary 2D image

    *** If four corners are not found, change k and/or sigma ***
    k = 0.05 (float in range (0.02-0.5) )
        Parameter passed to skimage.feature.corner_harris.
        A lower value tends to detect sharper corners

    sigma = 4 (positive integer)
            Parameter passed to skimage.feature.corner_harris

    plot = False, if True plots the result of corner_harris and the
           computed corner coordinates on the convex hull of the approximate
           pattern.
    '''
    edges = canny(approximate_pattern, sigma=sigma)
    square = morphology.convex_hull_image(edges)

    corners = corner_harris(square, k=k, sigma=sigma)
    coords = corner_peaks(corners, min_distance=10, num_peaks=4, threshold_rel=0.14)

    corner_coords = np.fliplr(coords)
    if plot:
        plot_compute_corner_process(corners, square, coords)
    return corner_coords
