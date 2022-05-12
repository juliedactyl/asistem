from skimage import morphology
from skimage.feature import canny, corner_harris, corner_peaks
import matplotlib.pyplot as plt
import numpy as np
import cv2
from time import time


def plot_compute_corner_process(corners, square, coords):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4,2))
    axs[0].matshow(corners, cmap='viridis')
    axs[1].imshow(square, cmap=plt.cm.gray)
    axs[1].plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            linestyle='None', markersize=6)
    plt.show()

def compute_corners(approximate_pattern, k=0.05, sigma=8, plot=False):
    '''
    Computes corners of the a binary image of the approximate pattern.

    approximate_pattern = binary 2D image

    *** If four corners are not found, change k and/or sigma ***
    k = 0.05 (float in range (0.02-0.5) )
        Parameter passed to skimage.feature.corner_harris.
        A lower value tends to detect sharper corners

    sigma = 8 (positive integer), should be in range (3, 10)
            Parameter passed to skimage.feature.corner_harris

    plot = False, if True plots the result of corner_harris and the
           computed corner coordinates on the convex hull of the approximate
           pattern.

    returns: corner_coords (numpy array)
    '''
    edges = canny(approximate_pattern, sigma=sigma)
    square = morphology.convex_hull_image(edges)

    corners = corner_harris(square, k=k, sigma=sigma)
    coords = corner_peaks(corners, num_peaks=4, threshold_rel=0.14)

    corner_coords = np.fliplr(coords)
    if plot:
        plot_compute_corner_process(corners, square, coords)
    return corner_coords

def sort_corners(corner_coords):
    # Sorts the corner_coords in the order top_left, top_right, bottom_right, bottom_left
    sorted_corners = corner_coords.copy()
    sumall = np.sum(corner_coords, axis=1)
    # The bottom right corner will always have the highest sum of x and y coords
    # The top left corner will always have the lowest sum of x and y coords
    for i, xy in enumerate(corner_coords):
        if np.sum(xy) >= sumall.max():
            sorted_corners[2] = xy
        elif np.sum(xy) <= sumall.min():
            sorted_corners[0] = xy
        elif xy[0] > xy[1]:
            sorted_corners[1] = xy
        else:
            sorted_corners[3] = xy
    return sorted_corners

def maximise_pattern_fit(bf_image, pattern, corner_coords, g=0.3):
    '''
    Maximising fit of mask by minimising standard deviation of the resulting
    (masked) image.

    The findHomography-function finds the way in which the pattern fits at the
    image "surface". All it needs to know is the corners of the pattern in the
    image. These I have attempted to find by doing some image processing.
    Subsequently, this pattern will be wiggled around a bit to try and find a
    better fit.

    bf_image =  hyperspy signal2D, recreated bright field image to mask with the pattern and
                minimise std in
    pattern =   pattern to use as a mask

    corner_coords = corned coordinates

    g = goal std reduction, default is 30% which is quite high.
        Should be in the interval (10-30%)

    returns: mask (numpy array)
    '''
    img = bf_image.data
    # Define the corners of the pattern
    pts_pattern = np.array([[0, 0],
                            [pattern.shape[1] - 1, 0],
                            [pattern.shape[1] - 1, pattern.shape[0] - 1],
                            [0, pattern.shape[0] - 1]])
    pts = sort_corners(corner_coords)
    homographyMat, status = cv2.findHomography(pts_pattern, pts)
    mask = np.invert(cv2.warpPerspective(pattern, homographyMat,
                                (img.shape[1], img.shape[0])))
    resulting_image = img*mask

    init_std = np.std(resulting_image)
    goal = init_std*(1-g)

    print(f'Initial std: {round(init_std,3)}')
    print(f'Goal: {round(goal,3)}')
    w = 30
    h = 30
    milestone = init_std*0.995
    temp_pts = pts.copy()
    new_pts = temp_pts.copy()
    tic = time()
    while np.std(resulting_image) > goal:
        if np.std(resulting_image) < milestone:
            print(f'Still working, but found {round(np.std(resulting_image),3)}')
            milestone = np.std(resulting_image)
            temp_pts = new_pts.copy()
            if w > 2:
                w -= 2
            if h > 2:
                h -= 2
        new_pts = temp_pts.copy()
        for i, point in enumerate(new_pts):
            new_pts[i,0] = np.random.randint(temp_pts[i,0]-w, temp_pts[i,0]+w)
            new_pts[i,1] = np.random.randint(temp_pts[i,1]-h, temp_pts[i,1]+h)
        homographyMat, status = cv2.findHomography(pts_pattern, new_pts)
        mask = np.invert(cv2.warpPerspective(pattern, homographyMat,
                                             (img.shape[1], img.shape[0])))
        resulting_image = img*mask
        if time() - tic > 300:
            homographyMat, status = cv2.findHomography(pts_pattern, temp_pts)
            mask = np.invert(cv2.warpPerspective(pattern, homographyMat,
                                                 (img.shape[1], img.shape[0])))
            print(f'Timed out with std = {round(milestone,3)}')
            break
    toc = time()
    print(f'Finished after {round(toc-tic,3)}s,')
    print(f'with std = {round(np.min([np.std(resulting_image),milestone]),3)}.')
    return mask
