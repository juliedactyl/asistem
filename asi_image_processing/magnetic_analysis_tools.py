import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import label

class ASI_info:
    def __init__(self, fname, scan_rotation, pattern_rotation, tilt_info):
        self.fname = fname
        self.scan_rotation = scan_rotation
        self.pattern_rotation = pattern_rotation
        self.tilt_info = tilt_info

class Magnet:
    def __init__(self, coordinates, deflection):
        # Coordinates: [[start_x, start_y], [middle_x, middle_y], [end_x, end_y]]
        self.coordinates = coordinates
        self.deflection = deflection

def show_segmentation(seg1, label_image, ws):
    # Show the segmentations.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                             sharex=True, sharey=True)

    color1 = label2rgb(seg1, image=np.abs(label_image), bg_label=0)
    axes[0].imshow(color1)
    axes[0].set_title('Sobel+Watershed')

    axes[1].imshow(seg1)
    axes[1].set_title('Segmented image')

    [axi.set_axis_off() for axi in axes.ravel()]
    fig.tight_layout()


def get_segmentation_of_dpc_image(dpc_image, plot=False):
    '''
    Takes a masked DPC image of an ASI and separates the individual magnets.
    It is very important that the magnets do not overlap.

    dpc_image = hyperspy DPCsignal, the masked image to be segmented

    plot = False, choose True if you want to see plots of the segmentation
    '''
    # Need a 2D image to label the magnets using skimage
    # Collapsing the 0th axis to its mean value
    label_image = np.mean(dpc_image.data, axis=0)

    ## This code is adapted from the Skimage documentation:
    # https://scikit-image.org/docs/dev/auto_examples/segmentation/
    # plot_expand_labels.html#sphx-glr-auto-examples-segmentation-plot-expand-labels-py
    # Make segmentation using edge-detection and watershed.
    edges = sobel(label_image)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(label_image)
    foreground, background = 1, 2
    markers[label_image == 0] = background
    markers[label_image != 0] = foreground

    ws = watershed(edges, markers)
    seg1 = label(ws == foreground)

    if plot:
        show_segmentation(seg1, label_image, ws)
    return seg1

def sort_magnets(sorting_array, magnets):
    temp_array = np.copy(sorting_array)
    sorted_magnets = []
    while len(temp_array[:,0]) > 0:
        index = np.where(temp_array[:,1] == np.min(temp_array[:,1]))[0][0]
        sorting_array = temp_array[index]
        for i in range(len(temp_array)):
            if temp_array[i,1] < sorting_array[1]+10 and temp_array[i,0] < sorting_array[0]:
                sorting_array = temp_array[i]
        n = np.where(temp_array[:,2:] == sorting_array[2:])[0][0]
        sorted_magnets.append(magnets[int(sorting_array[2])-2])
        temp_array = np.delete(temp_array, n, axis=0)
    return sorted_magnets

def extract_magnet_information(seg1, dpc_image):
    # Make one array to save all the necessary info about the magnets
    # and one to save the middle points and magnet number for sorting purposes.
    N = np.max(seg1)
    magnets_unsorted = []
    midpoints = np.zeros((N-1,3))
    # Since this for-loop starts from 2, it automatically
    # disregards the magnetic area surrounding the pattern.
    for n in range(2,N+1):
        magnet = np.where(seg1 == n)
        magnet_coords = np.fliplr(np.stack(magnet, axis=1))
        middle = [int((magnet_coords[0][0]+magnet_coords[-1][0])/2),
                  int((magnet_coords[0][1]+magnet_coords[-1][1])/2)]
        # Defining coordinates: [[start_x, start_y], [middle_x, middle_y], [end_x, end_y]]
        defining_coordinates = np.array([magnet_coords[0], middle, magnet_coords[-1]])

        deflection = np.zeros(magnet_coords.shape)
        for i, coord in enumerate(magnet_coords):
            deflection[i] = dpc_image.data[:,coord[1],coord[0]]
        m1 = Magnet(defining_coordinates, deflection)
        magnets_unsorted.append(m1)
        midpoints[n-2] = [middle[0], middle[1], n]

    magnets_sorted = sort_magnets(midpoints, magnets_unsorted)
    return magnets_sorted
