import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.measure import label
from tqdm import tqdm

class ASI_info:
    def __init__(self, scan_rotation, pattern_rotation, tilt_info):
        self.scan_rotation = scan_rotation
        self.pattern_rotation = pattern_rotation
        self.tilt_info = tilt_info

class Magnet:
    def __init__(self, coordinates, deflection):
        # Coordinates: [[start_x, start_y], [middle_x, middle_y], [end_x, end_y]]
        self.coordinates = coordinates
        self.deflection = deflection

def show_segmentation(seg1, label_image, ws):
    '''
    Show the segmentation of the ASI image.
    Runs automatically if plot=True is passed to
    asistem.magnetic_analysis.get_segmentation_of_dpc_image()
    '''
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

    plot = False, choose True if you want to see plots of the segmentation.


    returns: segmented image where each magnet has its own, unique number
             (numpy array)
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
    '''
    Sorts the magnets so that they appear in the array in the same order as in
    the image from the top left to the bottom right.
    '''
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
    '''
    Make one array to save all the necessary info about the magnets
    and one to save the middle points and magnet number for sorting purposes.

    '''
    N = np.max(seg1)
    magnets_unsorted = []
    midpoints = np.zeros((N-1,3))
    # This for-loop starts from 2, such that it automatically
    # disregards the magnetic area surrounding the pattern, which is
    # useful for FIB samples.
    # This means that for EBL samples, a small area in the top left corner
    # should be forced to be True
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


def calculate_magnetic_direction(deflection_arr, theta):
    '''
    Takes an array of electron beam deflection vectors for a single magnet,
    and the scan rotation (theta), and calculates the direction of magnetisation
    for the magnet using the median of the beam defleciton.


    deflection_arr = numpy array, deflection at all pixels contained in the
                     magnet

    theta = scan rotation


    returns: median magnetisation of the magnet (numpy array)
    '''
    mediandefx = np.median(deflection_arr[:,0])
    mediandefy = np.median(deflection_arr[:,1])
    F_L = np.array([mediandefx*np.cos(theta)-mediandefy*np.sin(theta),
                    mediandefx*np.sin(theta)+mediandefy*np.cos(theta), 0])
    v_e = np.array([0, 0, 1])
    B = np.cross(F_L, v_e)
    return B[0:2]

def generate_fixed_position_lattice(magnets):
    '''
    Generates the grid positions in which to plot the arrows.
    NB! Assumes a 10/11 by 10/11 array of magnets.

    magnets = list of all 220 magnets


    returns: coordinates of fixed positions (numpy array)
    '''
    positions = np.zeros((len(magnets),2))
    y = 0
    c = 0
    for i, position in enumerate(positions):
        if y%2 == 0:
            positions[i, 0] = c%10*2+1
            positions[i, 1] = y
            c += 1
            if c == 10:
                c = 0
                y += 1
        else:
            positions[i, 0] = c%10*2
            positions[i, 1] = y
            if c%10 == 0 and c != 0:
                positions[i, 0] = 20
            c += 1
            if c == 11:
                c = 0
                y += 1
    return np.array(positions)

def analyse_artificial_spin_ice(magnets, asiinfo, variance_threshold=0.05, angle_threshold=50):
    '''
    Takes an array of magnet object and an asi object and analyses it for
    plotting.

    magnets = array of magnet objects

    asi = array of information about the ASI
          [scan_rotation, pattern_rotation, tilt_info]

    variance_threshold = 0.05, default threshold for the variance of electron
    deflection within a magnet for the magnet to be accepted. Increase if
    need be.

    returns: arrows, points, approx_macrospin, points_fixed, colours
    '''
    asi = ASI_info(asiinfo[0], asiinfo[1], asiinfo[2])
    positions = generate_fixed_position_lattice(magnets)
    arrows = np.zeros((len(magnets),4))
    approx_macrospin = np.zeros((len(magnets),4))
    colours = np.zeros(len(magnets), dtype='object')
    points = []
    points_fixed = []
    sq_counter = 0
    for n in tqdm(range(0, len(magnets))):
        M = calculate_magnetic_direction(magnets[n].deflection, -asi.scan_rotation)
        x0y0 = magnets[n].coordinates[0]
        x1y1 = magnets[n].coordinates[1]
        x2y2 = magnets[n].coordinates[2]

        # Calculating the magnet vectors (mv) and
        # translating them to pattern-specific unit vectors
        mv0 = [x0y0[0]-x1y1[0], x0y0[1]-x1y1[1]]
        mv2 = [x2y2[0]-x1y1[0], x2y2[1]-x1y1[1]]
        umv0_ = mv0/(np.sqrt(np.dot(mv0,mv0)))
        umv2_ = mv2/(np.sqrt(np.dot(mv2,mv2)))
        # Find vectors aligning with the pattern rotation
        rot = asi.pattern_rotation/360*2*np.pi
        unit_vectors = np.array([[np.cos(rot          ), -np.sin(rot)],
                                 [np.cos(rot+np.pi/2  ), -np.sin(rot+np.pi/2)],
                                 [np.cos(rot+np.pi    ), -np.sin(rot+np.pi)],
                                 [np.cos(rot+np.pi*3/2), -np.sin(rot+np.pi*3/2)]
                                ])
        colour_choices = np.array(['tab:green', 'mediumblue', 'red', 'gold'])

        # Determine which unit vector is closest to the magnet vector
        alpha0, alpha2 = 360, 360
        if asi.pattern_rotation == 0:
            # This is a brute force way to determine this for sqaure ASI.
            # It assumes the same lattice configuration as the function generate_fixed_position_lattice()
            if sq_counter < 10:
                # The magnet is horisontal
                umv0 = unit_vectors[2]
                col0 = colour_choices[2]
                umv2 = unit_vectors[0]
                col2 = colour_choices[0]
                sq_counter += 1
            else:
                # The magnet is vertical
                umv0 = unit_vectors[3]
                col0 = colour_choices[3]
                umv2 = unit_vectors[1]
                col2 = colour_choices[1]
                sq_counter += 1
                if sq_counter == 21:
                    sq_counter = 0
        else:
            for i, uv in enumerate(unit_vectors):
                temp_alpha0 = float(np.arccos(np.dot(uv, umv0_))/(2*np.pi)*360)
                if temp_alpha0 < alpha0:
                    alpha0 = temp_alpha0
                    umv0 = uv
                    col0 = colour_choices[i]
                temp_alpha2 = float(np.arccos(np.dot(uv, umv2_))/(2*np.pi)*360)
                if temp_alpha2 < alpha2:
                    alpha2 = temp_alpha2
                    umv2 = uv
                    col2 = colour_choices[i]
        uM = M/(np.sqrt(np.dot(M,M)))
        angle0 = int(np.arccos(np.dot(umv0, uM))/(2*np.pi)*360)
        angle2 = int(np.arccos(np.dot(umv2, uM))/(2*np.pi)*360)

        varx = np.var(magnets[n].deflection[:,0])
        vary = np.var(magnets[n].deflection[:,1])
        arrows[n] = [x1y1[0],x1y1[1],uM[0]*100,uM[1]*100]
        if (angle0 <= angle_threshold or angle2 <= angle_threshold) and varx < variance_threshold and vary < variance_threshold:
            if angle0 < angle2:
                approx_macrospin[n] = [positions[n,0]-umv0[0]/2,
                                       positions[n,1]-umv0[1]/2, umv0[0], umv0[1]]
                colours[n] = col0
            else:
                approx_macrospin[n] = [positions[n,0]-umv2[0]/2,
                                       positions[n,1]-umv2[1]/2, umv2[0], umv2[1]]
                colours[n] = col2
        else:
            points.append(x1y1)
            points_fixed.append(positions[n])
            colours[n] = 'k'
    points = np.array(points)
    points_fixed = np.array(points_fixed)
    return arrows, points, approx_macrospin, points_fixed, colours
