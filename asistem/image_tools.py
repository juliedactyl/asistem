import hyperspy.api as hs
import numpy as np
from skimage import transform, morphology
from skimage.exposure import match_histograms
from fpd.ransac_tools import ransac_im_fit

def level_intensity(signal, mask=None, max_trials=100):
    '''
    Levels slow varying background intensity due to imperfect de-scan in large
    area scan images using the ransac (Random Sample Consensus) model.

    For more information on the ransac method, see:
    https://fpdpy.gitlab.io/fpd/_modules/fpd/ransac_tools.html


    signal = hyperspy.signals.Signal2D, image to be intensity leveled.


    returns: hyperspy.signals.Signal2D, intensity leveled image
    '''
    s_corr = signal.deepcopy()
    maxval = np.max((np.max(s_corr.data), np.abs(np.min(s_corr.data))))
    norm_s = np.array(s_corr.data/maxval, dtype='float64')
    ransac_output_norm = ransac_im_fit(norm_s, mask=mask, max_trials=max_trials)
    return hs.signals.Signal2D(norm_s-ransac_output_norm[0])

def recreate_bf_image(ss, sn, se, sw, return_all=False):
    '''
    Deprecated in asistem 0.0.2

    Recreates a bright field image by aligning and summing the images from four
    edges of the ADF detector.


    ss, sn, se, sw = leveled hyperspy signals from south, north, east and west
                     sides respectively
    return_all = False (default), whether or not to return ss, sn, se and sw.
                 Sometimes they will be cropped and you'll need the cropped
                 ones for further processing.


    returns: hyperspy.signals.Signal2D, pseudo-BF image
    '''
    stack = (ss,sn,se,sw)
    stack = hs.stack(stack)
    stack.align2D()
    ss = hs.signals.Signal2D(stack.data[0])
    sn = hs.signals.Signal2D(stack.data[1])
    se = hs.signals.Signal2D(stack.data[2])
    sw = hs.signals.Signal2D(stack.data[3])
    if return_all:
        return ss+sn+se+sw, ss, sn, se, sw
    else:
        return ss+sn+se+sw

def compute_max_and_min(derimg, d=64):
    '''
    Deprecated in asistem 0.0.2
    
    Computes a binary image of the pattern position in a BF image of a
    FIB-made ASI pattern.

    This is done by computing binary images of where the given image
    (derimg = derivated image) is less than (derimg_min) and greater than
    (derimg_max) the variance of the image.
    The smallest objects are removed and the two images are combined with a
    bitwise or.

    derimg = derivated images

    d = positive integer, minimum size of features to be included in the
        binary image. Default 64.
    '''
    variance = np.var(derimg)
    derimg_min = morphology.remove_small_objects(derimg < -variance, min_size=d)
    derimg_max = morphology.remove_small_objects(derimg > variance, min_size=d)
    minmax = np.bitwise_or(derimg_min, derimg_max)
    minmax_ = morphology.remove_small_objects(minmax, min_size=4*d)
    return minmax_

def compute_approximate_pattern(bf_img):
    '''
    Deprecated in asistem 0.0.2
    Compute approximate pattern from thresholding the BF image instead.
    E.g.
    thresh = np.average(bf)+.3
    approximate_pattern = ap = morphology.binary_dilation(
        bf.data > thresh
    )

    ap = morphology.remove_small_objects(ap, min_size=200)
    ________________________________________________________
    Takes a pseudo-BF image and estimates where the FIB milled pattern is
    based on the derivative of the image in x and y directions.

    bf_img = hypserspy.signals.Signal2D, pseudo-BF image

    returns: 2D numpy array, dtype=bool
    '''
    der0 = bf_img.derivative(axis=0)
    der1 = bf_img.derivative(axis=1)
    der0_ = transform.resize(der0.data, bf_img.data.shape)
    der1_ = transform.resize(der1.data, bf_img.data.shape)
    dersum = der0_+der1_
    derdiff = der0_-der1_

    dersum_minmax = compute_max_and_min(dersum, d=128)
    derdiff_minmax = compute_max_and_min(derdiff, d=128)
    ap_org = morphology.binary_closing(np.bitwise_or(dersum_minmax, derdiff_minmax))
    ap_org = morphology.remove_small_objects(ap_org, min_size=2)

    approximate_pattern = morphology.binary_closing(ap_org)

    approximate_pattern = morphology.remove_small_objects(approximate_pattern, min_size=10)
    approximate_pattern = morphology.binary_dilation(approximate_pattern)
    return approximate_pattern


def calculate_dpc_image(signals, mask, crop=False, coords=None):
    '''
    Calculates the DPC colour image from intensity leveled BF images with
    magnetic contrast from south, north, west and east edges of the ADF
    detector.

    signals = list of HyperSpy Signal2D, 
              images from west, south, east, and north
              edges of ADF detector (in that order)

    mask = 2D np.array, the mask as made by the pattern fitting

    crop = bool, 
           whether or not to crop the image
           defaults to False
           NB! The cropping may work poorly if the
           ASI pattern is close to an image edge

    coords = coordinates to the corners of the pattern, used to crop the image
             if None is provided, crop is forced to be False

    returns: hyperspy.signals.DPCsignal x2, masked and unmasked signals
    '''
    if coords is None:
        crop = False
    if crop:
        minxy = np.min(coords, axis=0)
        maxxy = np.max(coords, axis=0)
        y1 = minxy[1]-50
        y2 = maxxy[1]+100
        x1 = minxy[0]-50
        x2 = x1+(y2-y1)
        if y1 < 0 or x1 < 0:
            y1 = minxy[1]
            y2 = maxxy[1]+50
            x1 = minxy[0]
            x2 = x1+(y2-y1)
        sw = signals[0].isig[x1:x2, y1:y2]
        ss = signals[1].isig[x1:x2, y1:y2]
        se = signals[2].isig[x1:x2, y1:y2]
        sn = signals[3].isig[x1:x2, y1:y2]
        m = mask[y1:y2, x1:x2]
    else:
        sw, ss, se, sn = signals[0],signals[1],signals[2],signals[3]
        m = mask
    sw = match_histograms(np.asarray(sw), np.asarray(sw))
    ss = match_histograms(np.asarray(ss), np.asarray(sw))
    se = match_histograms(np.asarray(se), np.asarray(sw))
    sn = match_histograms(np.asarray(sn), np.asarray(sw))
    mask_signal = hs.signals.Signal2D(np.array(m, dtype='bool'))

    # This makes a masked signal
    m_sx = (sw - se)*mask_signal
    m_sy = (sn - ss)*mask_signal
    m_s = (m_sy, m_sx)
    m_s = hs.stack(m_s)
    m_s.set_signal_type('dpc')

    # This makes an unmasked signal
    sx = (sw - se)
    sy = (sn - ss)
    sx = hs.signals.Signal2D(sx)
    sy = hs.signals.Signal2D(sy)
    s = (sy, sx)
    s = hs.stack(s)
    s.set_signal_type('dpc')

    return m_s, s
