import hyperspy.api as hs
import numpy as np
from skimage import transform, morphology

def level_intensity(s_signal):
    '''
    Levels the intensity of the signal by generating a flat signal to
    correct ramp against.

    s_signal = hyperspy 2d signal
    '''
    # Generate a dummy signal to level the intensity
    level_signal = hs.signals.Signal2D(np.ones(s_signal.data.shape))
    # s_signal = rescale_intensity(np.asarray(s_signal), out_range=(1, -1))
    # s_signal = hs.signals.Signal2D(s_signal)
    comb_signal = (level_signal, s_signal)
    comb_signal = hs.stack(comb_signal)
    comb_signal.set_signal_type('dpc')
    comb_signal.change_dtype('float64')
    corr = comb_signal.correct_ramp()
    img = corr.data[1]/(np.max(corr.data))
    return hs.signals.Signal2D(img)

def recreate_bf_image(ss, sn, se, sw):
    '''
    Recreates a bright field image by aligning and summing the images from four
    edges of the ADF detector.

    ss, sn, se, sw = leveled hyperspy signals from south, north, east and west
                     sides respectively
    '''
    # Sometimes the signals are not properly in the same place and the "BF" image looks blurred.
    stack = (ss,sn,se,sw)
    stack = hs.stack(stack)
    stack.align2D()
    ss = hs.signals.Signal2D(stack.data[0])
    sn = hs.signals.Signal2D(stack.data[1])
    se = hs.signals.Signal2D(stack.data[2])
    sw = hs.signals.Signal2D(stack.data[3])
    # Create a "BF" image which you can use for the corner detection.
    return ss+sn+se+sw

def compute_max_and_min(derimg, d=64):
    '''
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
    derimg_min = morphology.remove_small_objects(derimg < -np.var(derimg), min_size=d)
    derimg_max = morphology.remove_small_objects(derimg > np.var(derimg), min_size=d)
    minmax = np.bitwise_or(derimg_min, derimg_max)
    # minmax_ = morphology.remove_small_objects(minmax, min_size=100)
    # minmax_ = morphology.binary_dilation(minmax_)
    return minmax  # , minmax_

def compute_approximate_pattern(bf_img):
    der0 = bf_img.derivative(axis=0)
    der1 = bf_img.derivative(axis=1)
    der0_ = transform.resize(der0.data, bf_img.data.shape)
    der1_ = transform.resize(der1.data, bf_img.data.shape)
    dersum = der0_+der1_
    derdiff = der0_-der1_

    dersum_minmax = compute_max_and_min(dersum, d=64)
    derdiff_minmax = compute_max_and_min(derdiff, d=64)

    ap_org = morphology.binary_closing(np.bitwise_or(dersum_minmax, derdiff_minmax))
    ap_org = morphology.remove_small_objects(ap_org, min_size=400)

    approximate_pattern = morphology.binary_closing(ap_org)
    approximate_pattern = morphology.remove_small_objects(approximate_pattern, min_size=400)
    approximate_pattern = morphology.binary_dilation(approximate_pattern)
    return approximate_pattern
