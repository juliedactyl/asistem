import hyperspy.api as hs
import numpy as np
from skimage import transform, morphology
from skimage.exposure import rescale_intensity, match_histograms
from fpd.ransac_tools import ransac_im_fit


def old_level_intensity(s_signal, corner_size=0.05, only_offset=False):
    '''
    Levels the intensity of the signal by generating a flat signal to
    correct ramp against.

    s_signal = hyperspy 2d signal

    only_offset = False, passed to correct_ramp() from pyxem

    corner_size = 0.05, size of the corners to fit a plane to for leveling
                  the intensity. Passed to correct_ramp() from pyxem.
    '''
    # Generate a dummy signal to level the intensity
    level_signal = hs.signals.Signal2D(np.ones(s_signal.data.shape))
    # s_signal = rescale_intensity(np.asarray(s_signal), out_range=(1, -1))
    # s_signal = hs.signals.Signal2D(s_signal)
    comb_signal = (level_signal, s_signal)
    comb_signal = hs.stack(comb_signal)
    comb_signal.set_signal_type('dpc')
    comb_signal.change_dtype('float64')
    corr = comb_signal.correct_ramp(corner_size=corner_size, only_offset=only_offset)
    maxval = np.max((np.max(corr.data), np.abs(np.min(corr.data))))
    img = corr.data[1]/maxval
    return hs.signals.Signal2D(img)

def level_intensity(signal):
    s_corr = signal.deepcopy()
    maxval = np.max((np.max(s_corr.data), np.abs(np.min(s_corr.data))))
    norm_s = np.array(s_corr.data/maxval, dtype='float64')
    ransac_output_norm = ransac_im_fit(norm_s, max_trials=10)
    return hs.signals.Signal2D(norm_s-ransac_output_norm[0])

def recreate_bf_image(ss, sn, se, sw, return_all=False):
    '''
    Recreates a bright field image by aligning and summing the images from four
    edges of the ADF detector.

    ss, sn, se, sw = leveled hyperspy signals from south, north, east and west
                     sides respectively
    return_all = False (default), whether or not to return ss, sn, se and sw.
                 Sometimes they will be cropped and you'll need the cropped
                 ones for further processing.
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


def calculate_dpc_image(ss, sn, sw, se, mask, coords=None, crop=True):
    '''
    Calculates the dpc colour image from the four edge signals.

    coords = coordinates to the corners of the pattern, used to crop the image
             if None is provided, crop is forced to be False

    ss, sn, sw, se = hyperspy Signal2D, images from south, north, west and east
                     edges of the ADF detector respectively

    mask = 2D np.array, the mask as made by the pattern fitting

    crop = True, whether or not to crop the image
    '''
    if coords is None:
        crop = False
    if crop:
        minxy = np.min(coords, axis=0)
        maxxy = np.max(coords, axis=0)
        # print(minxy, maxxy)
        y1 = minxy[1]-50
        y2 = maxxy[1]+100
        x1 = minxy[0]-50
        x2 = x1+(y2-y1)
        if y1 < 0 or x1 < 0:
            y1 = minxy[1]
            y2 = maxxy[1]+50
            x1 = minxy[0]
            x2 = x1+(y2-y1)
        # print(f'{x1}:{x2}, {y1}:{y2}')
        ss_ = ss.isig[x1:x2, y1:y2]
        sn_ = sn.isig[x1:x2, y1:y2]
        sw_ = sw.isig[x1:x2, y1:y2]
        se_ = se.isig[x1:x2, y1:y2]
        m = mask[y1:y2, x1:x2]
    else:
        ss_, sn_, sw_, se_ = ss, sn, sw, se
        m = mask

    ss_= match_histograms(np.asarray(ss_), np.asarray(ss_))
    sn_= match_histograms(np.asarray(sn_), np.asarray(ss_))
    sw_= match_histograms(np.asarray(sw_), np.asarray(ss_))
    se_= match_histograms(np.asarray(se_), np.asarray(ss_))
    m_ = hs.signals.Signal2D(np.array(m, dtype='bool'))

    # This makes a masked signal
    m_sy = (sn_ - ss_)*m_
    m_sx = (sw_ - se_)*m_
    m_s = (m_sy, m_sx)
    m_s = hs.stack(m_s)
    m_s.set_signal_type('dpc')

    # This makes an unmasked signal
    sy_ = (sn_ - ss_)
    sx_ = (sw_ - se_)
    sy = hs.signals.Signal2D(sy_)
    sx = hs.signals.Signal2D(sx_)
    s = (sy, sx)
    s = hs.stack(s)
    s.set_signal_type('dpc')

    return m_s, s
