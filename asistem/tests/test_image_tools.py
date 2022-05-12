import asi_image_processing.image_tools as aipit
import hyperspy.api as hs
import numpy as np

def test_level_intensity():
    s = hs.signals.Signal2D(np.ones((100,100)))
    aipit.level_intensity(s)

def test_recreate_bf_image():
    ss = hs.signals.Signal2D(np.ones((100,100)))
    sn = hs.signals.Signal2D(np.ones((100,100)))
    se = hs.signals.Signal2D(np.ones((100,100)))
    sw = hs.signals.Signal2D(np.ones((100,100)))
    aipit.recreate_bf_image(ss,sn,se,sw)

def test_compute_approximate_pattern():
    bfimage = hs.signals.Signal2D(np.ones((100,100)))
    aipit.compute_approximate_pattern(bfimage)
