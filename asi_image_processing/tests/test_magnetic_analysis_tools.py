import asi_image_processing.magnetic_analysis_tools as aipmat
import hyperspy.api as hs
import numpy as np

def test_get_segmentation_of_dpc_image():
    im1 = np.zeros((100,100))
    im1[10:20, 10:20] = 1
    im1[50:60, 50:60] = 1
    im2 = np.copy(im1)
    img = (im1, im2)
    img = hs.stack(img)
    img.set_signal_type('dpc')
    seg = get_segmentation_of_dpc_image(img)
