import asi_image_processing.pattern_finding as aippf
import numpy as np

def test_compute_corners():
    image = np.ones((100,100))
    image[:10 , :  ] = 0
    image[:   , :10] = 0
    image[-10:,   :] = 0
    image[   :,-10:] = 0
    coords = aippf.compute_corners(image)
    print(coords)
