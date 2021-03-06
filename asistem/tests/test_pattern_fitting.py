import asi_image_processing.pattern_fitting as aippf
import numpy as np

def test_compute_corners():
    image = np.ones((100,100))
    image[:10 , :  ] = 0
    image[:   , :10] = 0
    image[-10:,   :] = 0
    image[   :,-10:] = 0
    coords = aippf.compute_corners(image)
    assert coords.shape[0] == 4

def test_sort_corners():
    image = np.ones((100,100))
    image[:10 , :  ] = 0
    image[:   , :10] = 0
    image[-10:,   :] = 0
    image[   :,-10:] = 0
    coords = aippf.compute_corners(image)
    sorted_coords = aippf.sort_corners(coords)
    assert sorted_coords.shape[0] == 4
