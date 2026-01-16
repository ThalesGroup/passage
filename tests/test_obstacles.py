import numpy as np

from passage.obstacles import superellipse_mask


def test_superellipse_n_limits_and_angle_units():
    shape = (101, 101)
    center = (50, 50)

    # n = 2 with a == b should yield a circle-like mask (symmetry)
    a = b = 20.0
    mask_n2 = superellipse_mask(shape, center, a, b, n=2.0, angle=0.0, angle_degrees=True)
    assert mask_n2.sum() > 0

    # For large n the mask area should approach that of an axis-aligned rectangle of size 2a x 2b
    mask_large_n = superellipse_mask(shape, center, a, b, n=100.0, angle=0.0, angle_degrees=True)
    rect_area = (2 * a) * (2 * b)
    # area will be approximate; assert the large-n mask area is within reasonable fraction of rectangle area
    assert abs(mask_large_n.sum() - rect_area) / rect_area < 0.3

    # Angle units: 45 degrees rotation vs pi/4 radians should give similar masks
    mask_deg = superellipse_mask(shape, center, a, b, n=4.0, angle=45.0, angle_degrees=True)
    mask_rad = superellipse_mask(shape, center, a, b, n=4.0, angle=np.pi / 4, angle_degrees=False)
    # Masks should be identical
    assert np.array_equal(mask_deg, mask_rad)
