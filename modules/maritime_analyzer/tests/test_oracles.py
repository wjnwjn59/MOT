import numpy as np
from modules.maritime_analyzer import oracles as o


def test_laplacian_variance_zero_for_uniform():
    assert o.laplacian_variance(np.full((20, 20), 128.0, dtype=np.float32)) == 0.0


def test_is_motion_blur_true_for_uniform_false_for_checkerboard():
    uniform = np.full((20, 20), 128.0, dtype=np.float32)
    checker = np.indices((20, 20)).sum(axis=0) % 2 * 255.0
    assert o.is_motion_blur(uniform, (0, 0, 20, 20), threshold=100.0) is True
    assert o.is_motion_blur(checker.astype(np.float32), (0, 0, 20, 20), threshold=100.0) is False


def test_is_out_of_frame():
    assert o.is_out_of_frame((10, 10, 20, 20), (100, 100), margin=1) is False
    assert o.is_out_of_frame((0, 10, 20, 20), (100, 100), margin=1) is True   # touches left edge
    assert o.is_out_of_frame((85, 10, 20, 20), (100, 100), margin=1) is True  # x+w exceeds frame
    # exact right-edge boundary: out when x+w >= W-margin (99), inside one pixel before
    assert o.is_out_of_frame((79, 10, 20, 20), (100, 100), margin=1) is True   # x+w == 99
    assert o.is_out_of_frame((78, 10, 20, 20), (100, 100), margin=1) is False  # x+w == 98
    # non-square frame: frame_size is (W, H); bottom edge uses H
    assert o.is_out_of_frame((100, 462, 20, 20), (640, 480), margin=1) is True   # y+h=482 >= 479
    assert o.is_out_of_frame((100, 100, 20, 20), (640, 480), margin=1) is False


def test_highlight_ratio():
    bright = np.full((10, 10), 255.0, dtype=np.float32)
    dark = np.zeros((10, 10), dtype=np.float32)
    assert o.highlight_ratio(bright, (0, 0, 10, 10)) == 1.0
    assert o.highlight_ratio(dark, (0, 0, 10, 10)) == 0.0


def test_compute_oracle_attributes_keys():
    img = np.full((100, 100), 128.0, dtype=np.float32)
    out = o.compute_oracle_attributes(img, img, (10, 10, 20, 20), (10, 10, 20, 20), (100, 100))
    for k in ["scale_variation", "low_resolution", "low_contrast", "motion_blur", "out_of_frame"]:
        assert k in out and out[k] in (0, 1)
    assert "highlight_ratio" in out["_features"]
    assert "scale_ratio" in out["_features"]
