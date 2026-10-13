"""Validation helpers on synthetic data."""
import logging

import numpy as np
import pytest

from implicit_heart.validation import bland_altman, leave_one_out

logging.disable(logging.WARNING)


def disk(n, r):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - n // 2) ** 2 + (xx - n // 2) ** 2 <= r ** 2).astype(np.uint8)


def test_leave_one_out_beats_nearest_on_a_smooth_stack():
    masks = np.array([disk(48, r) for r in (6, 9, 12, 15, 18)])  # radius grows linearly
    scores = leave_one_out(masks, coords=[0, 4, 8, 12, 16], grid_spacing=0)
    assert sorted(scores) == [1, 2, 3]                            # interior slices only
    interp, nearest = np.array(list(scores.values())).T
    assert interp.min() > 0.9 and interp.mean() > nearest.mean()


def test_leave_one_out_holds_out_frames_in_4d():
    frames = np.array([[disk(40, r), disk(40, r + 2)] for r in (8, 10, 12)])
    scores = leave_one_out(frames, coords=[0, 1, 2], grid_spacing=0)
    assert list(scores) == [1] and scores[1][0] > 0.9


def test_bland_altman():
    result = bland_altman([52, 61, 40, 33], [50, 60, 41, 30])
    assert result["bias"] == pytest.approx(1.25)
    assert result["lower"] < result["bias"] < result["upper"]
    assert result["r"] > 0.95


def test_acdc_loader_puts_slices_first(tmp_path):
    nib = pytest.importorskip("nibabel")
    from implicit_heart.acdc import load_frames

    vol = np.zeros((20, 24, 6))  # NIfTI order (x, y, slice)
    vol[5:10, 8:12, 2] = 3
    for frame in ("01", "12"):
        img = nib.Nifti1Image(vol, np.diag([1.5625, 1.5625, 10, 1]))
        nib.save(img, tmp_path / f"patient001_frame{frame}_gt.nii.gz")
    f1, f2, spacing = load_frames(str(tmp_path))
    assert f1.shape == (6, 20, 24) and spacing == (10.0, 1.5625, 1.5625)
    assert f1[2, 5:10, 8:12].min() == 3
