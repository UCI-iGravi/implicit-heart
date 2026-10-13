"""Loader for ACDC challenge data (requires nibabel, in the `notebooks` extra)."""
import glob
import os


def load_frames(folder: str, pattern: str = "*_gt.nii.gz"):
    """
    Loads the two frames of one ACDC patient folder (end-diastole and end-systole, in file-name order).
    :param folder: e.g. data/acdc/training/patient001
    :param pattern: glob for the frame files: ground truth by default; use your own naming for e.g.
                    automatic segmentations stored the same way.
    :return: (frame1, frame2, spacing): label volumes indexed [slice, x, y], and spacing as
             (slice, x, y) in mm, i.e. the `spacing` argument ImplicitFunction expects.
             ACDC labels: 1 = RV, 2 = myocardium, 3 = LV cavity.
    """
    import nibabel as nib
    import numpy as np

    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(paths) < 2:
        raise FileNotFoundError(f"Expected two frames matching {pattern} in {folder}, found {len(paths)}.")
    images = [nib.load(p) for p in paths[:2]]
    # NIfTI arrays and zooms are (x, y, slice); put the slice axis first in both.
    frames = [np.moveaxis(img.get_fdata(), -1, 0) for img in images]
    spacing = tuple(float(s) for s in np.roll(images[0].header.get_zooms()[:3], 1))
    return frames[0], frames[1], spacing
