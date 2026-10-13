"""Validation helpers: held-out interpolation accuracy and method agreement."""
import numpy as np

from .implicit import ImplicitFunction
from .metrics import dice


def leave_one_out(masks: np.ndarray, coords, held=None, **fit_kwargs) -> dict:
    """
    Interpolation accuracy on held-out data: each held-out index is removed, the rest is fitted, and the
    function is evaluated at the removed coordinate.
    :param masks: 3D input masks[slice, y, x] (holds out slices) or 4D input masks[t, slice, y, x] (holds
                  out frames).
    :param coords: coordinate of every index along the held-out axis (slice positions or frame times).
    :param held: indices to hold out; default every interior index (the ends would be extrapolation).
    :param fit_kwargs: passed to ImplicitFunction, e.g. grid_spacing, window, spacing (4D: slice spacing).
    :return: {index: (dice of the interpolation, dice of copying the nearest remaining index)}.
    """
    coords = np.asarray(coords, dtype=float)
    held = range(1, len(masks) - 1) if held is None else held
    scores = {}
    for i in held:
        keep = [j for j in range(len(masks)) if j != i]
        imp = ImplicitFunction(masks[keep], tlist=list(coords[keep]), **fit_kwargs)
        nearest = keep[int(np.argmin(np.abs(coords[keep] - coords[i])))]
        scores[i] = (dice(imp.interpolate(coords[i]), masks[i]), dice(masks[nearest], masks[i]))
    return scores


def bland_altman(estimate, reference) -> dict:
    """
    Agreement between two measurement methods (e.g. ejection fraction from the implicit representation of
    automatic segmentations vs. from ground-truth masks).
    :return: bias (mean difference), lower/upper 95% limits of agreement, and Pearson r.
    """
    estimate, reference = np.asarray(estimate, dtype=float), np.asarray(reference, dtype=float)
    diff = estimate - reference
    bias, sd = diff.mean(), diff.std(ddof=1)
    return {"bias": bias, "lower": bias - 1.96 * sd, "upper": bias + 1.96 * sd,
            "r": np.corrcoef(estimate, reference)[0, 1]}
