"""
Regression test: function values must match tests/data/golden.npz.

The golden file was regenerated after the constraint fixes (tlist/time coordinates, signed distances,
per-slice distance maps, de-duplication, boundary offset). It stores continuous function values rather
than thresholded masks, which flip on rounding wherever f is ~0. To regenerate deliberately after an
intended numerical change: python tests/test_regression.py
"""
from pathlib import Path

import numpy as np
import pytest

from implicit_heart import ImplicitFunction
from implicit_heart.metrics import dice, ejection_fraction

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.npz"


def disk(n, cy, cx, r):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)


def square(n, y0, x0, s):
    a = np.zeros((n, n), np.uint8)
    a[y0:y0 + s, x0:x0 + s] = 1
    return a


def grid_points(shape, t):
    """All pixel centres of an output of the given shape at coordinate t, as evaluate() expects."""
    idx = np.indices(shape).reshape(len(shape), -1)[::-1]  # (x, y[, z])
    return np.column_stack([*idx, np.full(idx.shape[1], t)])


CASES = {
    # 3D: two slices, circle -> square
    "a": lambda: (ImplicitFunction(np.array([disk(40, 15, 15, 8), square(40, 18, 18, 14)]), tmax=8), (40, 40), 4),
    # 3D: three slices with propagation and coarser grid
    "b": lambda: (ImplicitFunction(np.array([disk(40, 15, 15, 6), disk(40, 20, 20, 10), square(40, 14, 16, 12)]),
                                   tmax=2, grid_spacing=0.2, use_propagation=True), (40, 40), 0.5),
    # 4D: two time steps of a 3-slice volume
    "c": lambda: (ImplicitFunction(np.array([np.array([disk(24, 12, 12, r) for r in (4, 6, 5)]),
                                             np.array([disk(24, 11, 13, r) for r in (6, 8, 7)])]),
                                   tmax=1, grid_spacing=0.25), (3, 24, 24), 0.5),
}


def compute(name):
    imp, shape, t = CASES[name]()
    return imp.evaluate(grid_points(shape, t))


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_golden(name):
    golden = np.load(GOLDEN_PATH)
    f = compute(name)
    np.testing.assert_allclose(f, golden[f"{name}_f"], rtol=1e-6, atol=1e-6)


def test_metrics():
    a = disk(20, 10, 10, 5)
    assert dice(a, a) == 1.0
    assert ejection_fraction(np.array([a, np.zeros_like(a)])) == 100.0
    with pytest.raises(Exception):
        ImplicitFunction(np.zeros((1, 8, 8)))  # needs at least 2 slices


if __name__ == "__main__":
    results = {}
    for name in sorted(CASES):
        results[f"{name}_f"] = compute(name)
    np.savez_compressed(GOLDEN_PATH, **results)
    print("wrote", GOLDEN_PATH, {k: v.shape for k, v in results.items()})
