"""
Regression test: outputs must match tests/data/golden.npz, which was produced by the
original pre-refactor code on the same synthetic shapes.
"""
from pathlib import Path

import numpy as np
import pytest

from implicit_heart import ImplicitFunction
from implicit_heart.metrics import _polygon_area, dice, ejection_fraction

GOLDEN = np.load(Path(__file__).parent / "data" / "golden.npz")


def disk(n, cy, cx, r):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)


def square(n, y0, x0, s):
    a = np.zeros((n, n), np.uint8)
    a[y0:y0 + s, x0:x0 + s] = 1
    return a


def check(key, value):
    np.testing.assert_allclose(np.asarray(value, float), np.asarray(GOLDEN[key], float), atol=1e-8)


def test_3d_two_slices():
    imp = ImplicitFunction(np.array([disk(40, 15, 15, 8), square(40, 18, 18, 14)]), tmax=8)
    check("a_d", imp.d)
    check("a_t0", imp.interpolate(0))
    check("a_t4", imp.interpolate(4))


def test_3d_propagation_and_grid():
    I = np.array([disk(40, 15, 15, 6), disk(40, 20, 20, 10), square(40, 14, 16, 12)])
    imp = ImplicitFunction(I, tmax=2, grid_spacing=0.2, use_propagation=True)
    check("b_d", imp.d)
    check("b_t05", imp.interpolate(0.5))


def test_4d_two_timesteps():
    V0 = np.array([disk(24, 12, 12, r) for r in (4, 6, 5)])
    V1 = np.array([disk(24, 11, 13, r) for r in (6, 8, 7)])
    imp = ImplicitFunction(np.array([V0, V1]), tmax=1, grid_spacing=0.25)
    check("c_d", imp.d)
    check("c_t05", imp.interpolate(0.5))


def test_metrics():
    a = disk(20, 10, 10, 5)
    assert dice(a, a) == 1.0
    assert ejection_fraction(np.array([a, np.zeros_like(a)])) == 100.0
    assert _polygon_area([[0, 0], [2, 0], [2, 2], [0, 2]]) == 4.0
    assert _polygon_area([]) == 0.0
    with pytest.raises(Exception):
        ImplicitFunction(np.zeros((1, 8, 8)))  # needs at least 2 slices
