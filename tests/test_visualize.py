"""Smoke tests for the visualization helpers on tiny synthetic disks."""
import logging

import matplotlib
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
matplotlib.use("Agg")

from implicit_heart import ImplicitFunction
from implicit_heart import visualize as viz

pv.OFF_SCREEN = True
logging.disable(logging.WARNING)


def disk(n, cy, cx, r):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)


@pytest.fixture(scope="module")
def imp4d():
    V = np.array([[disk(24, 12, 12, r)] * 2 for r in (6, 8)])  # (T=2, Z=2, 24, 24)
    return ImplicitFunction(V, tmax=1, grid_spacing=0)


@pytest.fixture(scope="module")
def imp3d():
    return ImplicitFunction(np.array([disk(24, 12, 12, 6)] * 3), tmax=4, grid_spacing=0)


def test_sample_field_shapes(imp4d, imp3d):
    vol, origin, spacing = viz.sample_field(imp4d, t=0, step=2)
    assert vol.shape == (2, 12, 12) and spacing == (2, 2, 1.0)
    vol, origin, spacing = viz.sample_field(imp3d, t=0, step=2, z_samples=3)
    assert vol.shape == (7, 12, 12) and spacing[2] == pytest.approx(4 / 6)  # 2 gaps * 3 samples + 1
    assert vol[:, 6, 6].min() > 0 > vol[:, 0, 0].max()  # inside positive, corner negative


def test_isosurface_matches_disk(imp3d):
    mesh = viz.isosurface(imp3d, step=1, spacing=(5, 1, 1), z_samples=2)  # slice gap = 5 units
    assert mesh.n_points > 0
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    assert xmin == pytest.approx(6, abs=1.5) and xmax == pytest.approx(18, abs=1.5)
    assert ymin == pytest.approx(6, abs=1.5) and ymax == pytest.approx(18, abs=1.5)
    assert zmin == pytest.approx(0, abs=0.5) and zmax == pytest.approx(10, abs=0.5)
    assert mesh.n_open_edges < 0.1 * mesh.n_cells  # capped at the end slices: nearly closed


def test_contour_lines_and_plot_field_slice(imp4d):
    assert viz.contour_lines(imp4d, t=0).n_points > 0
    fig = viz.plot_field_slice(imp4d, t=0, z=1)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_coarse_field_matches_exact_near_the_surface():
    from implicit_heart.visualize import sample_field
    yy, xx = np.mgrid[:40, :40]
    V = np.array([[(yy - 20) ** 2 + (xx - 20) ** 2 <= r ** 2 for r in (8, 12, 9)],
                  [(yy - 18) ** 2 + (xx - 21) ** 2 <= r ** 2 for r in (6, 10, 7)]]).astype(np.uint8)
    imp = ImplicitFunction(V, tmax=1, grid_spacing=0.1)
    exact = sample_field(imp, 0.4, coarse_step=1)[0]
    coarse = sample_field(imp, 0.4)[0]
    assert np.array_equal(exact > 0, coarse > 0)
    near = np.abs(exact) < 1
    np.testing.assert_allclose(coarse[near], exact[near], atol=1e-8)


def test_height_field_explains_an_empty_plane(tmp_path):
    # A shrinking disk: f decreases along t, so far beyond the last slice it is negative everywhere.
    imp = ImplicitFunction(np.array([disk(30, 15, 15, 8), disk(30, 15, 15, 6)]), tmax=1, grid_spacing=0)
    with pytest.raises(ValueError, match="nowhere positive"):
        viz.plot_height_field(imp, t=100, path=str(tmp_path / "empty.png"))  # far beyond the last slice
