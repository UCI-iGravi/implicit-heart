"""Constraint-generation correctness: each test pins down one bug fixed after the restructure."""
import logging

import numpy as np
import pytest

from implicit_heart import Data, ImplicitFunction
from implicit_heart.metrics import dice, ejection_fraction, ejection_fraction_polygon

logging.disable(logging.WARNING)


def disk(n, cy, cx, r):
    yy, xx = np.mgrid[:n, :n]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.uint8)


def grid_value(d, idx, x, y):
    """Value of the constraint at integer pixel (x, y) in slice idx of a 3D Data object."""
    return next(v for pt, v in d.constraints[idx] if pt[0] == x and pt[1] == y)


def test_tlist_places_contours_at_given_coordinates():
    I = np.array([disk(40, 20, 20, r) for r in (6, 9, 12)])
    d = Data(I, tlist=[0, 2, 5], grid_spacing=0)
    assert sorted(set(d.get_c()[:, 2])) == [0, 2, 5]


def test_4d_time_is_scaled_to_tmax():
    V = np.array([[disk(24, 12, 12, r)] * 2 for r in (4, 6, 8)])  # (T=3, Z=2, H, W)
    for grid_spacing in (0, 0.25):
        d = Data(V, tmax=1, grid_spacing=grid_spacing)
        assert sorted(set(d.get_c()[:, 3])) == [0, 0.5, 1]


def test_grid_values_are_signed_distance_to_the_boundary():
    # Disk of radius 8 at (20, 20); grid step is 4 px. The boundary lies BOUNDARY_OFFSET (0.25 px)
    # outward from the outermost foreground pixel (x=28), towards the first background pixel (x=29).
    I = np.array([disk(40, 20, 20, 8)] * 2)
    d = Data(I, tmax=1, grid_spacing=0.1)
    assert grid_value(d, 0, 28, 20) == 0.25   # outermost foreground pixel
    assert grid_value(d, 0, 20, 20) == pytest.approx(65 ** 0.5 - 0.75)  # nearest background (24, 27)
    assert grid_value(d, 0, 32, 20) == -3.75  # 4 px from the nearest foreground pixel


def test_grid_distances_use_the_coordinate_metric():
    # 3D distance to the stacked masks (paper Sec. 3.5.2), with slices as far apart in the distance
    # metric as they are in the constraint coordinates. (32, 20) is 8 px outside the small disk in
    # slice 0 but inside the large disk in slice 1.
    I = np.array([disk(40, 20, 20, 4), disk(40, 20, 20, 14)])
    assert grid_value(Data(I, tmax=1, grid_spacing=0.1), 0, 32, 20) == -0.75   # next slice 1 unit away
    assert grid_value(Data(I, tmax=20, grid_spacing=0.1), 0, 32, 20) == -7.75  # in-plane is nearer


def test_empty_slices_get_consistent_distances():
    # An empty slice must not get an arbitrary constant (a fixed -3 made 4D fits fill the volume).
    I = np.array([np.zeros((40, 40), np.uint8), disk(40, 20, 20, 8), disk(40, 20, 20, 10)])
    d = Data(I, tmax=2, grid_spacing=0.1)
    assert grid_value(d, 0, 20, 20) == -0.75   # one slice above the disk centre
    assert grid_value(d, 0, 0, 0) < -10        # far corner


def test_propagated_values_use_the_same_signed_distance():
    I = np.array([disk(40, 20, 20, 6), disk(40, 20, 20, 9), disk(40, 20, 20, 12)])
    d = Data(I, tmax=2, grid_spacing=0, use_propagation=True)
    # Points copied into slice 1 from its neighbours carry slice 1's signed distance at that pixel.
    yy, xx = np.mgrid[:40, :40]
    r = np.hypot(xx - 20, yy - 20)
    own = len([1 for _ in Data(I, tmax=2, grid_spacing=0).constraints[1]])
    for pt, v in d.constraints[1][own:]:
        x, y = round(pt[0]), round(pt[1])
        inside = I[1][y, x] == 1
        assert (v > 0) == inside
        assert v == pytest.approx(d._signed_distance(1, x, y))
    # A pixel 1 px outside the boundary must not be valued as the boundary itself.
    assert d._signed_distance(1, 30, 20) == -0.75


def test_propagation_copies_only_immediate_neighbours():
    I = np.array([disk(40, 20, 20, r) for r in (5, 7, 9, 11, 13)])
    own = [len(Data(I, tmax=4, grid_spacing=0).constraints[i]) for i in range(5)]
    got = [len(Data(I, tmax=4, grid_spacing=0, use_propagation=True).constraints[i]) for i in range(5)]
    assert got == [own[0], *[own[i] + own[i - 1] + own[i + 1] for i in range(1, 4)], own[4]]


@pytest.mark.parametrize("I", [
    np.array([disk(40, 20, 20, 8), disk(40, 20, 20, 12)]),
    np.array([[disk(40, 20, 20, 8)] * 2, [disk(40, 20, 20, 12)] * 2]),
])
def test_no_duplicate_constraint_points(I):
    c = Data(I, tmax=1, grid_spacing=0.1).get_c()
    assert len(np.unique(c, axis=0)) == len(c)


def test_solution_is_well_posed_in_4d():
    V0 = np.array([disk(24, 12, 12, r) for r in (4, 6, 5)])
    V1 = np.array([disk(24, 11, 13, r) for r in (6, 8, 7)])
    imp = ImplicitFunction(np.array([V0, V1]), tmax=1, grid_spacing=0.25)
    M = imp._create_matrix(imp.c * imp._scale)
    assert np.linalg.cond(M) < 1e10  # duplicate points made M singular (cond ~1e17, d ~ 1e13)


def test_reconstruction_matches_input_slices():
    I = np.array([disk(40, 20, 20, 8), disk(40, 20, 20, 12)])
    imp = ImplicitFunction(I, tmax=1, grid_spacing=0.1)
    for t, mask in [(0, I[0]), (1, I[1])]:
        out = imp.interpolate(t, postprocess=False)
        assert dice(out, mask) > 0.97
        assert abs(out.sum() / mask.sum() - 1) < 0.08
    # Contour pixels must be decided by the function, not by rounding noise at f = 0 (the original
    # placed boundary constraints exactly on them).
    yy, xx = np.mgrid[:40, :40]
    f = imp.evaluate(np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])).reshape(40, 40)
    assert f[20, 28] > 0.1 and f[20, 29] < -0.1


def test_resolution_keeps_masks_and_returns_original_size():
    I = np.array([disk(40, 20, 20, 10), disk(40, 20, 20, 14)])
    imp = ImplicitFunction(I, tmax=1, grid_spacing=0, resolution=0.5)
    for postprocess in (False, True):
        out = imp.interpolate(0, postprocess=postprocess)
        assert out.shape == (40, 40)
        assert set(np.unique(out)) <= {0, 1}
        assert dice(out, I[0]) > 0.9


def test_polygon_ejection_fraction_with_several_components():
    def frame(r):
        return np.array([disk(60, 15, 15, r) | disk(60, 45, 45, r)] * 2)
    volumes = np.array([frame(12), frame(8)])
    assert ejection_fraction_polygon(volumes) == pytest.approx(ejection_fraction(volumes), abs=5)


def ring(n, c, r_out, r_in):
    return (disk(n, c, c, r_out) - disk(n, c, c, r_in)).astype(np.uint8)


def test_holes_are_kept():
    # e.g. the myocardium, a ring around the LV cavity: inner contours must become constraints too
    I = np.array([ring(50, 25, 18, 10), ring(50, 25, 16, 8)])
    imp = ImplicitFunction(I, tmax=1, grid_spacing=0)
    out = imp.interpolate(0, postprocess=False)
    assert out[25, 25] == 0
    assert dice(out, I[0]) > 0.95


def test_polygon_ejection_fraction_subtracts_holes():
    volumes = np.array([[ring(50, 25, 18, 10)], [ring(50, 25, 16, 8)]])
    assert ejection_fraction_polygon(volumes) == pytest.approx(ejection_fraction(volumes), abs=3)


def test_empty_end_slices_are_dropped_and_come_back_empty():
    # An empty slice next to a large contour used to get a spurious blob (the end cap overshoots).
    big, small = disk(40, 20, 20, 12), disk(40, 20, 20, 9)
    zero = np.zeros_like(big)
    imp = ImplicitFunction(np.array([[zero, big, big], [zero, small, small]]), tmax=1, grid_spacing=0.1)
    assert sorted(set(imp.c[:, 2])) == [1, 2]          # original slice coordinates are kept
    out = imp.interpolate(0)
    assert out.shape == (3, 40, 40) and out[0].sum() == 0
    assert dice(out[1:], np.array([big, big])) > 0.97
    # 3D slice stack: the dropped slice's position comes back empty too
    imp = ImplicitFunction(np.array([zero, big, small]), tmax=2, grid_spacing=0)
    assert sorted(set(imp.c[:, 2])) == [1, 2]
    assert imp.interpolate(0).sum() == 0 and imp.interpolate(0.5).sum() == 0
    assert dice(imp.interpolate(1), big) > 0.97


def test_postprocessing_is_off_by_default():
    small = disk(40, 20, 20, 4)  # 49 px: remove_small_objects (64 px) used to delete it
    imp = ImplicitFunction(np.array([small, small]), tmax=1, grid_spacing=0)
    assert imp.interpolate(0).sum() > 40


def test_spacing_places_slices_at_physical_distances():
    V = np.array([[disk(40, 20, 20, r) for r in (8, 10)], [disk(40, 20, 20, r) for r in (6, 8)]])
    imp = ImplicitFunction(V, tmax=1, grid_spacing=0, spacing=(10, 2, 2))
    assert sorted(set(imp.c[:, 2])) == [0, 5]          # 10 mm slices / 2 mm pixels
    assert dice(imp.interpolate(0), V[0]) > 0.97
    imp = ImplicitFunction(V[0], grid_spacing=0, spacing=(10, 2, 2))
    assert sorted(set(imp.c[:, 2])) == [0, 5] and imp.tmax == 5
    with pytest.raises(ValueError):
        ImplicitFunction(V, spacing=(10, 2, 3))     # in-plane pixels must be square


def test_surface_ejection_fraction_matches_masks():
    from implicit_heart.metrics import ejection_fraction_surface
    V = np.array([[disk(40, 20, 20, r) for r in (10, 12, 10)], [disk(40, 20, 20, r) for r in (8, 9, 8)]])
    imp = ImplicitFunction(V, tmax=1, grid_spacing=0)
    assert ejection_fraction_surface(imp) == pytest.approx(ejection_fraction(V), abs=3)


@pytest.mark.parametrize("I, kw", [
    (np.array([ring(50, 25, 18, 10), ring(50, 25, 16, 8)]), dict(tmax=1, grid_spacing=0.1)),
    (np.array([[disk(40, 20, 20, r) for r in (6, 10, 8)], [disk(40, 18, 22, r) for r in (4, 8, 6)]]),
     dict(tmax=1, grid_spacing=0.25)),
])
def test_coarse_to_fine_matches_exact(I, kw):
    imp = ImplicitFunction(I, **kw)
    for t in (0, 0.3, 1):
        assert np.array_equal(imp.interpolate(t), imp.interpolate(t, coarse_step=1))


def stack(radii, n=48):
    return np.array([disk(n, n // 2, n // 2 + i % 3, r) for i, r in enumerate(radii)])


RADII = [6, 8, 10, 12, 13, 14, 14, 13, 12, 10, 9, 7]


def test_windowed_fit_matches_global():
    keys = list(range(0, len(RADII), 2))
    I = stack(RADII)
    g = ImplicitFunction(I[keys], tlist=[2.0 * k for k in keys], grid_spacing=0)
    w = ImplicitFunction(I[keys], tlist=[2.0 * k for k in keys], grid_spacing=0, window=4)  # overlap 2
    assert len(w.windows) == 2 and max(len(win[2]) for win in w.windows) < len(w.c)
    ours, ref = [], []
    for z in range(len(RADII) - 1):                     # keyframes and held-out slices
        out, glob = w.interpolate(2.0 * z), g.interpolate(2.0 * z)
        assert dice(out, glob) > 0.98
        ours.append(dice(out, I[z]))
        ref.append(dice(glob, I[z]))
    assert np.mean(ours) > np.mean(ref) - 0.005         # as accurate as the global fit
    assert np.array_equal(w.interpolate(5.0), w.interpolate(5.0, coarse_step=1))


def test_windowed_fit_falls_back_to_global_when_window_covers_everything():
    I = stack(RADII[:4])
    assert ImplicitFunction(I, grid_spacing=0, window=4).windows is None


def test_windowed_fit_4d_runs_along_slices():
    V = np.array([stack(RADII[:8]), stack([r - 2 for r in RADII[:8]])])
    w = ImplicitFunction(V, tmax=1, grid_spacing=0, window=4, overlap=2)
    assert len(w.windows) == 3
    for t, frame in ((0, V[0]), (1, V[1])):
        assert dice(w.interpolate(t), frame) > 0.97
