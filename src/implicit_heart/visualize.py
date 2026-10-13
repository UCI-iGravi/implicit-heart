"""
Visualization of fitted implicit functions: isosurfaces, field slices and height fields.

PyVista (the `viz` extra) is imported lazily inside the functions that render 3D scenes.

Coordinates: x, y are pixels of the (possibly resized) masks in `imp.data.I`. The slice axis is the
function's third coordinate, `imp.slice_coords[i]` for input slice i (for 3D input the slice's
coordinate in [0, tmax]; for 4D input its index, scaled if the fit was given a spacing). Only slices in
`imp.z_range` were fitted. `spacing=(z, y, x)` is the physical voxel size per pixel and per slice gap, so
surfaces come out with true proportions.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

DIVERGING = "RdBu_r"


def _is_4d(imp):
    return len(imp.dim) == 4


def _evaluate_grid(imp, xs, ys, zs, t, coarse_step=4):
    """
    f on the grid xs x ys x zs (4D input: at time t), as a (len(zs), len(ys), len(xs)) array; exact near
    f = 0 (see ImplicitFunction.sample).
    """
    if _is_4d(imp):
        return imp.sample(list(zs), t, ys, xs, coarse_step)
    # 3D input: each z is a plane t = z of the function (x, y, t).
    return np.concatenate([imp.sample([None], z, ys, xs, coarse_step) for z in zs])


def _z_scale(imp, spacing):
    """Physical units per unit of the slice coordinate (one slice gap = spacing[0])."""
    return spacing[0] / np.mean(np.diff(imp.slice_coords))


def sample_field(imp, t, step=1, z_samples=None, coarse_step=4):
    """
    Sample f on a regular grid with one batched evaluate() call.
    :param imp: a fitted ImplicitFunction.
    :param t: time coordinate in [0, tmax] (4D input only; ignored for a 3D slice stack).
    :param step: pixel stride in x and y.
    :param z_samples: samples per slice gap along the slice axis (default 4 for a 3D slice stack so the
                      surface through all slices can be extracted, 1 for 4D input: the slices themselves).
    :param coarse_step: f is exact near f = 0 and upsampled from every coarse_step-th grid point elsewhere;
                        1 evaluates every grid point.
    :return: (volume, origin, spacing): volume is (Z, Y, X); origin and spacing are (x, y, z) in
             pixels (4D) or in the coordinate of the slice axis (3D).
    """
    h, w = imp.dim[-2:]
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)
    first, last = imp.z_range  # fitted slices
    n = last - first + 1
    z_samples = z_samples or (1 if _is_4d(imp) else 4)
    zs = np.linspace(imp.slice_coords[first], imp.slice_coords[last], max(n - 1, 1) * z_samples + 1)
    dz = zs[1] - zs[0] if len(zs) > 1 else 1.0
    return _evaluate_grid(imp, xs, ys, zs, t, coarse_step), (0.0, 0.0, float(zs[0])), (step, step, float(dz))


def isosurface(imp, t=0, step=1, spacing=(1, 1, 1), z_samples=None, coarse_step=4):
    """
    The f = 0 surface as a smoothed pv.PolyData.
    :param spacing: physical voxel size (z, y, x); z is one slice gap.
    """
    import pyvista as pv

    vol, origin, dxyz = sample_field(imp, t, step, z_samples, coarse_step)
    sz = _z_scale(imp, spacing)
    # Pad the slice axis with a very negative layer so the surface is capped flat at the end slices.
    vol = np.pad(vol, ((1, 1), (0, 0), (0, 0)), constant_values=-1e6)
    grid = pv.ImageData(dimensions=vol.shape[::-1],
                        spacing=(dxyz[0] * spacing[2], dxyz[1] * spacing[1], dxyz[2] * sz),
                        origin=(0, 0, (origin[2] - dxyz[2]) * sz))
    grid.point_data["f"] = vol.ravel()
    mesh = grid.contour([0.0], scalars="f")
    if mesh.n_points == 0:
        raise ValueError("f has no zero crossing on the sampled grid")
    return mesh.smooth_taubin(n_iter=20, pass_band=0.1).compute_normals(auto_orient_normals=False)


def contour_lines(imp, t=0, spacing=(1, 1, 1)):
    """
    Polylines of the input masks' contours at each slice's z (4D input: the frame nearest t).
    :param spacing: physical voxel size (z, y, x).
    """
    import pyvista as pv

    masks = imp.data.I  # fitted slices only
    if _is_4d(imp):
        frame = int(np.argmin([abs(imp.data._coord(i) - t) for i in range(imp.dim[0])]))
        masks = masks[frame]
    z_scale = _z_scale(imp, spacing)
    points, lines = [], []
    for i, mask in enumerate(masks):
        z = imp.slice_coords[imp.z_range[0] + i]
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            xy = c[:, 0, :].astype(float)
            ring = np.column_stack([xy[:, 0] * spacing[2], xy[:, 1] * spacing[1],
                                    np.full(len(xy), z * z_scale)])
            lines.append(np.r_[len(ring) + 1, len(points) + np.arange(len(ring)), len(points)])
            points.extend(ring)
    if not lines:
        raise ValueError("The fitted masks have no contours to draw.")
    return pv.PolyData(np.array(points), lines=np.concatenate(lines))


def _add_surface(pl, mesh, opacity=1.0, name="surface"):
    pl.add_mesh(mesh, color="#c8434f", smooth_shading=True, specular=0.6, specular_power=30,
                diffuse=0.8, ambient=0.15, opacity=opacity, name=name)


def render_surface(mesh, lines=None, path=None, orbit=False, title=None):
    """
    Render a surface (and optional contour polylines) on a dark background.
    :param path: None to show interactively; a .png for a screenshot; a .gif for an orbit animation
                 (or a single frame if orbit is False).
    :param orbit: with a .gif path, rotate the camera 360 degrees.
    """
    import pyvista as pv

    pl = pv.Plotter(off_screen=path is not None, window_size=(1000, 800))
    pl.set_background("black", top="#2b2d42")
    _add_surface(pl, mesh, opacity=0.6 if lines is not None else 1.0)  # see the contours through it
    if lines is not None:
        pl.enable_depth_peeling()
        pl.add_mesh(lines.tube(radius=0.5 * lines.length / 150), color="#ffd23f")
    if title:
        pl.add_text(title, position="upper_edge", color="white", font_size=14)
    pl.camera_position = "iso"
    if path is None:
        pl.show()
    elif str(path).endswith(".gif"):
        pl.open_gif(str(path))
        pl.show(auto_close=False)
        for _ in range(36 if orbit else 1):
            pl.camera.Azimuth(10)
            pl.write_frame()
        pl.close()
    else:
        pl.show(screenshot=str(path))


def animate_surfaces(imp, ts, path, step=2, spacing=(1, 1, 1), z_samples=None):
    """GIF of the f = 0 isosurface of a 4D function at each t in ts, with a fixed camera."""
    import pyvista as pv

    if not _is_4d(imp):
        raise ValueError("animate_surfaces needs 4D input")
    meshes = [isosurface(imp, t, step, spacing, z_samples) for t in ts]
    bounds = np.array([m.bounds for m in meshes])
    union = (bounds[:, 0].min(), bounds[:, 1].max(), bounds[:, 2].min(), bounds[:, 3].max(),
             bounds[:, 4].min(), bounds[:, 5].max())
    pl = pv.Plotter(off_screen=True, window_size=(800, 800))
    pl.set_background("black", top="#2b2d42")
    pl.add_mesh(pv.Box(bounds=union), opacity=0.0)  # fixes the camera framing for all frames
    pl.camera_position = "iso"
    pl.open_gif(str(path))
    for t, mesh in zip(ts, meshes):
        _add_surface(pl, mesh)
        pl.add_text(f"t = {t:.2f}", position="upper_edge", color="white", font_size=14, name="label")
        pl.write_frame()
    pl.close()


def _plane(imp, t, z, step):
    """f on one 2D plane as (xs, ys, F) plus the constraint points (c, h) lying on that plane."""
    h, w = imp.data.I.shape[-2:]
    xs, ys = np.arange(0, w, step), np.arange(0, h, step)
    c, hv = imp.c, imp.data.get_h()[:, 0]
    if _is_4d(imp):
        z = sum(imp.z_range) // 2 if z is None else z
        zc = imp.slice_coords[z]
        on = np.isclose(c[:, 2], zc) & np.isclose(c[:, 3], t)
        F = _evaluate_grid(imp, xs, ys, [zc], t)[0]
    else:
        on = np.isclose(c[:, 2], t)
        F = _evaluate_grid(imp, xs, ys, [t], t)[0]
    return xs, ys, F, c[on], hv[on]


def plot_field_slice(imp, t=0, z=None, ax=None, step=1):
    """
    f on one 2D plane: diverging colormap centred at 0, thick f = 0 contour, and the constraint
    points on that plane by type. The debugging view.
    :param t: 3D input: coordinate of the plane in [0, tmax]. 4D input: time coordinate.
    :param z: 4D input: slice index (default: middle fitted slice).
    :return: the matplotlib Figure.
    """
    xs, ys, F, c, h = _plane(imp, t, z, step)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    vmax = max(F.max(), 1e-9)  # symmetric limits: the far field would otherwise wash out the shape
    norm = TwoSlopeNorm(0, vmin=-vmax, vmax=vmax)
    im = ax.pcolormesh(xs, ys, F, cmap=DIVERGING, norm=norm, shading="nearest")
    ax.contour(xs, ys, F, [0], colors="k", linewidths=3)
    for name, sel, style in [("boundary (0)", h == 0, dict(c="k", marker="o", s=14)),
                             ("inside (+1)", h == 1, dict(c="#2ca02c", marker="+", s=40)),
                             ("outside (-1)", h == -1, dict(c="#ff7f0e", marker="x", s=30)),
                             ("grid / other", (h != 0) & (abs(h) != 1), dict(c="#9467bd", marker="s", s=14))]:
        if sel.any():
            ax.scatter(c[sel, 0], c[sel, 1], label=f"{name}  n={sel.sum()}", linewidths=1, **style)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if len(c):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.figure.colorbar(im, ax=ax, label="f  (> 0 inside)")
    ax.set_title(f"f on plane t={t}" + (f", slice z={z if z is not None else sum(imp.z_range) // 2}" if _is_4d(imp) else ""))
    return ax.figure


def plot_height_field(imp, t=0, z=None, path=None, step=1):
    """
    A 2D slice of f drawn as a 3D height field (cf. Figs. 2 and 4 of Turk & O'Brien): the surface is
    coloured by f, the f = 0 level is a red tube, and a translucent plane marks z = 0.
    :param path: None to show interactively, else a .png screenshot path.
    """
    import pyvista as pv

    xs, ys, F, _, _ = _plane(imp, t, z, step)
    # Crop to the neighbourhood of the shape; the far field is nearly linear and would flatten the bump.
    rows, cols = np.where(F > 0)
    if not len(rows):
        raise ValueError(f"f is nowhere positive on the plane t={t}, z={z}: no zero level to show.")
    pad = int(1.0 * max(np.ptp(rows), np.ptp(cols))) + 1
    r0, r1 = max(rows.min() - pad, 0), rows.max() + pad + 1
    c0, c1 = max(cols.min() - pad, 0), cols.max() + pad + 1
    xs, ys, F = xs[c0:c1], ys[r0:r1], F[r0:r1, c0:c1]
    ref = max(F.max(), 1e-9)
    F = np.clip(F, -ref, ref)  # symmetric colour range; the height scale maps ref to a third of the width
    scale = (xs.max() - xs.min()) / 3 / ref
    xx, yy = np.meshgrid(xs.astype(float), ys.astype(float))
    grid = pv.StructuredGrid(xx, yy, F * scale)
    grid.point_data["f"] = F.ravel(order="F")
    pl = pv.Plotter(off_screen=path is not None, window_size=(1000, 800))
    pl.set_background("white")
    pl.enable_depth_peeling()
    pl.add_mesh(grid, scalars="f", cmap=DIVERGING, clim=(-ref, ref), smooth_shading=True, specular=0.4,
                scalar_bar_args=dict(title="f", color="black"))
    zero = grid.contour([0.0], scalars="f")
    if zero.n_points:
        pl.add_mesh(zero.tube(radius=0.008 * (xs.max() - xs.min())), color="red")
    pl.add_mesh(pv.Plane(center=(xs.mean(), ys.mean(), 0), i_size=xs.max() - xs.min(),
                         j_size=ys.max() - ys.min()), color="gray", opacity=0.3)
    pl.camera_position = "iso"
    pl.camera.Azimuth(30)
    if path is None:
        pl.show()
    else:
        pl.show(screenshot=str(path))
