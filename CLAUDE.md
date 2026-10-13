# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`implicit_heart`: an implementation of Turk & O'Brien's *Shape Transformation Using Variational Implicit Functions* applied to cardiac segmentation masks (ACDC, MM-WHS, Sunnybrook). Given binary mask slices/volumes, it fits an RBF implicit function and samples it at intermediate `t` to interpolate between slices (3D input) or between time frames (4D input). Prepared as a public release (BSD-3-Clause).

## Commands

```bash
pip install -e ".[dev]"            # extras: viz (pyvista), notebooks (jupyter, nibabel), sunnybrook (pandas, pydicom)
pytest                             # all tests (~3s)
pytest tests/test_regression.py::test_4d_two_timesteps   # single test
python scripts/render_volume.py <vol.npy> [--spacing Z Y X] [--labels] [--clip AXIS ORIGIN] [--gif out.gif [--orbit]]
```

On this machine use `.venv/Scripts/python` (Python 3.12). The system Python 3.14 has no `edt` wheel.

## Regression contract

`tests/data/golden.npz` holds continuous function values f on synthetic shapes (masks are a brittle reference: they flip on rounding wherever f is about 0). Any change to `data.py` or `implicit.py` that alters numerics fails these tests. If a behavior change is intended, regenerate with `python tests/test_regression.py` and say so in the commit; don't loosen tolerances to make it pass. `tests/test_constraints.py` pins the constraint conventions below, one test per fixed bug.

## Layout

- `src/implicit_heart/`: the package. `implicit.py` (solver), `data.py` (constraint generation), `metrics.py` (dice/jaccard/precision/recall, ejection fraction), `validation.py` (leave-one-out interpolation accuracy, Bland–Altman), `visualize.py` (isosurfaces, field slices, height fields; PyVista imported lazily), `acdc.py` (ACDC NIfTI loader: slice axis first, spacing as (slice, x, y)), `plotting.py`, `sunnybrook/` (DICOM/contour loaders).
- CI: `.github/workflows/tests.yml` runs pytest on Python 3.10 and 3.12 with a headless display for the PyVista tests.
- `notebooks/`: three curated examples, run from inside `notebooks/` (paths are `../data/`, `../output/`). Stored without outputs; keep it that way.
- `examples/data/M0.npy`, `M15.npy`: two bundled LV frames `(10, 216, 256)` used by notebook 02.
- `data/`, `output/`, `archive/` are gitignored. `archive/` holds the pre-restructure tree (old notebooks, viz scripts, the abandoned `shape-transformation-4d` fork) for reference only.

## Architecture

Pipeline: binary masks → `Data` builds constraints → `ImplicitFunction` builds and solves the RBF system → `interpolate(t)` samples a grid and thresholds.

**Input shape conventions** (set in `Data.__init__`):
- 3D input `I[n, y, x]`: a stack of 2D slices; slice `n` maps to a third coordinate in `[0, tmax]` (evenly spaced, or from `tlist`). `interpolate(t)` returns one 2D image.
- 4D input `I[t, z, y, x]`: volumes over time; slice z sits at `imp.slice_coords[z]` (its index, times `spacing_z / spacing_x` if `spacing` is given), and the 4th coordinate is time. `interpolate(t)` returns a 3D volume.
- `ImplicitFunction` crops leading/trailing slices that are empty in every frame before building `Data` (`imp.z_range` = fitted slice indices), so `imp.data.I` and `imp.dim` are the cropped shape. Coordinates stay those of the original slices, and `interpolate` pads the dropped slices back as empty. `Data(slice_coords=...)` carries the 4D slice coordinates.
- Constraint points are in pixel coordinates `(x, y, z[, t])`; the slice/time coordinate comes from `Data._coord` (tlist or evenly spaced on `[0, tmax]`). In 4D, z is the raw slice index (physical slice spacing is ignored).

**`data.py`: `Data`** generates the constraint set:
- Boundary constraints (value 0) from `cv2.findContours` per slice (`RETR_CCOMP`: outer and hole contours; `RETR_EXTERNAL` used to fill holes such as the myocardium ring), shifted `BOUNDARY_OFFSET` (0.25 px, chosen empirically) outward along the Scharr gradient. Exactly on the contour pixel centres (the original) put f = 0 on pixels, so their classification was rounding noise and shapes eroded.
- Inside/outside normal constraints (+1 / −1) 0.9 px either side of each boundary point.
- `Data._signed_distance` is the single signed-distance convention used by grid and propagated constraints: a 3D EDT over each volume (paper Sec. 3.5.2), with the slice axis scaled like the constraint coordinates (`_coord` spacing for 3D input, 1 per slice in 4D), measured from the same offset boundary. A per-slice 2D EDT was tried and broke 4D fits: empty slices had no distance to measure.
- Optional grid constraints (`grid_spacing` fraction of image size; `0` disables).
- Optional propagation (`use_propagation`): copies each neighbouring slice's *own* constraint locations into the current slice with signed-distance values.
- Stored as `(point, value)` tuples in `self.constraints[slice_idx]` (3D) or `self.constraints4D[t][z]` (4D). `_flatten()` (behind `get_c()` / `get_h()`) flattens them in sorted-key order and drops duplicate points, keeping the first; duplicates make M singular.
- `resolution != 1.0` downsamples input (nearest neighbour); `ImplicitFunction.dim` is the downsampled shape and `interpolate` resizes back to `original_shape`.

**`implicit.py`: `ImplicitFunction`**:
- `_create_matrix` builds the dense `(k+1+n)²` system with thin-plate RBF `φ(r) = r² log r` plus affine `P(x) = p0 + p·x`, solved by `scipy.linalg.lu_factor/lu_solve`. Memory O(k²), time O(k³) in constraint count; two ACDC-sized frames give k ≈ 2000–4000.
- The solve and `evaluate` work in coordinates scaled by `_scale = 1/max(H, W)`: the thin-plate interpolant is invariant to uniform scaling, and conditioning improves by about 5 orders of magnitude. `imp.c` stays in pixels; `imp.d` and `imp.p` are in scaled units.
- Windowed fits (`window=`): `imp.windows` holds `(lo, hi, constraint indices, d, p)` per run of slices, split on constraint column 2 (the slice axis in both 3D and 4D). `evaluate` blends the windows with weights (1 − u²)² across each window, without fading at the stack ends. `imp.d`/`imp.p` are None then.
- `evaluate(points)` is the vectorised function: cdist in chunks on up to 8 threads (NumPy releases the GIL). `_phi` (r² log r) dominates its cost.
- `interpolate(t)` thresholds via `_threshold_grid`: f on a `coarse_step` grid, exact evaluation only in cells whose corners are within `coarse_step` of the threshold. `tests/test_constraints.py` checks it equals `coarse_step=1`. About 1 s per 216×256×10 volume.
- Passing a file path instead of an array loads a pickled instance (`save`/`load` pickle the whole object). The k×k matrix is factored in place and not kept, and `evaluate` sizes its chunks to about 64 MB per thread, so memory stays about 2 × 8k² bytes during the fit and flat afterwards.
- Debug/timing output goes through `logging` (`logger.debug`), not prints.

## Known limitations

- `metrics.ejection_fraction_polygon` uses contour polygons through pixel centres, so it underestimates areas (19.50% vs 18.92% on M0/M15). Prefer `ejection_fraction_surface`, which gives 18.89%.
- Temporal interpolation between keyframes is not validated against held-out frames anywhere in the repo.
