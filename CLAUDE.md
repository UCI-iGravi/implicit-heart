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

`tests/data/golden.npz` holds outputs (RBF weights `d` and interpolated masks) produced by the **original pre-refactor code** on synthetic shapes. Any change to `data.py` or `implicit.py` that alters numerics fails these tests. If a behavior change is intended, regenerate the golden file deliberately and say so in the commit; don't loosen tolerances to make it pass.

## Layout

- `src/implicit_heart/`: the package. `implicit.py` (solver), `data.py` (constraint generation), `metrics.py` (dice/jaccard/precision/recall, ejection fraction), `plotting.py`, `sunnybrook/` (DICOM/contour loaders).
- `notebooks/`: three curated examples, run from inside `notebooks/` (paths are `../data/`, `../output/`). Stored without outputs; keep it that way.
- `examples/data/M0.npy`, `M15.npy`: two bundled LV frames `(10, 216, 256)` used by notebook 02.
- `data/`, `output/`, `archive/` are gitignored. `archive/` holds the pre-restructure tree (old notebooks, viz scripts, the abandoned `shape-transformation-4d` fork) for reference only.

## Architecture

Pipeline: binary masks → `Data` builds constraints → `ImplicitFunction` builds and solves the RBF system → `interpolate(t)` samples a grid and thresholds.

**Input shape conventions** (set in `Data.__init__`):
- 3D input `I[n, y, x]`: a stack of 2D slices; slice `n` maps to a third coordinate in `[0, tmax]` (evenly spaced, or from `tlist`). `interpolate(t)` returns one 2D image.
- 4D input `I[t, z, y, x]`: volumes over time; `z` uses raw slice indices, the 4th coordinate is time. `interpolate(t)` returns a 3D volume.
- Constraint points are in pixel coordinates `(x, y, z[, t])`. Normalization (`Data._normalize`) exists but is disabled, which is why grid constraints (pixel-distance values) dominate when `tmax` is small.

**`data.py`: `Data`** generates the constraint set:
- Boundary constraints (value 0) from `cv2.findContours` per slice.
- Inside/outside normal constraints (+1 / −1) offset 0.9 px along the Scharr gradient from each boundary point.
- Optional grid constraints (`grid_spacing` fraction of image size; `0` disables) valued by signed Euclidean distance from `edt`.
- Optional propagation (`use_propagation`): copies neighbouring slices' constraint locations into the current slice with signed-distance values.
- Stored as `(point, value)` tuples in `self.constraints[slice_idx]` (3D) or `self.constraints4D[t][z]` (4D). `get_c()` / `get_h()` flatten them in sorted-key order into the point matrix and right-hand side, and the two must stay in the same order. `metrics.ejection_fraction_from_constraints` also reads `constraints4D` directly.
- `resolution != 1.0` downsamples input; `interpolate` resizes back to `original_shape` in postprocessing.

**`implicit.py`: `ImplicitFunction`**:
- `_create_matrix` builds the dense `(k+1+n)²` system with thin-plate RBF `φ(r) = r² log r` plus affine `P(x) = p0 + p·x`, solved by `scipy.linalg.lu_factor/lu_solve`. Memory O(k²), time O(k³) in constraint count; two ACDC-sized frames give k ≈ 4000.
- `interpolate` calls `_interpolate_point` per pixel in pure Python loops: about 12 ms/point at k ≈ 4000, so a full 216×256×10 volume takes about 2 h. This is the main performance bottleneck.
- Passing a file path instead of an array loads a pickled instance (`save`/`load` pickle the whole object, including the matrix `self.M`).
- Debug/timing output goes through `logging` (`logger.debug`), not prints.

## Known quirks (preserved by the regression tests)

- `Data._create_constraints3D` compares `self.I.shape == 3` (a tuple to an int), so `tlist` is never used for slice z-values there.
- Grid constraints with signed distance exactly 1 get value 0 and are counted as boundary points (this affects `ejection_fraction_from_constraints`).
