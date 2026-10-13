# implicit-heart

[![tests](https://github.com/UCI-iGravi/implicit-heart/actions/workflows/tests.yml/badge.svg)](https://github.com/UCI-iGravi/implicit-heart/actions/workflows/tests.yml)

Shape interpolation of cardiac segmentation masks using variational implicit functions, after
Turk & O'Brien, [*Shape Transformation Using Variational Implicit Functions*](https://doi.org/10.1145/311535.311580) (SIGGRAPH 1999).

Given binary masks, `implicit-heart` fits a thin-plate radial basis function through boundary,
inside/outside and signed-distance constraints, then samples it at any intermediate coordinate `t`:

- **3D**: a stack of 2D slices `I[slice, y, x]` → a 2D mask at any position between slices.
- **4D**: a sequence of volumes `I[time, slice, y, x]` → a full 3D volume at any point in time.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/UCI-iGravi/implicit-heart.git
cd implicit-heart
pip install -e .
```

Optional extras: `.[viz]` (PyVista rendering script), `.[notebooks]` (Jupyter, nibabel),
`.[sunnybrook]` (DICOM loaders), `.[dev]` (pytest).

## Quickstart

```python
import numpy as np
from implicit_heart import ImplicitFunction

yy, xx = np.mgrid[:40, :40]
small = (yy - 20) ** 2 + (xx - 20) ** 2 <= 6 ** 2
large = (yy - 20) ** 2 + (xx - 20) ** 2 <= 12 ** 2

imp = ImplicitFunction(np.array([small, large]).astype(np.uint8), tmax=10)
halfway = imp.interpolate(5)  # binary (40, 40) mask between the two shapes
```

Main `ImplicitFunction` arguments:

| Argument | Meaning |
|---|---|
| `tmax` | Coordinate of the last slice/frame; inputs are spaced evenly on `[0, tmax]`. |
| `tlist` | Explicit coordinate per slice/frame (overrides `tmax`). |
| `grid_spacing` | Extra signed-distance constraints on a grid, as a fraction of image size. `0` disables (Turk & O'Brien's original method). |
| `resolution` | Downsampling factor applied before fitting; output is resized back. |
| `use_propagation` | Copy neighbouring slices' constraint locations into each slice. |
| `spacing` | Physical voxel size `(z, y, x)` with square pixels. Slices are placed `z / x` pixel units apart, so the fit is isotropic in physical space (for 3D input this replaces `tmax`). |
| `window`, `overlap` | Fit separate functions on runs of `window` consecutive slices sharing `overlap` slices (default `window // 2`), blended along the slice axis. Use for many slices: cost grows linearly instead of cubically. |

Leading and trailing slices that are empty in every frame are left out of the fit (the surface would
overshoot into them) and come back empty.

`interpolate(t, thresh=0.0, postprocess=False, kernel_size=3, coarse_step=4)` returns the mask at `t`.
`postprocess=True` applies a median filter and removes small objects and holes. Beware that it deletes
components under 64 px, e.g. small apical slices. `evaluate(points)` gives the continuous function.
Set `logging.basicConfig(level=logging.DEBUG)` for timing output.

**Performance:** the linear system is dense, so memory grows with the square of the number of
constraints k and solve time with the cube (k ≈ 1500–4000 for two 216×256×10 frames: about 1 s;
k ≈ 19k for 35 sections of a 436×562 brain mask: about 1 min and 3 GB). For many slices use `window=4`
to `6`: on that brain it matched the global fit's accuracy (held-out dice 0.987) with 60–130 MB per
window and 2.5–4× less time.
`interpolate` evaluates f on a coarse grid and exactly only near the surface (`coarse_step=1` evaluates
every pixel). A 216×256×10 volume takes about 1 s.

## Notebooks

Run from the `notebooks/` folder; paths are relative to it.

| Notebook | Data needed |
|---|---|
| `01_3d_interpolation.ipynb` | `data/npy/lv.npy` (binary LV masks, `(T, Z, H, W)`) |
| `02_4d_interpolation.ipynb` | Bundled `examples/data/M0.npy`, `M15.npy` (two LV frames); last section uses `data/npy/lv.npy` |
| `03_acdc_ejection_fraction.ipynb` | [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) training set at `data/acdc/training/` |
| `04_validation.ipynb` | Held-out slices: bundled data. Held-out frames: `data/npy/lv.npy`. EF agreement: ACDC plus your automatic segmentations at `data/acdc_auto/` |

Datasets are not redistributed here. Download them from their providers and place them under `data/`
(ignored by git). Generated results go to `output/`.

## Rendering volumes

```bash
pip install -e ".[viz]"
python scripts/render_volume.py output/M0-M15-interp.npy --spacing 10 1.5625 1.5625
python scripts/render_volume.py "output/anim/*.npy" --gif anim.gif          # one frame per file
python scripts/render_volume.py all_labels.npy --labels --clip y 98 --gif orbit.gif --orbit
```

## Visualizing implicit surfaces

`scripts/visualize_implicit.py` fits the function to one slice stack (one `.npy`) or two frames (two `.npy`,
stacked as time) and renders it; omit `--out` for an interactive window. Sampling costs about 0.07 ms per
point, so use `--step 2` on full-size images.

```bash
M="examples/data/M0.npy examples/data/M15.npy"; SP="--spacing 10 1.5625 1.5625"
python scripts/visualize_implicit.py $M surface --t 0 --contours --step 2 $SP --out surface.png   # f = 0 isosurface + input contours
python scripts/visualize_implicit.py $M animate --frames 5 --step 2 $SP --out morph.gif           # surface morphing over t
python scripts/visualize_implicit.py $M slice --t 0 --z 5 --out slice.png                         # f and constraint points on one slice (debugging)
python scripts/visualize_implicit.py $M height --t 0.5 --z 5 --step 2 --out height.png            # f as a 3D height field
```

The same functions are available from `implicit_heart.visualize` (`isosurface`, `render_surface`,
`plot_field_slice`, `plot_height_field`, ...).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_regression.py` checks the fitted function against reference values on synthetic shapes;
`tests/test_constraints.py` checks the constraint conventions (coordinates, signed distances, holes, de-duplication).

## Citation

This code implements:

> Greg Turk and James F. O'Brien. 1999. Shape Transformation Using Variational Implicit Functions.
> In *Proceedings of SIGGRAPH 99*, 335–342. https://doi.org/10.1145/311535.311580

## License

BSD 3-Clause. See [LICENSE](LICENSE).
