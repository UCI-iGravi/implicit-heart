# implicit-heart

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
| `grid_spacing` | Extra signed-distance constraints on a grid, as a fraction of image size. `0` disables. Their values are pixel distances, so with a small `tmax` (slices close together relative to the image size) they dominate and the shape changes abruptly near the end; use a larger `tmax` or `grid_spacing=0`. |
| `resolution` | Downsampling factor applied before fitting; output is resized back. |
| `use_propagation` | Copy neighbouring slices' constraint locations into each slice. |

`interpolate(t, thresh=0.0, postprocess=True, kernel_size=3)` returns the mask at `t`; post-processing
applies a median filter and removes small objects and holes. Set `logging.basicConfig(level=logging.DEBUG)`
for timing output.

**Performance:** the linear system is dense, so memory grows with the square of the number of
constraints and solve time with the cube. `interpolate` evaluates every pixel in Python, which is slow
for full 4D volumes (expect minutes to hours at 216×256×10).

## Notebooks

Run from the `notebooks/` folder; paths are relative to it.

| Notebook | Data needed |
|---|---|
| `01_3d_interpolation.ipynb` | `data/npy/lv.npy` (binary LV masks, `(T, Z, H, W)`) |
| `02_4d_interpolation.ipynb` | Bundled `examples/data/M0.npy`, `M15.npy` (two LV frames); last section uses `data/npy/lv.npy` |
| `03_acdc_ejection_fraction.ipynb` | [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) training set at `data/acdc/training/` |

Datasets are not redistributed here. Download them from their providers and place them under `data/`
(ignored by git). Generated results go to `output/`.

## Rendering volumes

```bash
pip install -e ".[viz]"
python scripts/render_volume.py output/M0-M15-interp.npy --spacing 10 1.5625 1.5625
python scripts/render_volume.py "output/anim/*.npy" --gif anim.gif          # one frame per file
python scripts/render_volume.py all_labels.npy --labels --clip y 98 --gif orbit.gif --orbit
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`tests/test_regression.py` checks the solver and interpolation against reference outputs on synthetic shapes.

## Citation

This code implements:

> Greg Turk and James F. O'Brien. 1999. Shape Transformation Using Variational Implicit Functions.
> In *Proceedings of SIGGRAPH 99*, 335–342. https://doi.org/10.1145/311535.311580

## License

BSD 3-Clause. See [LICENSE](LICENSE).
