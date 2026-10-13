"""
Fit an implicit function to cardiac masks and visualize it (requires the `viz` extra).

Two .npy files are stacked as two time frames (4D input); one 3D file is a slice stack.
Omit --out to open an interactive window.

Examples (spacing is the ACDC voxel size in mm):
    # f = 0 isosurface of the two frames at t = 0, with the input contours
    python scripts/visualize_implicit.py examples/data/M0.npy examples/data/M15.npy surface --t 0 --contours --step 2 --spacing 10 1.5625 1.5625 --out surface.png

    # Surface through the slice stack of one volume, as an orbiting GIF
    python scripts/visualize_implicit.py examples/data/M0.npy surface --orbit --step 2 --spacing 10 1.5625 1.5625 --out stack.gif

    # Morph from frame 0 to frame 1 (5 frames)
    python scripts/visualize_implicit.py examples/data/M0.npy examples/data/M15.npy animate --frames 5 --step 2 --spacing 10 1.5625 1.5625 --out morph.gif

    # Debug view: f and the constraint points on slice 5 at t = 0.5
    python scripts/visualize_implicit.py examples/data/M0.npy examples/data/M15.npy slice --t 0.5 --z 5 --out slice.png

    # f as a 3D height field (Turk & O'Brien style)
    python scripts/visualize_implicit.py examples/data/M0.npy examples/data/M15.npy height --t 0.5 --z 5 --step 2 --out height.png
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from implicit_heart import ImplicitFunction
from implicit_heart import visualize as viz


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("masks", nargs="+", help="one 3D slice-stack .npy, or two .npy frames (stacked as time)")
    parser.add_argument("command", choices=["surface", "animate", "slice", "height"])
    parser.add_argument("--grid", type=float, default=0, help="grid_spacing of the fit (0 disables grid constraints)")
    parser.add_argument("--window", type=int, help="windowed fit over this many slices (for large stacks)")
    parser.add_argument("--tmax", type=float, help="coordinate of the last slice/frame "
                        "(default: 1 for frames; a slice stack is placed by --spacing instead)")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"))
    parser.add_argument("--step", type=int, default=1, help="pixel stride when sampling f")
    parser.add_argument("--z-samples", type=int, help="samples per slice gap along z (default 4 for a slice stack, 1 for frames)")
    parser.add_argument("--out", help="output .png (or .gif for surface/animate); omit for an interactive window")
    parser.add_argument("--t", type=float, default=0.0, help="time (frames) or slice coordinate (slice stack)")
    parser.add_argument("--z", type=int, help="slice index for the slice/height commands")
    parser.add_argument("--orbit", action="store_true", help="surface: orbit the camera (use a .gif --out)")
    parser.add_argument("--contours", action="store_true", help="surface: overlay the input mask contours")
    parser.add_argument("--frames", type=int, default=5, help="animate: number of frames")
    return parser.parse_args()


def main():
    args = get_args()
    masks = [np.load(p).astype(np.uint8) for p in args.masks]
    I = masks[0] if len(masks) == 1 else np.stack(masks)
    # The fit uses the physical spacing, so it is isotropic in physical space; for a slice stack an
    # explicit --tmax places the slices instead.
    spacing = None if I.ndim == 3 and args.tmax is not None else args.spacing
    start = time.time()
    imp = ImplicitFunction(I, tmax=1.0 if args.tmax is None else args.tmax, grid_spacing=args.grid, spacing=spacing,
                          window=args.window)
    print(f"fit {I.shape} with k={len(imp.c)} constraints in {time.time() - start:.1f}s")

    start = time.time()
    if args.command == "surface":
        mesh = viz.isosurface(imp, args.t, args.step, args.spacing, args.z_samples)
        lines = viz.contour_lines(imp, args.t, args.spacing) if args.contours else None
        viz.render_surface(mesh, lines, args.out, args.orbit, title=f"f = 0, t = {args.t}")
    elif args.command == "animate":
        if args.out is None:
            raise SystemExit("animate needs --out PATH.gif")
        viz.animate_surfaces(imp, np.linspace(0, imp.tmax, args.frames), args.out, args.step, args.spacing, args.z_samples)
    elif args.command == "slice":
        fig = viz.plot_field_slice(imp, args.t, args.z, step=args.step)
        if args.out:
            fig.savefig(args.out, dpi=150, bbox_inches="tight")
        else:
            plt.show()
    else:
        viz.plot_height_field(imp, args.t, args.z, args.out, args.step)
    print(f"{args.command} took {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
