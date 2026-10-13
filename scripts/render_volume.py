"""
Render .npy mask volumes with PyVista (requires the `viz` extra).

Examples:
    # Interactive view of one volume (ACDC / MICCAI 2017 spacing)
    python scripts/render_volume.py output/M0-M15-interp.npy --spacing 10 1.5625 1.5625

    # Multi-label volume with a clip plane, saved as an orbiting GIF
    python scripts/render_volume.py output/all_vol.npy --labels --clip y 98 --gif orbit.gif --orbit

    # Animate a sequence of volumes, one GIF frame per file
    python scripts/render_volume.py output/anim/*.npy --gif anim.gif
"""
import argparse
import glob

import numpy as np
import pyvista as pv

AXES = {"x": 0, "y": 1, "z": 2}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("volumes", nargs="+", help=".npy volume(s); several files become animation frames")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0), metavar=("Z", "Y", "X"),
                        help="voxel spacing, e.g. 10 1.5625 1.5625 (ACDC) or 1.05 0.972 0.972 (MM-WHS)")
    parser.add_argument("--labels", action="store_true",
                        help="colour each label value separately instead of rendering a binary mask")
    parser.add_argument("--clip", nargs=2, metavar=("AXIS", "ORIGIN"),
                        help="add a clip plane and cross-section along x/y/z at the given index")
    parser.add_argument("--gif", help="write a GIF here instead of opening an interactive window")
    parser.add_argument("--orbit", action="store_true", help="with --gif and one volume: orbit the camera")
    return parser.parse_args()


def load(path, labels):
    vol = np.load(path).astype(np.uint8)
    if not labels:
        vol[vol != 0] = 255
    grid = pv.wrap(vol)
    grid.spacing = args.spacing
    return grid


def add(pl, grid, frame_title):
    actors = [pl.add_volume(grid, cmap="glasbey" if args.labels else "Reds", opacity="foreground", shade=True)]
    if args.clip:
        axis, origin = args.clip
        normal, point = [0, 0, 0], [0.0, 0.0, 0.0]
        normal[AXES[axis]], point[AXES[axis]] = 1, float(origin)
        pl.add_volume_clip_plane(actors[0], normal=normal, origin=point)
        actors.append(pl.add_mesh_slice(grid, normal=normal, origin=point, cmap="glasbey"))
    pl.remove_scalar_bar()
    actors.append(pl.add_text(frame_title, position="upper_edge", color="white", shadow=True, font_size=18))
    return actors


args = get_args()
paths = sorted(p for pattern in args.volumes for p in glob.glob(pattern)) or args.volumes
pl = pv.Plotter(off_screen=bool(args.gif))
pl.set_background("black")

if not args.gif:
    add(pl, load(paths[0], args.labels), paths[0])
    pl.camera.zoom(2.5)
    pl.show()
elif args.orbit:
    add(pl, load(paths[0], args.labels), paths[0])
    pl.camera.zoom(2.5)
    path = pl.generate_orbital_path(factor=2.0, n_points=36, shift=0.2)
    pl.open_gif(args.gif)
    pl.orbit_on_path(path, write_frames=True, step=0.05)
    pl.close()
else:
    pl.open_gif(args.gif)
    for p in paths:
        actors = add(pl, load(p, args.labels), p)
        pl.write_frame()
        for a in actors:
            pl.remove_actor(a)
    pl.close()
