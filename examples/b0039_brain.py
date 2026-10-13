"""
Implicit representation of a whole mouse brain mask (B0039, 25 um TissueCyte, 280 coronal sections).

Fits every 8th section (400 um apart) with a windowed thin-plate fit, then renders the f = 0 surface, the
field on a keyframe and on a held-out section, and held-out sections against the true mask.
Requires the `viz` extra. Run from the repo root:

    python examples/b0039_brain.py            # writes output/b0039/

The mask (examples/data/B0039_brain_25um_mask.npz: `mask` [section, y, x], `spacing_mm`) is RegTools'
3-D GrabCut tissue mask seeded by Multi-Otsu-4.
"""
import os
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_heart import ImplicitFunction
from implicit_heart import visualize as viz
from implicit_heart.metrics import dice

STRIDE = 8  # keyframe every 8th section
OUT = "output/b0039"


def main():
    os.makedirs(OUT, exist_ok=True)
    data = np.load("examples/data/B0039_brain_25um_mask.npz")
    mask, spacing = data["mask"].astype(np.uint8), tuple(data["spacing_mm"])
    keys = list(range(0, len(mask), STRIDE))
    key_spacing = (spacing[0] * STRIDE, spacing[1], spacing[2])  # slice gap between keyframes
    to_t = spacing[0] / spacing[2]                                 # section index -> slice coordinate (px)

    start = time.time()
    imp = ImplicitFunction(mask[keys], grid_spacing=0, spacing=key_spacing, window=4)
    print(f"fit {len(keys)} keyframes, {len(imp.c)} constraints, {len(imp.windows)} windows: {time.time() - start:.1f}s")

    # 1. The f = 0 surface, alone, with the keyframe contours, and orbiting.
    start = time.time()
    mesh = viz.isosurface(imp, step=2, spacing=key_spacing, z_samples=2)
    viz.render_surface(mesh, path=f"{OUT}/surface.png", title="B0039: f = 0")
    viz.render_surface(mesh, viz.contour_lines(imp, spacing=key_spacing), path=f"{OUT}/surface_keyframes.png",
                       title=f"f = 0 through {len(keys)} keyframe contours")
    viz.render_surface(mesh, path=f"{OUT}/orbit.gif", orbit=True)
    print(f"surface ({mesh.n_points} points, {mesh.volume:.0f} mm^3): {time.time() - start:.1f}s")

    # 2. The field on a keyframe (with its constraints) and on a held-out section (none: pure interpolation).
    key, held = keys[len(keys) // 2], keys[len(keys) // 2] + STRIDE // 2
    for section, name in ((key, "keyframe"), (held, "heldout")):
        fig = viz.plot_field_slice(imp, t=section * to_t, step=2)
        fig.axes[0].set_title(f"f on section {section} ({name})")
        fig.savefig(f"{OUT}/field_{name}_{section}.png", dpi=110)
        plt.close(fig)
    viz.plot_height_field(imp, t=held * to_t, path=f"{OUT}/height_field_{held}.png", step=4)

    # 3. Held-out sections: interpolated vs. true mask vs. copying the nearest keyframe.
    held_out = [z for z in range(keys[0], keys[-1]) if z % STRIDE][::2]
    frames, scores = [], []
    for z in held_out:
        interp = imp.interpolate(z * to_t)
        nearest = mask[min(keys, key=lambda k: abs(k - z))]
        scores.append((z, dice(interp, mask[z]), dice(nearest, mask[z])))
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.imshow(mask[z], cmap="gray", vmin=0, vmax=2)
        ax.contour(interp, [0.5], colors="#e4572e", linewidths=1.5)
        ax.contour(nearest, [0.5], colors="#29a3ff", linewidths=1, linestyles="--")
        ax.set_title(f"section {z}: implicit dice {scores[-1][1]:.3f} | nearest keyframe {scores[-1][2]:.3f}",
                     fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    imageio.mimsave(f"{OUT}/heldout_sections.gif", frames, duration=0.15, loop=0)

    z, ours, near = np.array(scores).T
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(z, ours, ".-", color="#e4572e", label=f"implicit (mean {ours.mean():.3f})")
    ax.plot(z, near, ".--", color="#29a3ff", label=f"nearest keyframe (mean {near.mean():.3f})")
    ax.set_xlabel("held-out section"), ax.set_ylabel("dice vs. mask"), ax.legend()
    ax.set_title(f"B0039: interpolating sections {STRIDE * 50} um apart")
    fig.tight_layout()
    fig.savefig(f"{OUT}/heldout_dice.png", dpi=110)
    print(f"held-out sections: implicit dice {ours.mean():.4f} (min {ours.min():.4f}), "
          f"nearest keyframe {near.mean():.4f} (min {near.min():.4f})")


if __name__ == "__main__":
    main()
