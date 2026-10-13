import logging
from collections import defaultdict

import numpy as np
import cv2
import matplotlib.pyplot as plt
import edt
from skimage.transform import resize

# CHAIN_APPROX_NONE
# CHAIN_APPROX_SIMPLE
# CHAIN_APPROX_TC89_L1
# CHAIN_APPROX_TC89_KCOS
APPROX = cv2.CHAIN_APPROX_TC89_KCOS

# Retrieval mode. RETR_CCOMP returns outer contours and hole contours (two-level hierarchy), so holes
# (e.g. the myocardium ring around the LV cavity) get boundary constraints too; RETR_EXTERNAL filled them.
RET_MODE = cv2.RETR_CCOMP

# Distance (pixels) from the centre of an OpenCV contour pixel, outward along the gradient, to where the
# boundary constraint is placed. Signed distances of grid/propagated constraints use the same boundary.
# 0.25 reproduced real LV slices best (mean dice 0.996, area bias +0.2% vs 0.988 / -2.3% at 0, the
# original pixel-centre placement, and 0.988 / +2.3% at 0.5) on the bundled M0/M15 slices.
BOUNDARY_OFFSET = 0.25

logger = logging.getLogger(__name__)

class Data:
    def __init__(self, I: np.ndarray, tmax: float = 1.0, tlist: list = None, grid_spacing=0.1, resolution=1.0, use_propagation=False,
                 slice_coords: list = None):
        """
        Initializes the data object to perform computations on I.
        :param I: A NumPy array object. This should either be a 2D image,
                  3D volume, or 4D array of volumes. If I is a 3D volume,
                  the indices should be organized as I[n, y, x], where n
                  is the slice index in the volume, and x and y are the
                  coordinates in the image. A 4D array should be organized
                  as I[t, n, y, x], where t is a volume for a given time-step.
        :param slice_coords: 4D input only: z coordinate of each slice (default: its index).
        """
        tlist = tlist or []
        self.use_propagation = use_propagation  # Whether to use propagation or not.
        if grid_spacing == 0:
            self.use_grid = False
        else:
            self.use_grid = True  # Whether to use grid constraints or not.
        self.grid_spacing = grid_spacing  # Ratio of the image size to space the grid.

        # Resize image if resolution given
        self.resolution = resolution
        self.original_shape = I.shape[-2:]

        if resolution != 1.0:
            new_I = []
            if len(I.shape) == 3:  # 3D
                for z in range(I.shape[0]):
                    new_I.append(_resize_mask(I[z], resolution))
            else:  # 4D
                for t in range(I.shape[0]):
                    new_I.append([_resize_mask(I[t, z], resolution) for z in range(I.shape[1])])
            I = np.array(new_I)

        self.I = I
        self.tlist = tlist

        if tlist and len(tlist) == I.shape[0]:
            self.tmax = max(tlist)
        elif tlist and len(tlist) != I.shape[0]:
            raise Exception("Invalid tlist length: " + str(len(tlist)))
        else:
            self.tmax = tmax

        self.constraints = defaultdict(list)
        self.constraints4D = dict()
        self.dim = I.shape
        self.slice_coords = list(range(I.shape[1])) if slice_coords is None and len(I.shape) == 4 else slice_coords

        # Distance maps, indexed like I: dt is the distance from foreground voxels to the nearest
        # background voxel, dt_inv from background voxels to the nearest foreground. They are 3D over each
        # volume (paper Sec. 3.5.2), so empty slices get consistent values, with the slice axis scaled
        # like the constraint coordinates: _coord() spacing for 3D input, slice_coords spacing for 4D input.
        if len(self.dim) == 3:
            dz = np.mean(np.diff([self._coord(i) for i in range(self.dim[0])]))
            volumes, anisotropy = self.I[np.newaxis].astype(bool), (dz, 1, 1)
        else:
            dz = np.mean(np.diff(self.slice_coords)) if len(self.slice_coords) > 1 else 1
            volumes, anisotropy = self.I.astype(bool), (dz, 1, 1)
        self.dt = np.array([edt.edt(v, anisotropy=anisotropy) for v in volumes]).reshape(self.dim)
        self.dt_inv = np.array([edt.edt(~v, anisotropy=anisotropy) for v in volumes]).reshape(self.dim)

        self._create_constraints()

        # Propagate
        if self.use_propagation:
            if len(self.dim) == 3:
                self.propagate()
            else:
                self.propagate4D()

        if self.use_grid:
            if len(self.dim) == 3:
                for idx in range(self.dim[0]):
                    self._grid_constraints(idx, self._coord(idx), t=None, spacing=self.grid_spacing)
            else:
                for idx in range(self.dim[0]):
                    for z in range(self.dim[1]):
                        self._grid_constraints(idx, z=z, t=self._coord(idx), spacing=self.grid_spacing)


    def _coord(self, idx: int) -> float:
        """
        Coordinate along the interpolated axis (slice position for 3D input, time for 4D input)
        of index idx: tlist[idx] if given, otherwise evenly spaced on [0, tmax].
        """
        if self.tlist:
            return self.tlist[idx]
        return self.tmax / (self.dim[0] - 1) * idx


    def _signed_distance(self, idx, x, y) -> float:
        """
        Signed distance (pixels) from pixel (x, y) of slice idx to the mask boundary, which lies
        BOUNDARY_OFFSET outward from the outermost foreground pixels. Positive inside.
        :param idx: index into the leading axes of I: the slice for 3D input, (t, slice) for 4D input.
        """
        x, y = round(x), round(y)
        dt = self.dt[idx][y, x]
        # The boundary lies BOUNDARY_OFFSET of the way from a foreground to the adjacent background voxel.
        # That step is a pixel in-plane, but can be shorter towards a neighbouring slice; max() keeps the
        # in-plane convention for steps >= 1 and the right sign below.
        if dt != 0:
            return max(dt - 1 + BOUNDARY_OFFSET, BOUNDARY_OFFSET * dt)
        dist = -max(self.dt_inv[idx][y, x] - BOUNDARY_OFFSET, (1 - BOUNDARY_OFFSET) * self.dt_inv[idx][y, x])
        # Sanity check. set default value if the slice has no foreground at all
        return -3 if dist == float('-inf') else dist


    def show(self, idx: int, img: np.ndarray = None, target: [int] = [-1, 0, 1]):
        """
        Visualize the index.
        """
        plt.title(str(idx))
        if img is None:  # If no image is provided, use the original image at index idx.
            img = self.I[idx]
        plt.imshow(img, cmap='gray')
        curr_constraints = self.constraints[idx]
        for pt_tuple in curr_constraints:
            pt, pt_type = pt_tuple
            if pt_type > 0 and 1 in target:     # Normal constraint
                plt.plot(pt[0], pt[1], 'gX', ms=2)
            elif pt_type < 0 and -1 in target:  # Outside constraint
                plt.plot(pt[0], pt[1], 'bX', ms=2)
            elif pt_type == 0 and 0 in target:   # Boundary constraint
                plt.plot(pt[0], pt[1], 'ro', ms=2)
        plt.show()


    def _flatten(self):
        """
        Flattens the per-slice constraints in sorted-key order into points c (k, n) and values h (k,).
        Repeated points are dropped (keeping the first, i.e. boundary/normal constraints over
        propagated and grid ones), since duplicate rows make the interpolation matrix singular.
        """
        if len(self.dim) == 3:
            per_slice = [self.constraints[i] for i in sorted(self.constraints)]
        else:
            per_slice = [self.constraints4D[t][z] for t in sorted(self.constraints4D)
                         for z in sorted(self.constraints4D[t])]
        pairs = [pair for constraints in per_slice for pair in constraints]
        c = np.array([pt for pt, _ in pairs], dtype=float)
        h = np.array([value for _, value in pairs], dtype=float)
        keep = np.sort(np.unique(c, axis=0, return_index=True)[1])
        if len(keep) < len(c):
            logger.debug("Dropped %d duplicate constraint points.", len(c) - len(keep))
        return c[keep], h[keep]


    def get_c(self):
        """
        Creates constraint array to return to implicit.py.
        """
        c = self._flatten()[0]
        logger.debug("Constraint array shape: %s", c.shape)
        return c


    def get_h(self):
        """
        Creates h array to return to implicit.py
        """
        return self._flatten()[1][:, np.newaxis]


    def propagate4D(self):
        for t in sorted(self.constraints4D.keys()):
            self.constraints = self.constraints4D[t]  # propagate() edits this dict in place
            self.propagate(timestep_t=self._coord(t), t_idx=t)
        self.constraints = defaultdict(list)


    def propagate(self, timestep_t=None, t_idx=None):
        """
        Propagate neighboring constraints: each interior slice receives the constraint locations
        of its two neighbours, valued by their signed distance in the current slice.
        :param timestep_t: time coordinate, when propagating within one volume of a 4D input.
        :param t_idx: time index of that volume, used to look up its distance maps.
        """
        min_idx = min(self.constraints.keys())
        max_idx = max(self.constraints.keys())
        # Copy from the slices' own constraints, not ones already propagated into them.
        own = {i: list(self.constraints[i]) for i in range(min_idx, max_idx + 1)}
        for i in range(min_idx + 1, max_idx):
            logger.debug("Propagating constraints into slice %d", i)
            dist_idx = i if t_idx is None else (t_idx, i)
            for pt, _ in own[i - 1] + own[i + 1]:
                pt_dist = self._signed_distance(dist_idx, pt[0], pt[1])
                if timestep_t is not None:
                    pt = np.array([pt[0], pt[1], self.slice_coords[i], timestep_t])
                else:
                    pt = np.array([pt[0], pt[1], self._coord(i)])
                self.constraints[i].append((pt, pt_dist))


    def _create_constraints(self):
        """
        Creates boundary and normal constraints for the input I.
        """
        # I is a 3D volume.
        if len(self.dim) == 3:
            self._create_constraints3D()
        # I is a 4D series of volumes under time lapses.
        elif len(self.dim) == 4:
            self._create_constraints4D()
        else:
            raise Exception("Invalid dimensions: " + str(self.dim))


    def _create_constraints2D(self, img: np.ndarray, idx: int, z: float, time_t: float = None, dist: float = 0.9):
        """
        Creates boundary and normal constraints for a 2D image and stores them in self.constraints[idx].
        Boundary constraints (value 0) lie BOUNDARY_OFFSET outward from the contour pixels found by OpenCV. Each is paired with an inside (+1) and an outside (-1) normal constraint
        `dist` pixels away along the image gradient.
        :param z: the third coordinate value to embed coordinates in 3D
        :param time_t: the fourth (time) coordinate for 4D input, otherwise None
        """
        contours, _ = cv2.findContours(img.astype('uint8'), RET_MODE, APPROX)
        if not contours:
            return
        # The Scharr gradient of a binary mask points into the foreground.
        grad_x = cv2.Scharr(img.astype('uint8'), cv2.CV_64F, dx=1, dy=0)
        grad_y = cv2.Scharr(img.astype('uint8'), cv2.CV_64F, dx=0, dy=1)
        extra = [z] if time_t is None else [z, time_t]

        # Collect every component's contour.
        for x, y in np.concatenate([c[:, 0, :] for c in contours]):
            grad = np.array([grad_x[y, x], grad_y[y, x]])
            norm = np.linalg.norm(grad)
            if norm == 0:
                logger.warning("Sobel gradient is zero at %s; skipping its normal constraints.", (x, y))
                self.constraints[idx].append((np.array([x, y, *extra], dtype=float), 0))
                continue
            grad /= norm
            b = np.array([x, y]) - BOUNDARY_OFFSET * grad
            self.constraints[idx].append((np.array([*b, *extra]), 0))
            self.constraints[idx].append((np.array([*(b + dist * grad), *extra]), 1))   # Inside constraint
            self.constraints[idx].append((np.array([*(b - dist * grad), *extra]), -1))  # Outside constraint


    def _create_constraints3D(self, volume: np.ndarray = None, timestep_t: float = None):
        """
        Creates boundary and normal constraints for every slice of a 3D volume.
        :param volume: the volume to use; defaults to I (3D input)
        :param timestep_t: time coordinate when the volume is one frame of a 4D input. Its slices are
                           then placed at slice_coords; for 3D input they are placed at _coord(i).
        """
        volume = self.I if volume is None else volume
        for i in range(volume.shape[0]):
            z = self._coord(i) if timestep_t is None else self.slice_coords[i]
            self._create_constraints2D(volume[i], i, z=z, time_t=timestep_t)


    def _create_constraints4D(self):
        """
        Creates boundary and normal constraints for each frame of a 4D input, stored in constraints4D[t].
        """
        for i in range(self.dim[0]):
            self._create_constraints3D(self.I[i], timestep_t=self._coord(i))
            self.constraints4D[i] = self.constraints
            self.constraints = defaultdict(list)


    def _grid_constraints(self, idx, z, t=None, spacing=0.1):
        """
        Creates a grid of constraints valued by their signed distance to the boundary.
        """
        h, w = self.I.shape[-2:]
        target = self.constraints[idx] if t is None else self.constraints4D[idx][z]
        dist_idx = idx if t is None else (idx, z)
        for x in range(0, w, max(1, int(w * spacing))):
            for y in range(0, h, max(1, int(h * spacing))):
                pt = [x, y, z] if t is None else [x, y, self.slice_coords[z], t]
                target.append((np.array(pt, dtype=float), self._signed_distance(dist_idx, x, y)))


def _resize_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    """Nearest-neighbour resize that keeps a mask binary."""
    shape = (int(mask.shape[0] * factor), int(mask.shape[1] * factor))
    return resize(mask, shape, order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)
