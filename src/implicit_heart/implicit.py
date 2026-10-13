# Standard libraries
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

# 3rd party libraries
import numpy as np
import scipy.linalg
import scipy.spatial.distance
from tqdm.auto import tqdm
import skimage.morphology
from skimage.morphology import disk
from skimage.filters import median
from skimage.transform import resize

# My libraries
from .data import Data

logger = logging.getLogger(__name__)


def _phi(r2: np.ndarray) -> np.ndarray:
    """Thin-plate radial basis function phi(r) = r^2 log(r) = r^2 log(r^2) / 2, given squared distances r^2."""
    out = np.zeros_like(r2)
    np.log(r2, out=out, where=r2 > 0)
    out *= r2
    out *= 0.5
    return out


class ImplicitFunction:

    def __init__(self, I: np.ndarray or str = None, tmax: float = 1.0, tlist: list = None, grid_spacing=0.1, resolution=1.0, use_propagation=False,
                 spacing: tuple = None, window: int = None, overlap: int = None):
        """
        Initialize the implicit function object with an array I.
        :param I: some NumPy array representing the object to interpolate
                  if I is a string, assume that is the filepath to a pickled save
                  and load that.
        :param tmax: the maximum separation distance between the first index 
                     and the last index.
        :param tlist: optional explicit coordinate for each index of I; overrides tmax.
        :param grid_spacing: spacing of extra signed-distance grid constraints as a
                             fraction of the image size. 0 disables them.
        :param resolution: scale factor applied to each slice before fitting.
        :param use_propagation: copy neighbouring slices' constraints into each slice.
        :param spacing: physical voxel size (z, y, x) with square pixels (y == x). Slices are then placed
                        spacing_z / spacing_x pixel units apart (for 3D input this replaces tmax, unless
                        tlist is given). Default: one unit per slice in 4D, tmax/tlist in 3D.
        :param window: fit separate thin-plate functions on runs of `window` consecutive slices that have
                       constraints (for 4D input: in any frame), blended smoothly along the slice axis. Cost grows linearly
                       with the number of slices instead of cubically; on a 280-section brain it matched one
                       global fit's accuracy at 2.5-4x the speed. None: one global fit.
        :param overlap: slices shared by neighbouring windows (default window // 2, at least 1).

        Leading and trailing slices that are empty in every frame are left out of the fit (the surface
        would overshoot into them) and come back empty from interpolate().
        """
        tlist = tlist or []
        # If I is a string, assume that is a filepath to a pickled object and load that.
        if type(I) == str:
            self.load(I)
        # Else I is an array, so set up the
        # variational interpolation on I.
        elif type(I) == np.ndarray: 
            # Define all internal parameters first.
            if tlist and len(tlist) != I.shape[0]:
                raise Exception("tlist and I do not have the same number of time-steps.")
            if spacing is not None and not np.isclose(spacing[1], spacing[2], rtol=1e-3):
                raise ValueError(f"In-plane pixels must be square, got spacing (z, y, x) = {spacing}.")
            sz = 1.0 if spacing is None else spacing[0] / spacing[2]
            # Coordinate of every input slice along the slice axis.
            if len(I.shape) == 3:
                n = I.shape[0]
                if tlist:
                    coords = list(tlist)
                elif spacing is not None:
                    coords = [i * sz for i in range(n)]
                else:
                    coords = [tmax / (n - 1) * i for i in range(n)]
                self._tmax = max(coords) if tlist or spacing is not None else tmax
            else:
                coords = [i * sz for i in range(I.shape[1])]
                self._tmax = max(tlist) if tlist else tmax
            self.slice_coords = coords
            self._dim = I.shape
            self._c = None  # NumPy array of both boundary and normal constraints
            self._d = None  # Weights (d_i for i = 0 ... k - 1)
            self._p = None  # P(x) values: p0, p1, p2, p3, p4

            # Sanity check for input dimensions.
            if len(self.dim) not in [3, 4] or self.dim[0] < 2:
                raise Exception(f"Input image dimensions ({self.dim}) are invalid. " +                                 "Dimensions should be 3D or 4D and should have at least " +                                 "2 slices (3D) or 2 time-steps (4D).")

            logger.debug("Creating constraints.")
            start = time.time()
            
            # Leave out leading/trailing slices that are empty in every frame; keep at least two slices in 3D.
            occupied = np.flatnonzero(I.any(axis=(1, 2)) if len(I.shape) == 3 else I.any(axis=(0, 2, 3)))
            z0, z1 = (occupied[0], occupied[-1]) if len(occupied) else (0, len(coords) - 1)
            if len(I.shape) == 3 and z1 == z0:
                z0, z1 = (z0 - 1, z1) if z1 == len(coords) - 1 else (z0, z1 + 1)
            self.z_range = (int(z0), int(z1))
            kept = slice(z0, z1 + 1)

            # Create constraints and generate matrix.
            if len(I.shape) == 3:
                data_ = Data(I[kept], tmax, coords[kept], grid_spacing=grid_spacing, resolution=resolution,
                             use_propagation=use_propagation)
            else:
                data_ = Data(I[:, kept], tmax, tlist, grid_spacing=grid_spacing, resolution=resolution,
                             use_propagation=use_propagation, slice_coords=coords[kept])
            self.data = data_
            self._dim = data_.I.shape  # constraints live on the (possibly resized) grid
            # The thin-plate interpolant is invariant to uniform scaling of the coordinates, so solve in
            # coordinates scaled to ~[0, 1]: same f, far better conditioned matrix (cond ~1e7 vs ~1e12+).
            self._scale = 1.0 / max(self._dim[-2:])
            self._c = data_.get_c()
            h = data_.get_h()
            k = len(self._c)
            
            logger.debug("Created %d constraints in %.2fs. Solving.", k, time.time() - start)
            start = time.time()

            fitted = np.unique(self._c[:, 2])  # slice coordinates that carry constraints (column 2)
            if window is not None and window < 2:
                raise ValueError(f"window must be at least 2 slices, got {window}.")
            self.windows = None  # (lo, hi, constraint indices, d, p) per window of a windowed fit
            if window is None or window >= len(fitted):
                self._d, self._p = self._solve(self._c, h)
            else:
                overlap = window // 2 if overlap is None else overlap
                if not 1 <= overlap < window:
                    raise ValueError(f"overlap must be between 1 and window - 1, got {overlap} for window {window}.")
                starts = list(range(0, len(fitted) - window + 1, window - overlap))
                if starts[-1] + window < len(fitted):
                    starts.append(len(fitted) - window)
                def fit_window(first):
                    lo, hi = fitted[first], fitted[first + window - 1]
                    idx = np.flatnonzero((self._c[:, 2] >= lo) & (self._c[:, 2] <= hi))  # column 2: slice axis
                    return (lo, hi, idx, *self._solve(self._c[idx], h[idx]))

                # Windows are independent and too small to keep every BLAS thread busy; LAPACK releases
                # the GIL, so solve several at once.
                with ThreadPoolExecutor(min(4, len(starts))) as pool:
                    self.windows = list(pool.map(fit_window, starts))
            logger.debug("Solved in %.2fs.", time.time() - start)


    def _solve(self, c: np.ndarray, h: np.ndarray):
        """
        Solves the interpolation system for constraints c with values h; returns (d, p).
        For systems with up to a few thousand constraints, the system can be solved directly with a
        technique such as symmetric LU decomposition (Turk & O'Brien used it for all their examples).
        """
        if len(c) == 0:
            raise ValueError("No constraints to fit: the masks are empty.")
        # A coordinate that is the same for every constraint (e.g. one occupied slice) would make its affine
        # column a multiple of the constant one, i.e. a singular matrix; leave it out of P (its p is 0).
        varying = np.flatnonzero(np.ptp(c, axis=0) > 0)
        M = self._create_matrix(c[:, varying] * self._scale)
        rhs = np.concatenate([np.ravel(h), np.zeros(len(varying) + 1)])
        lu, piv = scipy.linalg.lu_factor(M, overwrite_a=True, check_finite=False)  # M is not kept: k^2 floats
        del M
        x = scipy.linalg.lu_solve((lu, piv), rhs)
        p = np.zeros(c.shape[1] + 1)
        p[0], p[1 + varying] = x[len(c)], x[len(c) + 1:]
        return x[:len(c)], p
        

    def interpolate(self, t: float, thresh: float = 0.0, postprocess: bool = False, kernel_size=3,
                    coarse_step: int = 4) -> np.ndarray:
        """
        Generate an interpolated output given a t.
        :param t: the index in which to generate an interpolated output for.
                  t should be between 0 and tmax.
        :param thresh: which threshold value to consider a pixel as within the boundary.
        :param postprocess: median filter and remove small objects/holes per slice. Off by default: it deletes
                            components under 64 px (e.g. apical slices) for a negligible gain elsewhere.
        :param coarse_step: evaluate f exactly only near the surface, found on a grid with this pixel stride
                            (see sample). 1 evaluates every pixel.
        :return: Returns a 2D image if dimensions are size 3. 
                 Returns a 3D volume if dimensions are size 4.
        """
        if t > self.tmax or t < 0:
            raise Exception(f"t = {t} is out of range! Implicit function\'s t_max is {self.tmax}.")
        # Threshold f on the pixel grid of the output; coordinates are (x, y, [z,] t).
        z0, z1 = self.z_range
        if len(self.dim) == 3:
            _, h, w = self.dim
            zs = [None]
        else:
            _, _, h, w = self.dim
            zs = self.slice_coords[z0:z1 + 1]
        logger.debug("Output dimensions: %s, threshold: %s", (len(zs), h, w), thresh)
        if len(self.dim) == 3 and not self.slice_coords[z0] <= t <= self.slice_coords[z1]:
            interp = np.zeros((h, w))  # between dropped empty slices
        else:
            interp = (self.sample(zs, t, np.arange(h), np.arange(w), coarse_step, thresh) > thresh).astype(float)
            interp = interp[0] if len(self.dim) == 3 else interp
        
        if postprocess:
            # Post-process each 2D slice independently.
            slices = interp.reshape(-1, *interp.shape[-2:])
            for z in range(slices.shape[0]):
                cleaned = median(slices[z], disk(kernel_size)).astype(bool)
                cleaned = skimage.morphology.remove_small_objects(cleaned)
                slices[z] = skimage.morphology.remove_small_holes(cleaned)

        if len(self.dim) == 4:  # put back the dropped empty slices
            full = np.zeros((len(self.slice_coords), h, w))
            full[z0:z1 + 1] = interp
            interp = full
        
        # Resize to original dimensions
        if self.data.resolution != 1:
            interp = np.array([resize(s, self.data.original_shape, order=0, preserve_range=True, anti_aliasing=False)
                               for s in interp.reshape(-1, *interp.shape[-2:])]).reshape(*interp.shape[:-2], *self.data.original_shape)
        
        return interp
    
    
    def sample(self, zs, t, ys, xs, coarse_step: int = 4, level: float = 0.0) -> np.ndarray:
        """
        f on the grid zs x ys x xs as a (len(zs), len(ys), len(xs)) array, evaluated exactly only near
        the `level` set. f is first sampled every `coarse_step` grid points in y and x; a coarse cell whose
        four corners all lie more than a margin (coarse_step grid steps, in pixels) above or below `level`
        gets bilinearly upsampled values, which have the right sign there, and every other cell is
        evaluated exactly. So f > level is exact everywhere, and f is exact wherever it crosses `level`.
        :param zs: slice coordinates (4D input) or [None] (3D input: the plane t).
        :param ys, xs: increasing pixel coordinates of the grid.
        :param coarse_step: 1 evaluates every grid point.
        """
        # ponytail: the margin assumes |grad f| <~ 1 per pixel near the surface (f is ~ a signed distance in
        # pixels there); coarse_step=1 is the exact fallback. Equality is checked in tests/test_constraints.py.
        ys, xs = np.asarray(ys, dtype=float), np.asarray(xs, dtype=float)

        def points(zi, yi, xi):
            cols = [xs[xi], ys[yi]]
            if zs[0] is not None:
                cols.append(np.asarray(zs, dtype=float)[zi])
            return np.column_stack(cols + [np.full(len(zi), float(t))])

        def full(nz, ny, nx):
            zi, yi, xi = (a.ravel() for a in np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing='ij'))
            return zi, yi, xi

        nz, ny, nx = len(zs), len(ys), len(xs)
        iy = np.unique(np.minimum(np.arange(0, ny + coarse_step, coarse_step), ny - 1))
        ix = np.unique(np.minimum(np.arange(0, nx + coarse_step, coarse_step), nx - 1))
        if coarse_step <= 1 or len(iy) < 2 or len(ix) < 2:
            return self.evaluate(points(*full(nz, ny, nx))).reshape(nz, ny, nx)

        zi, yi, xi = full(nz, len(iy), len(ix))
        coarse = self.evaluate(points(zi, iy[yi], ix[xi])).reshape(nz, len(iy), len(ix))
        corners = np.stack([coarse[:, :-1, :-1], coarse[:, 1:, :-1], coarse[:, :-1, 1:], coarse[:, 1:, 1:]])
        margin = coarse_step * max(np.diff(ys).max(initial=1), np.diff(xs).max(initial=1))
        sure = (corners.min(0) > level + margin) | (corners.max(0) <= level - margin)

        # Bilinear upsampling: grid point j lies in coarse cell c[j] at fraction w[j] between its corners.
        cy = np.clip(np.searchsorted(iy, np.arange(ny), side='right') - 1, 0, len(iy) - 2)
        cx = np.clip(np.searchsorted(ix, np.arange(nx), side='right') - 1, 0, len(ix) - 2)
        wy = ((np.arange(ny) - iy[cy]) / (iy[cy + 1] - iy[cy]))[:, None]
        wx = (np.arange(nx) - ix[cx]) / (ix[cx + 1] - ix[cx])
        rows = coarse[:, cy] * (1 - wy) + coarse[:, cy + 1] * wy
        F = rows[:, :, cx] * (1 - wx) + rows[:, :, cx + 1] * wx

        zi, yi, xi = np.nonzero(~sure[:, cy][:, :, cx])
        if len(zi):
            F[zi, yi, xi] = self.evaluate(points(zi, yi, xi))
        logger.debug("Evaluated %d of %d grid points exactly.", len(zi) + coarse.size, F.size)
        return F
    
    
    def save(self, filepath: str):
        """
        Saves the ImplicitFunction object to a provided file (a pickle: load it with the same package
        version).
        :param filepath: the file to save the current ImplicitFunction object to
        """
        with open(filepath, 'wb') as f:        
            pickle.dump(self, f)


    def load(self, filepath: str):
        """
        Loads ImplicitFunction data from a saved file.
        :param filepath: the file to load into the current ImplicitFunction object
        """
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)
    
    
    def evaluate(self, points: np.ndarray, chunk_size: int = None) -> np.ndarray:
        """
        Evaluates the implicit function f(x) = sum_j d_j phi(x - c_j) + P(x); for a windowed fit, the
        weighted average of the windows' functions (weights fade to 0 at each window's interior end).
        :param points: (m, n) array of points in the form (x, y, t) if 3D, and (x, y, z, t) if 4D.
        :param chunk_size: points evaluated per batch, i.e. chunk_size * k floats per thread. Default: a ~64 MB
                           block per thread, so memory does not grow with k.
        :return: (m,) array of function values.
        """
        points = np.asarray(points, dtype=float)
        c = self.c * self._scale
        if self.windows is None:
            return self._rbf(points * self._scale, c, self.d, self.p, chunk_size)
        z, total, weight = points[:, 2], np.zeros(len(points)), np.zeros(len(points))
        for i, (lo, hi, idx, d, p) in enumerate(self.windows):
            u = (z - (lo + hi) / 2) / ((hi - lo) / 2)  # -1 .. 1 across the window
            if i == 0:
                u[u < 0] = 0                             # no fade before the first slice...
            if i == len(self.windows) - 1:
                u[u > 0] = 0                             # ...or after the last
            w = np.where(np.abs(u) <= 1, (1 - u * u) ** 2 + 1e-12, 0.0)
            sel = np.flatnonzero(w)
            if len(sel):
                total[sel] += w[sel] * self._rbf(points[sel] * self._scale, c[idx], d, p, chunk_size)
                weight[sel] += w[sel]
        return total / weight


    def _rbf(self, points, c, d, p, chunk_size=None):
        """sum_j d_j phi(x - c_j) + P(x) for scaled points and centres c."""
        chunk_size = chunk_size or int(np.clip(2 ** 23 // len(c), 256, 4096))
        out = np.empty(len(points))

        def batch(start):  # NumPy releases the GIL here, so chunks run in parallel threads
            out[start:start + chunk_size] = _phi(scipy.spatial.distance.cdist(
                points[start:start + chunk_size], c, 'sqeuclidean')) @ d

        starts = range(0, len(points), chunk_size)
        with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
            list(tqdm(pool.map(batch, starts), total=len(starts), desc="Interpolating",
                      disable=True if len(starts) < 64 else None))  # only long runs, only on a terminal/notebook
        return out + p[0] + points @ p[1:]

    
    def _create_matrix(self, c: np.ndarray) -> np.ndarray:
        """
        Helper function to create a variational 
        interpolation matrix from a set of constraints.
        :param c: a set of constraint vectors in the form
                  [[c1x, c1y, t1],
                   [c2x, c2y, t2],
                   [c3x, c3y, t3] ...]
        :param phi: the radial basis function, phi
        :return: a NumPy matrix 
        """
        k, n = c.shape  # Number of vectors, dimensionality of vectors

        # Main matrix
        A = np.zeros((k + 1 + n, k + 1 + n))

        # Create the submatrix for phi_ij
        # [ phi_11, phi_12, ..., phi_1k
        #   phi_21, phi_22, ..., phi_2k
        #    ...,    ...,   ...,  ...
        #   phi_k1, phi_k2, ..., phi_kk ]
        phi_matrix = _phi(scipy.spatial.distance.cdist(c, c, 'sqeuclidean'))
        A[:k, :k] = phi_matrix

        # Create ones row and ones column
        A[k, :k] = np.ones((1, k))  # Row
        A[:k, k] = np.ones(k)       # Column

        # Create cx, cy, cz (if cz is available)
        A[k+1:, :k] = c.T
        A[:k, k+1:] = c
        return A
   
    #######################
    # Getters and setters #
    #######################
    
    @property
    def tmax(self):
        return self._tmax
    
    @tmax.setter
    def tmax(self, value):
        raise Exception("tmax is read only!")    
    
    
    @property
    def dim(self):
        return self._dim
    
    @dim.setter
    def dim(self, value):
        raise Exception("dim is read only!")
    
    
    @property
    def c(self):
        return self._c
    
    @c.setter
    def c(self, value):
        raise Exception("c is read only!")
        
        
    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        raise Exception("d is read only!")
       
    
    @property
    def p(self):
        return self._p
    
    @p.setter
    def p(self, value):
        raise Exception("p is read only!")