# Standard libraries
import logging
import pickle
import time
import math

# 3rd party libraries
import numpy as np
import scipy.linalg
from tqdm.auto import tqdm
import skimage.morphology
from skimage.morphology import disk
from skimage.filters import median
from skimage.transform import resize

# My libraries
from .data import Data

logger = logging.getLogger(__name__)


class ImplicitFunction:
    def __init__(self, I: np.ndarray or str = None, tmax: float = 1.0, tlist: list = None, grid_spacing=0.1, resolution=1.0, use_propagation=False):
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
        """
        tlist = tlist or []
        # If I is a string, assume that is a filepath to a pickled object and load that.
        if type(I) == str:
            self.load(I)
        # Else I is an array, so set up the
        # variational interpolation on I.
        elif type(I) == np.ndarray: 
            # Define all internal parameters first.
            if tlist and len(tlist) == I.shape[0]:
                self._tmax = max(tlist)
            elif tlist and len(tlist) != I.shape[0]:
                raise Exception("tlist and I do not have the same number of time-steps.")
            else:
                self._tmax = tmax
            self._dim = I.shape
            self._c = None  # NumPy array of both boundary and normal constraints
            self._d = None  # Weights (d_i for i = 0 ... k - 1)
            self._p = None  # P(x) values: p0, p1, p2, p3, p4

            # Sanity check for input dimensions.
            if len(self.dim) not in [3, 4] or self.dim[0] < 2:
                raise Exception(f"Input image dimensions ({self.dim}) are invalid. " +                                 "Dimensions should be 3D or 4D and should have at least " +                                 "2 slices (3D) or 2 time-steps (4D).")

            logger.debug("Creating constraints.")
            start = time.time()
            
            # Create constraints and generate matrix.
            data_ = Data(I, tmax, tlist, grid_spacing=grid_spacing, resolution=resolution,
                         use_propagation=use_propagation)
            self.data = data_
            self._c = data_.get_c()
            h = data_.get_h()
            k = len(self._c)
            
            logger.debug("Created %d constraints in %.2fs. Creating matrix.", k, time.time() - start)
            start = time.time()
            
            # Create the matrix from the constraints and the height values
            # height values are 0 for boundary constraints, 1 for normal constraints, -1 for outside
            M = self._create_matrix(self.c)
            self.M = M
            
            h = np.concatenate([h, np.zeros((self._c.shape[1] + 1, 1))])
            logger.debug("Constructed matrix in %.2fs. Solving matrix.", time.time() - start)
            start = time.time()
                
            # Solve for matrix and get solution.
            # For systems with up to a few thousand constraints, the system can be
            # solved directly with a technique such as symmetric LU decomposition. 
            # We used symmetric LU decomposition to solve this system
            # for all of the examples shown in this paper.
            lu, piv = scipy.linalg.lu_factor(M)
            x = scipy.linalg.lu_solve((lu, piv), h)
            x = x.flatten()
            
            logger.debug("Solved matrix in %.2fs.", time.time() - start)
            
            self._d = x[:k]
            self._p = x[k:]
        

    def interpolate(self, t: float, thresh: float = 0.0, postprocess: bool = True, kernel_size=3) -> np.ndarray:
        """
        Generate an interpolated output given a t.
        :param t: the index in which to generate an interpolated output for.
                  t should be between 0 and tmax.
        :param thresh: which threshold value to consider a pixel as within the boundary.
        :return: Returns a 2D image if dimensions are size 3. 
                 Returns a 3D volume if dimensions are size 4.
        """
        if t > self.tmax or t < 0:
            raise Exception(f"t = {t} is out of range! Implicit function\'s t_max is {self.tmax}.")
        k = len(self.d)
        
        # If dimensions are size 3, generate a 2D image.
        if len(self.dim) == 3:
            _, h, w = self.dim
            interp = np.zeros((h, w))
            for y in tqdm(range(h), desc="Interpolating image"):
                for x in range(w):
                    #arr = np.array([x / (w - 1), y / (h - 1), t])
                    arr = np.array([x, y, t])
                    output = self._interpolate_point(arr)
                    #interp[y, x] = output
                    interp[y, x] = 0 if output <= thresh else 1
        else:  # If dimensions are size 4, generate a 3D volume.
            _, num_slices, h, w = self.dim
            interp = np.zeros((num_slices, h, w))
            logger.debug("Output dimensions: %s, threshold: %s", interp.shape, thresh)
            with tqdm(total=num_slices * h, desc=f"Interpolating volume stack") as pbar:
                for z in range(num_slices):  # Z-axis (stacks of volume)
                    for y in range(h):       # Y-axis (vertical pixels)
                        for x in range(w):   # X-axis (horizontal pixels)
                            #arr = np.array([x / (w - 1), y / (h - 1), z / (num_slices - 1), t])
                            arr = np.array([x, y, z, t])
                            output = self._interpolate_point(arr)
                            #interp[y, x] = output
                            interp[z, y, x] = 0 if output <= thresh else 1
                        pbar.update(1)
        
        if postprocess:
            if len(self.dim) == 3:
                interp = median(interp, disk(kernel_size))
                interp = skimage.morphology.remove_small_objects(interp.astype(bool))
                interp = skimage.morphology.remove_small_holes(interp.astype(bool))
                
                # Resize to original dimensions
                if self.data.resolution != 1:
                    interp = resize(interp, self.data.original_shape)
            
            else:
                new_interp = []
                for z in range(interp.shape[0]):
                    interp[z, :, :] = median(interp[z, :, :], disk(kernel_size))
                    interp[z, :, :] = skimage.morphology.remove_small_objects(interp[z, :, :].astype(bool))
                    interp[z, :, :] = skimage.morphology.remove_small_holes(interp[z, :, :].astype(bool))
                    
                    # Resize to original dimensions
                    if self.data.resolution != 1:
                        new_interp.append(resize(interp[z, :, :], self.data.original_shape))
                if self.data.resolution != 1:
                    interp = np.array(new_interp)
        
        return interp
    
    
    def save(self, filepath: str):
        """
        Saves the ImplicitFunction object to a provided file.
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
    
    
    def _interpolate_point(self, x):
        """
        Helper function to compute interpolated values for a point (x, y, t).
        :param x: An input vector in the form (x, y, t) if 3D, and (x, y, z, t) if 4D.
        :return: interpolated value for the input vector x.
        """
        k = len(self.d)
        # phi(x - c[j]), where phi is the radial basis function r^2 * log(r)
        #print('x:', x.shape)
        #print('c:', self.c.shape)
        #print(self.c)
        phi_xcj = np.sum(np.square(x - self.c), axis=1)
        phi_xcj = np.array([r2 * math.log(math.sqrt(r2)) if r2 != 0 else 0 for r2 in phi_xcj])
        # sum [ d[j] * phi(x - c[j]) ] + P(x)
        return np.dot(self.d, phi_xcj) + self.p[0] + np.dot(self.p[1:], x)

    
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
        phi_matrix = []
        for i in tqdm(range(k), desc="Creating matrix"):
            ci = c[i, :]
            ci_cj = ci - c
            phi = np.sum(np.square(ci_cj), axis=1)
            phi = np.array([r2 * math.log(math.sqrt(r2)) if r2 != 0 else 0 for r2 in phi])
            phi_matrix.append(phi)
        phi_matrix = np.asarray(phi_matrix)
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