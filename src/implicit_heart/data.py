import logging
from collections import defaultdict
from copy import deepcopy

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

# Retrieval mode
# cv2.RETR_EXTERNAL
# cv2.RETR_LIST
RET_MODE = cv2.RETR_EXTERNAL

logger = logging.getLogger(__name__)

class Data:
    def __init__(self, I: np.ndarray, tmax: float = 1.0, tlist: list = None, grid_spacing=0.1, resolution=1.0, use_propagation=False):
        """
        Initializes the data object to perform computations on I. 
        :param I: A NumPy array object. This should either be a 2D image, 
                  3D volume, or 4D array of volumes. If I is a 3D volume,
                  the indices should be organized as I[n, y, x], where n
                  is the slice index in the volume, and x and y are the
                  coordinates in the image. A 4D array should be organized
                  as I[t, n, y, x], where t is a volume for a given time-step.
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
                    new_I.append(resize(I[z].copy(), (int(I[z].shape[0] * resolution), int(I[z].shape[1] * resolution))))
            else:  # 4D
                new_I = []
                for t in range(I.shape[0]):
                    curr_t = []
                    for z in range(I.shape[1]):
                        curr_t.append(resize(I[t, z].copy(), (int(I[t, z].shape[0] * resolution), int(I[t, z].shape[1] * resolution))))
                    new_I.append(curr_t)
            I = np.array(new_I)
            #print('min', np.min(I))
            #print('max', np.max(I))
            #print('shape', I.shape)

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
        self._create_constraints(tmax)
        #print('self.constraints')
        #print(self.constraints)
        
        # Make edt distance map
        if len(self.dim) == 3:
            self.dt = edt.edt(self.I)
            self.dt_inv = edt.edt(np.invert(self.I.astype(dtype=bool)))
        else:
            self.dt = []
            self.dt_inv = []
            for i in range(self.dim[0]):
                self.dt.append(edt.edt(self.I[i]))
                self.dt_inv.append(edt.edt(np.invert(self.I[i].astype(dtype=bool))))
            self.dt = np.array(self.dt)
            self.dt_inv = np.array(self.dt_inv)
        
        # Propagate
        if self.use_propagation:
            if len(self.dim) == 3:
                self.propagate()
            else:
                self.propagate4D()

        # 3D interp
        if self.use_grid:
            if len(self.dim) == 3:
                for idx in range(self.dim[0]):
                    if len(self.tlist) > 1:
                        z = self.tlist[idx]
                    else:
                        z = (self.tmax / (self.dim[0] - 1)) * idx
                    self._grid_constraints(idx, z, t=None, spacing=self.grid_spacing)
            # 4D interp
            else:
                for idx in range(self.dim[0]):
                    if self.tmax != 0:
                        if len(self.tlist) > 1:
                            t = self.tlist[idx]
                        else:
                            t = (self.tmax / (self.dim[0] - 1)) * idx
                    else:
                        t = 0
                    for z in range(self.dim[1]):
                        self._grid_constraints(idx, z=z, t=t, spacing=self.grid_spacing)
                

      
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
        
    
    def get_c(self):
        """
        Creates constraint array to return to implicit.py.
        """
        c = []
        if len(self.dim) == 3:
            for i in sorted(self.constraints.keys()):
                curr_constraint = self.constraints[i]
                for c_tuple in curr_constraint:
                    c_pt, _ = c_tuple
                    c.append(c_pt)
        else:
            for i in sorted(self.constraints4D.keys()):
                curr_constraints = self.constraints4D[i]
                for j in sorted(curr_constraints.keys()):
                    curr_constraint = curr_constraints[j]
                    for c_tuple in curr_constraint:
                        c_pt, _ = c_tuple
                        c.append(c_pt)
                        #if len(c_pt) != 4:
                        #    print(c_pt)
        #print(c)
        c = np.array(c)
        logger.debug("Constraint array shape: %s", c.shape)
        return c
        
        
    def get_h(self):
        """
        Creates h array to return to implicit.py
        """
        h = []
        if len(self.dim) == 3:
            for i in sorted(self.constraints.keys()):
                curr_constraint = self.constraints[i]
                for c_tuple in curr_constraint:
                    _, c_type = c_tuple
                    h.append(c_type)
        else:
            for i in sorted(self.constraints4D.keys()):
                curr_constraints = self.constraints4D[i]
                for j in sorted(curr_constraints.keys()):
                    curr_constraint = curr_constraints[j]
                    for c_tuple in curr_constraint:
                        _, c_type = c_tuple
                        h.append(c_type)
        h = np.array(h)
        h = h[:, np.newaxis]
        return h
    

    def propagate4D(self):
        original_I = self.I.copy()
        original_dim = deepcopy(self.dim)
        original_tmax = self.tmax
        for t in sorted(self.constraints4D.keys()):
            self.constraints = self.constraints4D[t]
            self.I = original_I[t].copy()
            self.dim = self.I.shape
            self.tmax = self.dim[0] - 1
            if len(self.tlist) > 1:
                curr_t = self.tlist[t]
            else:
                curr_t = t
            self.propagate(timestep_t=curr_t)
            self.constraints4D[t] = deepcopy(self.constraints)
        self.I = original_I
        self.dim = original_dim
        self.tmax = original_tmax
        

    def propagate(self, timestep_t = None):
        """
        Propagate neighboring constraints.
        """
        min_idx = min(self.constraints.keys())
        max_idx = max(self.constraints.keys())
        # 3D (2d to 2d)
        for i in range(min_idx + 1, max_idx):
            num_slices, _, _ = self.dim
            
            if len(self.tlist) > 1:
                t_step = self.tlist[i]
            else:
                t_step = (self.tmax / (num_slices - 1)) * i
                
            curr_img = self.I[i]
            curr_img_inv = np.invert(self.I[i].astype('bool'))
            dist_curr_img = cv2.distanceTransform(curr_img.astype('uint8'), cv2.DIST_L2, 0)
            dist_curr_img_inv = cv2.distanceTransform(curr_img_inv.astype('uint8'), cv2.DIST_L2, 0)
            
            logger.debug("Propagating constraints into slice %d", i)
            prev_constraints = self.constraints[i-1]
            curr_constraints = self.constraints[i]
            next_constraints = self.constraints[i+1]
            for pc_tuple in prev_constraints:
                pt, _ = pc_tuple
                pt_dist = dist_curr_img[round(pt[1]), round(pt[0])]
                if pt_dist != 0:
                    pt_dist = pt_dist - 1
                else:
                    pt_dist = -(dist_curr_img_inv[round(pt[1]), round(pt[0])] - 1)
                
                if timestep_t is not None:
                    pt = np.array([pt[0], pt[1], i, timestep_t])
                    #pt = np.array([pt[0], pt[1], t_step, timestep_t])
                else:
                    pt = np.array([pt[0], pt[1], t_step])

                # Check for duplicates
                is_in = False
                for c_tuple in curr_constraints:
                    c, _ = c_tuple
                    if np.array_equal(c, pt):
                        is_in = True
                        break
                if not is_in:
                    self.constraints[i].append((pt, pt_dist))
            for nc_tuple in next_constraints:
                pt, _ = nc_tuple
                pt_dist = dist_curr_img[round(pt[1]), round(pt[0])]
                if pt_dist != 0:
                    pt_dist = pt_dist - 1
                else:
                    pt_dist = -(dist_curr_img_inv[round(pt[1]), round(pt[0])] - 1)
                if timestep_t is not None:
                    pt = np.array([pt[0], pt[1], i, timestep_t])
                    #pt = np.array([pt[0], pt[1], t_step, timestep_t])
                else:
                    pt = np.array([pt[0], pt[1], t_step])

                # Check for duplicates
                is_in = False
                for c_tuple in curr_constraints:
                    c, _ = c_tuple
                    if np.array_equal(c, pt):
                        is_in = True
                        break
                if not is_in:
                    self.constraints[i].append((pt, pt_dist))

    

        
    def _create_constraints(self, t: float) -> (np.ndarray, np.ndarray):
        """
        Creates boundary and normal constraints for a given input image I.
        :param t: for a 2D image, t is the embedded 3D coordinate value.
                  for a 3D and 4D volume, t is t_max, or the maximum 
                  separation distance between the first index and the
                  last index.
        :return: two NumPy arrays containing the boundary and normal constraints, respectively.
        """
        # I is a 3D volume.
        if len(self.dim) == 3:
            b, n = self._create_constraints3D(t)
        # I is a 4D series of volumes under time lapses.
        elif len(self.dim) == 4:
            b, n = self._create_constraints4D()
        else:
            raise Exception("Invalid dimensions: " + str(self.dim))
        #return self._normalize(b, n)
        return b, n
            
            
    def _normalize(self, b, n):
        h, w = self.dim[-2:]
        b[:, 0] = b[:, 0] / (w - 1)
        b[:, 1] = b[:, 1] / (h - 1)
        n[:, 0] = n[:, 0] / (w - 1)
        n[:, 1] = n[:, 1] / (h - 1)
        if len(self.dim) == 4:
            num_slices = self.dim[-3]
            b[:, 2] = b[:, 2] / (num_slices - 1)
            n[:, 2] = n[:, 2] / (num_slices - 1)
        return b, n
            
            
    def _get_normal_point(self, img: np.ndarray, pt: np.ndarray, dist: float = 0.9) -> np.ndarray:
        """
        Creates the normal point from a given boundary, based on the gradient
        of an image. 
        :param img: The image to create a normal point in
        :param pt: the boundary point to make a normal point from
        :param dist: the distance the normal point is away from the boundary
        :return: a NumPy array containing the normal point
        """
        sobelx = cv2.Scharr(img.astype('uint8'), cv2.CV_64F, dx=1, dy=0)
        sobely = cv2.Scharr(img.astype('uint8'), cv2.CV_64F, dx=0, dy=1)
        #sobelx = cv2.Sobel(img.astype('uint8'), cv2.CV_64F, dx=1, dy=0, ksize=1)
        #sobely = cv2.Sobel(img.astype('uint8'), cv2.CV_64F, dx=0, dy=1, ksize=1)
        grad_x = sobelx[pt[1], pt[0]]
        grad_y = sobely[pt[1], pt[0]]
        grad = np.array([grad_x, grad_y])
        
        # If no gradient can be found
        if np.linalg.norm(grad) == 0:
            logger.warning("Sobel gradient is zero at %s; skipping its normal constraints.", pt)
            #plt.imshow(img); plt.show()
            #norm = pt
            #outside = pt
            return [], []
        else:
            grad = grad / np.linalg.norm(grad)
            norm = pt + dist * grad
            outside = pt - dist * grad
        return norm, outside


    def _create_constraints2D(self, idx: int, z: float, time_t: float = None) -> (np.ndarray, np.ndarray):
        """ 
        Creates boundary and normal constraints for a given 2D image I.
        :param z: the third coordinate value to embed coordinates in 3D
        :return: two NumPy arrays containing the boundary and normal constraints, respectively.
        """
        img = self.I[idx]
        contours, _ = cv2.findContours(img.astype('uint8'), RET_MODE, APPROX)
        # If boundary constraints can be generated from the image slice.
        if contours:
            boundary_constraints = []
            normal_constraints = []
            # Collect every component's contour and save into boundary_constraints.
            for i in range(len(contours)):
                component = contours[i][:, 0, :]
                for pt in component:
                    boundary_constraints.append(pt)
            boundary_constraints = np.array(boundary_constraints)
            #normal_constraints = [self._get_normal_point(img, b) for b in boundary_constraints]
            
            # Create normal constraints
            for b in boundary_constraints:
                n, o = self._get_normal_point(img, b)
                if len(n) != 0:
                    normal_constraints.append(n)
                if len(o) != 0:
                    normal_constraints.append(o)
                    
            if len(normal_constraints) != 0:
                normal_constraints = np.array(normal_constraints)
            
            # Append the z and t dimension to the constraints
            z_boundary = np.full((boundary_constraints.shape[0], 1), z)
            
            if len(normal_constraints) != 0:
                z_normal = np.full((normal_constraints.shape[0], 1), z)
                timestep_normal = np.full((normal_constraints.shape[0], 1), time_t)

            timestep_boundary = np.full((boundary_constraints.shape[0], 1), time_t)
            if time_t is not None:  # Timestep coordinate, 3d output
                boundary_constraints = np.concatenate([boundary_constraints, z_boundary, 
                                                       timestep_boundary], axis=1)
                if len(normal_constraints) != 0:
                    normal_constraints = np.concatenate([normal_constraints, z_normal, 
                                                         timestep_normal], axis=1)
            else:  # No timestep coordinate, 2d output
                boundary_constraints = np.concatenate([boundary_constraints, z_boundary], axis=1)
                if len(normal_constraints) != 0:
                    normal_constraints = np.concatenate([normal_constraints, z_normal], axis=1)
            
            # Add to self.constraints
            for b in boundary_constraints:
                self.constraints[idx].append((b, 0))
                
            if len(normal_constraints) != 0:
                for i in range(len(normal_constraints)):
                    n = normal_constraints[i]
                    if i % 2 == 0:  # Inside constraint
                        self.constraints[idx].append((n, 1))
                    else:           # Outside constraint = -1
                        self.constraints[idx].append((n, -1))
            return np.array(boundary_constraints), np.array(normal_constraints)
        else:
            return None, None
    
    
    def _create_constraints3D(self, t_max: float, timestep_t: float = None) -> (np.ndarray, np.ndarray):
        """ 
        Creates boundary and normal constraints for a given 3D volume I.
        :param t_max: the separation distance between the first image slice and the last image slice.
        :return: two NumPy arrays containing the boundary and normal constraints, respectively.
        """
        num_slices, h, w = self.I.shape
        boundary_constraints = []
        normal_constraints = []
        t_step = t_max / (num_slices - 1)
        
        curr_t = 0
        # Go through each slice i and pixel for I(i, x, y) and 
        # generate constraints t_maxif conditions are met.
        for i in range(num_slices):
            #print(i)
            if len(self.tlist) > 1 and self.I.shape == 3:
                curr_t = self.tlist[i]
            elif self.I.shape == 4:
                curr_t = i
            b, n = self._create_constraints2D(i, z=curr_t, time_t=timestep_t)
            # If this index contains boundary constraints
            if b is not None:
                for pt in b:
                    boundary_constraints.append(pt)
                for pt in n:
                    normal_constraints.append(pt)
            curr_t += t_step
        return np.array(boundary_constraints), np.array(normal_constraints)
    
    
    # Test copied from backup
    def _create_constraints4D(self) -> (np.ndarray, np.ndarray):
        """ 
        Creates boundary and normal constraints for a given image I.
        :param t_max: the separation distance between the first image slice and the last image slice.
        :return: two NumPy arrays containing the boundary and normal constraints, respectively.
        """
        curr_I = self.I.copy()
        time_steps, num_slices, _, _ = self.I.shape
        boundary_constraints = []
        normal_constraints = []
        
        for i in range(time_steps):
            self.I = curr_I[i].copy()
            curr_t = i
            if len(self.tlist) > 1:
                curr_t = self.tlist[i]
            b, n = self._create_constraints3D(t_max=num_slices-1, timestep_t=curr_t)
            t = np.full((b.shape[0], 1), curr_t)
            b = np.concatenate([b, t], axis=1)
            t = np.full((n.shape[0], 1), curr_t)
            n = np.concatenate([n, t], axis=1)
            for pt in b:
                boundary_constraints.append(pt)
            for pt in n:
                normal_constraints.append(pt)
                
            # Setup 4D dict
            self.constraints4D[i] = deepcopy(self.constraints)
            self.constraints = defaultdict(list)
            
        self.I = curr_I
        
        # b, z, t
        # n, z, t
        #print(boundary_constraints[0])
        return np.array(boundary_constraints), np.array(normal_constraints)

    def _grid_constraints(self, idx, z, t=None, spacing=0.1):
        """
        Creates a grid of constraints for a given image I.
        """
        w = self.I.shape[-1]
        h = self.I.shape[-2]
        
        if len(self.dim) == 4:
            curr_img = self.I[idx, z].copy()
        else:
            curr_img = self.I[idx].copy()

        grid_constraints = []
        for x in range(0, w, int(w * spacing)):
            for y in range(0, h, int(h * spacing)):
                if t is not None:
                    curr_constraint = [x, y, z, t]
                    #print(idx, z, round(curr_constraint[1]), round(curr_constraint[0]))
                    pt_dist = self.dt[idx, z, round(curr_constraint[1]), round(curr_constraint[0])]
                else:
                    curr_constraint = [x, y, z]
                    pt_dist = self.dt[idx, round(curr_constraint[1]), round(curr_constraint[0])]
                
                # Get the constraint distance from the boundary.
                #pt_dist = self.dt[round(curr_constraint[1]), round(curr_constraint[0])]
                if pt_dist != 0:
                    pt_dist = pt_dist - 1
                else:
                    if t is not None:
                        pt_dist = -self.dt_inv[idx, z, round(curr_constraint[1]), round(curr_constraint[0])] - 1
                    else:
                        pt_dist = -self.dt_inv[idx, round(curr_constraint[1]), round(curr_constraint[0])] - 1
            
                # Sanity check. set default value if distance is negative infinity
                if pt_dist == float('-inf'):
                    pt_dist = -3
                grid_constraints.append((np.array(curr_constraint), pt_dist))
                
        # Check for duplicates
        for grid in grid_constraints:
            pt = grid[0]
            pt_dist = grid[1]
            is_in = False
            for c_tuple in self.constraints[idx]:
                c, _ = c_tuple
                if np.array_equal(c, pt):
                    is_in = True
                    break
            if not is_in:
                if len(self.dim) == 3:
                    self.constraints[idx].append((pt, pt_dist))
                else:
                    self.constraints4D[idx][z].append((pt, pt_dist))
        
        #return np.array(boundary_constraints), np.array(normal_constraints)