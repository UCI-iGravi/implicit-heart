
    
"""
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

            prev_constraints = self.constraints[i-1]
            curr_constraints = self.constraints[i]
            next_constraints = self.constraints[i+1]
            for pc_tuple in prev_constraints:
                pt, _ = pc_tuple
                print(self.dt.shape)
                pt_dist = self.dt[i, round(pt[1]), round(pt[0])]
                if pt_dist != 0:
                    pt_dist = pt_dist - 1
                else:
                    pt_dist = -(self.dt_inv[i, round(pt[1]), round(pt[0])] - 1)
                
                # 4D
                if timestep_t is not None:
                    #pt = np.array([pt[0], pt[1], t_step, timestep_t])
                    pt = np.array([pt[0], pt[1], i, timestep_t])
                # 3D
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
                pt_dist = self.dt[i, round(pt[1]), round(pt[0])]
                if pt_dist != 0:
                    pt_dist = pt_dist - 1
                else:
                    pt_dist = -(self.dt_inv[i, round(pt[1]), round(pt[0])] - 1)
                if timestep_t is not None:
                    #pt = np.array([pt[0], pt[1], t_step, timestep_t])
                    pt = np.array([pt[0], pt[1], i, timestep_t])
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
"""