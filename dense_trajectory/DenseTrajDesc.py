class DenseTrajDesc:

    def read(self,line):
        # frameNum mean_x mean_y var_x var_y length scale x_pos y_pos t_pos Trajectory HOG HOF MBHx MBHy
        #The first 10 elements are information about the trajectory:
        D = line.split()

        self.frameNum = float(D[0]) #     The trajectory ends on which frame
        self.mean_x = float(D[1]) #       The mean value of the x coordinates of the trajectory
        self.mean_y = float(D[2]) #       The mean value of the y coordinates of the trajectory
        self.var_x = float(D[3]) #        The variance of the x coordinates of the trajectory
        self.var_y = float(D[4]) #        The variance of the y coordinates of the trajectory
        self.length = float(D[5]) #       The length of the trajectory
        self.scale = float(D[6]) #        The trajectory is computed on which scale
        self.x_pos = float(D[7]) #        The normalized x position w.r.t. the video (0~0.999), for spatio-temporal pyramid 
        self.y_pos = float(D[8]) #        The normalized y position w.r.t. the video (0~0.999), for spatio-temporal pyramid 
        self.t_pos = float(D[9]) #        The normalized t position w.r.t. the video (0~0.999), for spatio-temporal pyramid
#The following element are five descriptors concatenated one by one:
        traj_len_dim = 30
        hog_dim = 96
        hof_dim = 108
        mbhx_dim = 96
        mbhy_dim = 96
        desc_start_idx = 10
        hog_start_idx = desc_start_idx + traj_len_dim
        hof_start_idx = hog_start_idx + hog_dim
        mbhx_start_idx = hof_start_idx + hof_dim
        mbhy_start_idx = mbhx_start_idx + mbhx_dim
        desc_end_idx = mbhy_start_idx + mbhy_dim

        self.Trajectory = map(float, D[desc_start_idx:hog_start_idx]) #    2x[trajectory length] (default 30 dimension) 
        self.HOG = map(float, D[hog_start_idx:hof_start_idx])         #    8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
        self.HOF =  map(float, D[hof_start_idx:mbhx_start_idx])       #    9x[spatial cells]x[spatial cells]x[temporal cells] (default 108 dimension)
        self.MBHx = map(float, D[mbhx_start_idx:mbhy_start_idx ])     #    8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
        self.MBHy = map(float, D[mbhy_start_idx:desc_end_idx])        #    8x[spatial cells]x[spatial cells]x[temporal cells] (default 96 dimension)
       
        return self
