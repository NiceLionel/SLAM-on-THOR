import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *
from bresenham import bresenham


import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        # s.occupied_prob_thresh = 0.6
        # s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))
        s.occupied_prob_thresh = 0.6
        s.free_prob_thresh = 0.2
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))
        s.log_odds_free = np.log(s.free_prob_thresh/(1-s.free_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        cell_indices = np.zeros((2,x.shape[0]))
        cell_indices[0,:] = np.ceil((np.clip(x, s.xmin, s.xmax) - s.xmin)/s.resolution).astype(int) #round up the index
        cell_indices[1,:] = np.ceil((np.clip(y, s.ymin, s.ymax) - s.ymin)/s.resolution).astype(int)
        return cell_indices

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = 1e-8*np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)
        s.largest_weight = s.map.grid_cell_from_xy(np.array([0]), np.array([0])).reshape(2,1)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        num_particles = w.shape[0]
        p_new_loc = np.zeros(p.shape)
        w_cumsum = np.cumsum(w) / np.sum(w)
        for i in range(num_particles):
            list_index = list(np.random.rand() <  w_cumsum)
            true_index = list_index.index(True)
            p_new_loc[:,i] = p[:,true_index]
        w_new_loc = np.ones(num_particles)/float(num_particles)
        return p_new_loc, w_new_loc

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        ray_num = angles.shape[0]
        world_coord = np.zeros((2,ray_num))
        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        
        d = np.clip(d, s.lidar_dmin, s.lidar_dmax)
        d[d == s.lidar_dmin] = 0
        d[d == s.lidar_dmax] = 0
        
        # 1. lidar measurement along x, y direction under the LiDAR frame
        dist_x = d * np.cos(angles)
        dist_y = d * np.sin(angles)
        lidar_point= np.vstack((dist_x, dist_y, np.zeros(ray_num), np.ones(ray_num))) # [x, y, 0, 1]
        # 2. from LiDAR frame to the body frame
        lidar_to_body = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, s.lidar_height]))
        # 3. from body frame to world frame
        body_to_world = euler_to_se3(0, 0, p[2], np.array([p[0], p[1], s.head_height]))
        transformed_lidar_point = body_to_world @ lidar_to_body @ lidar_point
        # avoid the ground to be detected
        non_ground_index = transformed_lidar_point[2,:] > 0.1 #only keep the point that is above ground 0.1m
        world_coord[0:2, non_ground_index] = transformed_lidar_point[0:2,non_ground_index]
        return world_coord
    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        else:
            # relative movement in local frame (odom is in global frame)
            pose1 = s.lidar[t]['xyth']
            pose2 = s.lidar[t-1]["xyth"]
            pose1[-1] = s.lidar[t]['rpy'][2]
            pose2[-1] = s.lidar[t-1]['rpy'][2]
            control = smart_minus_2d(pose1,pose2)
            return control

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        control1 = s.get_control(t)
        control = deepcopy(control1)
        num_particles = s.n
        sigma_x = s.Q[0, 0]
        sigma_y = s.Q[1, 1]
        sigma_yaw = s.Q[2, 2]
        for i in range(num_particles):
            Q = np.array([np.random.normal(0, sigma_x), np.random.normal(0, sigma_y), np.random.normal(0, sigma_yaw)])
            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            s.p[:, i] = smart_plus_2d(s.p[:, i], Q)
            
            
    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        w = w * np.exp(obs_logp.reshape(w.shape[0],))
        weights = w / w.sum()
        return weights

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        num_particles = s.n
        head_angle = s.joint["head_angles"][joint_name_to_index["Head"],s.find_joint_t_idx_from_lidar(s.lidar[t]["t"])]
        neck_angle = s.joint["head_angles"][joint_name_to_index["Neck"],s.find_joint_t_idx_from_lidar(s.lidar[t]["t"])]
        logP_observ = np.zeros(num_particles)
        for i in range(num_particles):
            world_frame_coord = s.rays2world(s.p[:, i], s.lidar[t]["scan"], head_angle, neck_angle, s.lidar_angles)
            observ = s.map.grid_cell_from_xy(world_frame_coord[0,:], world_frame_coord[1,:]).astype(int)
            observ = np.unique(observ, axis = 1)
            observ_map = (s.map.cells[observ[1,:], observ[0,:]] == 1).astype(int)
            logP_observ[i] = np.sum(observ_map)
        # update weight and map
        s.w = s.update_weights(s.w,logP_observ)
        index = np.argmax(s.w)
        world_frame_coord = s.rays2world(s.p[:, i], s.lidar[t]["scan"], head_angle, neck_angle, s.lidar_angles)
        particle_pos = s.map.grid_cell_from_xy(np.array([s.p[0,index]]), np.array([s.p[1,index]])).astype(int)
        s.largest_weight = np.hstack((s.largest_weight, particle_pos.reshape(2,1))).astype(int)
        observ = s.map.grid_cell_from_xy(world_frame_coord[0,:], world_frame_coord[1,:]).astype(int)
        observ = np.unique(observ, axis = 1)
        observ_map = np.zeros(s.map.cells.shape)
        # only keep the ray on the cell
        empty_cell = np.array([[particle_pos[0,0], particle_pos[1,0]]])
        for i in range(observ.shape[1]):
            point_on_the_ray = np.array(list(bresenham(particle_pos[0,0], particle_pos[1,0], observ[0,i], observ[1,i])))
            empty_cell = np.vstack((empty_cell, point_on_the_ray))
        empty_cell = np.unique(empty_cell, axis = 0)
        observ_map[empty_cell[:,1], empty_cell[:,0]] = -1

        observ_map[particle_pos[1,0], particle_pos[0,0]] = 0
        observ_map[observ[1, :], observ[0, :]] = 1

        map_log_occu = (observ_map == 1) * s.lidar_log_odds_occ
        map_log_empty = (observ_map == -1) * s.lidar_log_odds_free
        
        s.map.log_odds = np.clip(s.map.log_odds + map_log_occu + map_log_empty, -s.map.log_odds_max, s.map.log_odds_max)
        s.map.cells = (s.map.log_odds >= s.map.log_odds_thresh).astype(int)

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
            
