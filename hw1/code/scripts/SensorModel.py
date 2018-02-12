import numpy as np
import math
import time
from matplotlib import pyplot as plt
import scipy.stats as stats
import pdb

from MapReader import MapReader


class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, map_reader):
        """
        TODO : Initialize Sensor Model parameters here
        """
        self.norm_std = .1
        self.max_range = 3000 # Zmax = 8333
        self.map = map_reader

        probShort = 0.15
        probMax = 0.01
        probRand = 0.15
        probHit = 1 - (probShort + probMax + probRand)
        # relative weights of each distribution in final pseudodistribution
        self.scale = (probHit, probShort, probMax, probRand)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        z_real = list()
        for i in range(0, 181):
            real_loc = self.map.raytrace(x_t1, i * np.pi / 360)
            dist = np.sqrt(np.square(real_loc[0] - x_t1[0]) + np.square(real_loc[1] - x_t1[1]))
            z_real.append(dist)

        q = 0
        for z_r, z_t in zip(z_real, z_t1_arr):
            # define array of probability distributions to combine and sample from
            # prob = (1, 1, 1, 1)
            prob = list()
            a, b = (0 - z_r) / self.norm_std, (self.max_range - z_r) / self.norm_std
            prob.append(stats.truncnorm(a, b))
            prob.append(stats.truncexpon(z_r))
            prob.append(stats.uniform(loc=self.max_range, scale=100000))
            prob.append(stats.uniform(loc=0, scale=self.max_range))

            for dist, s in zip(prob, self.scale):
                q += s * dist.pdf(z_t)

        return q


if __name__ == '__main__':
    pass