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

    def __init__(self, occupancy_map):
        """
        TODO : Initialize Sensor Model parameters here
        """
        self.norm_std = .1
        self.max_range = 3000

        # relative weights of each distribution in final pseudodistribution
        self.scale = (1, 1, 1, 1)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """
        for z_t in z_t1_arr:
            # define array of probability distributions to combine and sample from
            prob = (1, 1, 1, 1)

            a, b = (0 - z_t) / self.norm_std, (self.max_range - z_t) / self.norm_std
            prob[0] = stats.truncnorm(a, b)
            prob[1] = stats.truncexp(z_t)



        return q


if __name__ == '__main__':
    pass
