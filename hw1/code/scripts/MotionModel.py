import sys
import numpy as np
import math


class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):
        """
        TODO : Initialize Motion Model parameters here
        """
        self.sigma = 1
        self.mu = 1

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        TODO : Add your code here
        """
        # print x_t0

        noise = self.sigma * np.random.randn(1, 3) + self.mu

        # print noise[0]

        x_t1 = x_t0 + u_t1 + noise[0]
        # print x_t1
        return x_t1


if __name__ == "__main__":
    pass
