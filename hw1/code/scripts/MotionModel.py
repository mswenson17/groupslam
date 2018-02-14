import numpy as np
import math
# import sys
# import pdb


class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):
        """
        TODO : Initialize Motion Model parameters here
        """
        self.alpha_1 = 0.00001  # rotation
        self.alpha_2 = 0.00001  # rotation
        self.alpha_3 = 0.5  # linear
        self.alpha_4 = 0.5  # linear

        # self.alpha_1 = 0.0  # rotation
        # self.alpha_2 = 0.0  # rotation
        # self.alpha_3 = 0.0  # linear
        # self.alpha_4 = 0.0  # linear

        self.mu = 0.0  # zero mean noise for sampling

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        # [Chapter 5, Page 122]
        # pdb.set_trace()

        # find relative change in odometry since last measurement
        delta_rot1 = math.atan2((u_t1[1] - u_t0[1]) % math.pi, (u_t1[0] - u_t0[0]) % math.pi) - u_t0[2]
        delta_trans = math.sqrt((u_t0[0] - u_t1[0])**2 + (u_t0[1] - u_t1[1])**2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        # update previous position based on odometry and uncertainty
        sigma_delta_rot1 = math.sqrt(self.alpha_1 * abs(delta_rot1) + self.alpha_2 * abs(delta_trans))
        _delta_rot1 = delta_rot1 - np.random.normal(self.mu, sigma_delta_rot1)

        if ~(u_t0[0:3] == u_t1[0:3]).all():
            sigma_delta_trans = math.sqrt(self.alpha_3 * abs(delta_trans) + self.alpha_4 * abs(delta_rot1 + delta_rot2))
            # print(sigma_delta_trans)
        else:
            sigma_delta_trans = 0.001
            # print(delta_trans)
        _delta_trans = delta_trans - np.random.normal(self.mu, sigma_delta_trans)
        # print((_delta_trans - delta_trans, _delta_rot1 - delta_rot1))
        sigma_delta_rot2 = math.sqrt(self.alpha_1 * abs(delta_rot2) + self.alpha_2 * abs(delta_trans))
        _delta_rot2 = delta_rot2 - np.random.normal(self.mu, sigma_delta_rot2)

        # assign to updated odometry
        x = x_t0[0] + _delta_trans * math.cos(x_t0[2] + _delta_rot1)  # rad
        y = x_t0[1] + _delta_trans * math.sin(x_t0[2] + _delta_rot1)
        theta = x_t0[2] + _delta_rot1 + _delta_rot2

        # y = x_t0[1] + u_t1[1] - u_t0[1]
        # x = x_t0[0] + u_t0[0] - u_t1[0]

        x_t1 = [x, y, theta]

        # print(x_t1- x_t0)

        return x_t1


if __name__ == "__main__":
    pass
