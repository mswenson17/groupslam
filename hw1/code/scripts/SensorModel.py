import numpy as np
import math
# from matplotlib import pyplot as plt
# import scipy.stats as stats
# import pdb

# from MapReader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, map_reader):
        """
        TODO : Initialize Sensor Model parameters here
        """
        self.norm_std = 100.
        self.max_range = 8183.  # Zmax = 8333
        self.lambdaH = 100
        self.lambdaH = .005  # 1/10.

        self.sqrt2 = math.sqrt(2.)
        self.div1 = self.norm_std * self.sqrt2
        self.map_reader = map_reader
        self.map = map_reader.get_map()

        self.probShort = 0.195
        self.probMax = 0.005
        self.probRand = 0.5
        self.probOutside = .000
        self.probHit = 1

        probsum = (self.probHit + self.probShort + self.probMax + self.probRand + self.probOutside)

        self.probShort /= probsum
        self.probMax /= probsum
        self.probRand /= probsum
        self.probOutside /= probsum
        self.probHit /= probsum

        print(self.probHit + self.probShort + self.probMax + self.probRand + self.probOutside)
        # self.probHit = 1. - (self.probShort + self.probMax + self.probRand + self.probOutside)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        z_real = list()
        for i in range(0, 181, 20):  # take every 10th measurement 20
            real_loc = self.map_reader.raytrace(x_t1, i)  # * np.pi / 360
            # print(real_loc)
            dist = np.sqrt(np.square(real_loc[0] - x_t1[0]) + np.square(real_loc[1] - x_t1[1]))
            z_real.append(dist)

        q = 0
        lnp = 0
        for z_r, z_t in zip(z_real, z_t1_arr):

            # Auxiliary local variables
            self.x1 = -z_r + self.max_range
            self.x2 = (z_t - z_r) / self.norm_std

            up = math.sqrt(2. / math.pi) * math.exp(-1. / 2. * self.x2 * self.x2)
            down = self.norm_std * (math.erf(z_r / self.div1) + math.erf(self.x1 / self.div1))

            unexpected = self.lambdaH * math.exp(-self.lambdaH * z_t) / (1. - math.exp(-z_r * self.lambdaH))
            if z_t >= z_r:
                unexpected = 0

            if z_t >= self.max_range:
                self.flag = 1.
            else:
                self.flag = 0.
            out_of_map = 1 - self.map[int(x_t1[1]), int(x_t1[0])]
            # print(" hit: " + str(up / down) + " unexpected: " + str(unexpected) + " outside: " + str(out_of_map))

            # Probabilities
            q += (up / down) * self.probHit
            q += unexpected * self.probShort
            q += self.flag * self.probMax
            q += self.probRand / self.max_range
            q += out_of_map * self.probOutside

            # From Density to Probability
            lnp = lnp + math.log(q)
        return lnp  # math.exp(lnp)


if __name__ == '__main__':
    pass
