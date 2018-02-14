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
        self.norm_std = 300.
        self.max_range = 8183. # Zmax = 8333
        self.lambdaH=0.001
        
        self.sqrt2=math.sqrt(2.)
        self.div1=self.norm_std*self.sqrt2
        self.map = map_reader

        probShort = 0.00  
        probMax = 0.11
        probRand = 0.11
        probHit = 1. - (probShort + probMax + probRand)
                
        # relative weights of each distribution in final pseudodistribution
        self.scale = (probHit, probShort, probMax, probRand)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        z_real = list()
        for i in range(0, 181, 20): # take every 10th measurement
            real_loc = self.map.raytrace(x_t1, i) #* np.pi / 360
            print(real_loc)
            dist = np.sqrt(np.square(real_loc[0] - x_t1[0]) + np.square(real_loc[1] - x_t1[1]))
            z_real.append(dist)

        q = 0
        p = 1
        for z_r, z_t in zip(z_real, z_t1_arr):
            # define array of probability distributions to combine and sample from
            # prob = (1, 1, 1, 1)
            prob = list()
           
            # Auxiliar local variables
            self.x1=-z_r+self.max_range
            self.x2=(z_t-z_r)/self.norm_std

            up=math.sqrt(2./math.pi)*math.exp(-1./2.*self.x2*self.x2)
            down=self.norm_std*(math.erf(z_r/self.div1)+math.erf(self.x1/self.div1))

            unexpected=math.exp(-self.lambdaH*min(z_t,z_r))/(1.-math.exp(-z_r*self.lambdaH))           
            
            if z_r>=self.max_range: self.flag=1.
            else: self.flag=0.   

            # Probabilities
            prob.append(up/down)
            prob.append(unexpected)
            prob.append(self.flag)
            prob.append(1./self.max_range)
        
            for p, s in zip(prob, self.scale):
                q += p*s  # might need to use log sum here...
            #print(down)
                
            #From Density to Probability
            p=p*q
        return p


if __name__ == '__main__':
    pass
