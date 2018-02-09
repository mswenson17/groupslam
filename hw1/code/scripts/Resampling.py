import numpy as np
import pdb

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

        self.M = 30 # number of paticles to sample

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """
        # pdb.set_trace()

        wt = X_bar[:,3]

        X_bar_resampled = np.empty([self.M, 4])
        r = np.random.uniform(0, 1/self.M) # 0 to M^-1
        c = wt[0]
        i = 1
        for m in range(0, self.M):
            u = r + (m - 1) * self.M**-1
            while u > c:
                i = i + 1
                c = c + wt[i]
            X_bar_resampled[m,:] = X_bar[i]

        # pdb.set_trace()

        return X_bar_resampled

if __name__ == "__main__":
    pass