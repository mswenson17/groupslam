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

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        M = len(X_bar)
        wt = np.array(X_bar[:,3])

        if sum(wt) != 1.0:
            wt = wt/sum(wt) # Normalize to 1

        X_bar_resampled = np.empty([M, 4])
        r = np.random.uniform(0, 1/float(M))
        c = wt[0]
        i = 0

        for m in range(0, M):
            u = r + (m-1)*1/float(M)

            while u > c:
                i = i + 1
                c = c + wt[i]

            X_bar_resampled[m,:] = X_bar[i]

        return X_bar_resampled

if __name__ == "__main__":
    pass