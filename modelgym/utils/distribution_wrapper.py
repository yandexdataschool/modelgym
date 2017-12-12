

class DistributionWrapper(object):
    """
    Wrapper for scipy.stats distributions
    """

    def __init__(self, distribution, params):
        """
        :param distribution (object): distribution from scipy.stats
        :param params (dict): parameters of distribution
        """
        self.distribution = distribution
        self.params = params

    def sample(self, size=1):
        """
        :param size: number of samples to return
        :return: float or np.array(float)
        """
        if size == 1:
          return self.distribution.rvs(size=size, **self.params)[0]
        return self.distribution.rvs(size=size, **self.params)