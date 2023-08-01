import numpy as np
from scipy.stats import norm as Normal


class GaussianRiskyBandits:
    def __init__(self, mu, sigma, alpha):
        self.mu = mu
        self.sigma = sigma
        self.dist = [Normal(m, s) for m, s in zip(self.mu, self.sigma)]
        self.alpha = alpha

    def sample(self, arm):
        return self.dist[arm].rvs()


class ThompsonSamplingGRB:
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self):
        pass
