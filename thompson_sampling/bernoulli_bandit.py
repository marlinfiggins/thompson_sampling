import numpy as np
from scipy.stats import beta as Beta
from scipy.stats import bernoulli as Bernoulli


class BernoulliBandit:
    def __init__(self, probs):
        self.probs = probs
        self.reward_dist = [Bernoulli(p) for p in self.probs]

    def sample(self, k):
        return self.reward_dist[k].rvs()


class ThompsonSampling:
    def __init__(
        self,
        n_arms,
        alpha=1.0,
        beta=1.0,
    ):
        self.n_arms = n_arms
        self.alpha, self.beta = alpha, beta  # Prior alpha and beta

    def get_theta(self, successes, failures):
        return [
            Beta(self.alpha + successes[k], self.beta + failures[k])
            for k in range(self.n_arms)
        ]

    def sample(self, successes, failures):
        # Update theta given successes and failures
        self.theta = self.get_theta(successes, failures)

        # Draw posterior prob of success
        theta_sample = np.array([theta_rv.rvs() for theta_rv in self.theta])

        # Select highest theta arm
        selected_arm = np.argmax(theta_sample)

        return selected_arm
