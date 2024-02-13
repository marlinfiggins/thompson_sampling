import numpy as np
from scipy.stats import beta as Beta
from scipy.stats import bernoulli as Bernoulli


class BernoulliBandit:
    def __init__(self, probs):
        self.probs = probs
        self.reward_dist = [Bernoulli(p) for p in self.probs]

    def pull_arm(self, arm):
        return self.reward_dist[arm].rvs()


class EpsilonGreedy:
    """
    Epsilon-greedy agent for solving multi-armed bandit problems.

    Parameters:
    - n_arms (int): The number of arms in the bandit.
    - epsilon (float): Probability of selecting a random arm for exploration.
    """

    def __init__(self, n_arms: int, eps: float):
        self.n_arms = n_arms
        self.eps = eps
        self.expected_reward = np.zeros(n_arms)
        self.times_chosen = np.zeros(n_arms)

    def update_expected_rewards(self, arm: int, reward: float):
        self.times_chosen[arm] += 1

        # Online update for the mean reward
        self.expected_reward[arm] += (
            reward - self.expected_reward[arm]
        ) / self.times_chosen[arm]
        return None

    def choose_arm(self):
        # Explore with probability epsilon
        if np.random.uniform() < self.eps:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.expected_reward)


class UCB:
    """
    UCB (Upper Confidence Bound) agent for solving multi-armed bandit problems.

    Parameters:
    - c (float): Controls the balance between exploration and exploitation.
    - n_arms (int): The number of arms in the bandit.
    """

    def __init__(self, n_arms: int, c: float):
        self.n_arms = n_arms
        self.c = c
        self.expected_reward = np.zeros(n_arms)
        self.times_chosen = np.zeros(n_arms)
        self.total_actions = 0

    def update_expected_rewards(self, arm, reward):
        # Update Q-value for the pulled arm
        self.times_chosen[arm] += 1
        # Online update for the expected reward
        self.expected_reward[arm] += (
            reward - self.expected_reward[arm]
        ) / self.times_chosen[arm]

    def choose_arm(self):
        ucb_values = self.expected_reward + self.c * np.sqrt(
            np.log(self.total_actions + 1) / (self.times_chosen + 1e-4)
        )
        return np.argmax(ucb_values)


class ThompsonSampling:
    """
    Thompson Sampling agent for solving multi-armed bandit problems.

    Parameters:
    - n_arms (int): The number of arms in the bandit.

    Attributes:
    - n_arms (int): The number of arms in the bandit.
    - alpha (float): Alpha parameters of the beta distribution for each arm.
    - beta (float): Beta parameters of the beta distribution for each arm.
    """
    def __init__(
        self,
        n_arms: int,
        alpha=1.0,
        beta=1.0,
    ):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha
        self.beta = np.ones(n_arms) * beta  # Prior alpha and beta

    @property
    def _theta(self):
        return [Beta(self.alpha[k], self.beta[k]) for k in range(self.n_arms)]

    def update_expected_rewards(self, arm, reward):
        if reward == 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        return None

    def choose_arm(self):
        # Draw posterior prob of success
        theta_sample = np.array([theta_rv.rvs() for theta_rv in self._theta])

        # Select highest theta arm
        return np.argmax(theta_sample)
