from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import beta, invgamma, norm, gamma
from typing import Callable


class ConjugateDistribution(ABC):
    @abstractmethod
    def update(self, data) -> None:
        pass

    @abstractmethod
    def sample(self, num_samples: int) -> np.ndarray:
        pass


class BetaBinomial(ConjugateDistribution):
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def update(self, successes: int, trials: int) -> None:
        self.alpha += successes
        self.beta += trials - successes

    def sample(self, num_samples: int) -> np.ndarray:
        return beta(self.alpha, self.beta).rvs(size=num_samples)


class WeightedBetaBinomial(ConjugateDistribution):
    def __init__(self, alpha: float, beta: float, decay_factor: float = 0.9):
        self.alpha = alpha
        self.beta = beta
        self.decay_factor = decay_factor

    def update(self, successes: int, trials: int) -> None:
        # Apply decay factor to existing parameters
        self.alpha *= self.decay_factor
        self.beta *= self.decay_factor

        # Update parameters with new data
        self.alpha += successes
        self.beta += trials - successes

    def sample(self, num_samples: int) -> np.ndarray:
        return beta(self.alpha, self.beta).rvs(num_samples)


class NormalInverseGamma(ConjugateDistribution):
    def __init__(self, mu, lambda_, alpha, beta):
        self.mu = mu
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta

    def update(self, data):
        n = len(data)
        sample_mean = np.mean(data)
        self.mu = (self.lambda_ * self.mu + n * sample_mean) / (
            self.lambda_ + n
        )
        self.lambda_ += n
        self.alpha += n / 2
        self.beta += 0.5 * np.sum((data - sample_mean) ** 2) + (
            n * self.lambda_ * (sample_mean - self.mu) ** 2
        ) / (2 * (self.lambda_ + n))

    def sample(self, num_samples: int):
        sampled_variance = invgamma(self.alpha, scale=self.beta).rvs(
            size=num_samples
        )
        sampled_mean = norm(
            self.mu, np.sqrt(sampled_variance / self.lambda_)
        ).rvs()
        return sampled_mean, sampled_variance


class GammaPoisson(ConjugateDistribution):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def update(self, data):
        self.alpha += np.sum(data)
        self.beta += len(data)

    def sample(self, num_samples: int):
        return gamma(self.alpha, scale=1 / self.beta).rvs(size=num_samples)


UtilityFunction = Callable[[list[np.ndarray]], float]
Outcomes = list[ConjugateDistribution]


class ThompsonSampling:
    def __init__(
        self,
        arms: list[Outcomes],
        utility_function: UtilityFunction,
    ):
        self.arms: list[Outcomes] = arms
        self.utility_function = utility_function

    def sample_outcomes_for_arm(
        self, arm_distributions: Outcomes, num_samples: int
    ):
        return [dist.sample(num_samples) for dist in arm_distributions]

    def compute_rewards(self, num_samples: int):
        rewards = [
            self.utility_function(
                self.sample_outcomes_for_arm(arm, num_samples)
            )
            for arm in self.arms
        ]
        return np.stack(rewards, axis=-1)

    def select_arm(self):
        rewards = self.compute_rewards(num_samples=1)
        return np.argmax(rewards, axis=0)

    def select_arms(self, K: int, num_samples: int = 100):
        rewards = self.compute_rewards(num_samples)
        best_arms = np.argmax(rewards, axis=-1)  # Find best arm by sample
        counts = np.bincount(best_arms, minlength=len(self.arms))  # Count best
        probabilities = counts / num_samples
        vaccine_allocation = np.round(probabilities * K).astype(int)
        return vaccine_allocation

    def update_arm(self, arm_index: int, outcomes) -> None:
        for dist_index, outcome in enumerate(outcomes):
            self.arms[arm_index][dist_index].update(outcome)
