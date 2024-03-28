import numpy as np
from scipy.stats import beta as beta_dist


class WeightedBetaBinomial:
    def __init__(self, alpha, beta, decay_factor):
        self.alpha = alpha
        self.beta = beta
        self.decay_factor = decay_factor
        self.weighted_successes = 0
        self.weighted_trials = 0

    def update(self, successes, trials):
        # Update weighted successes and trials
        self.weighted_successes = (
            self.decay_factor * self.weighted_successes + successes
        )
        self.weighted_trials = (
            self.decay_factor * self.weighted_trials + trials
        )

        # Update Beta distribution parameters
        self.alpha += self.weighted_successes
        self.beta += self.weighted_trials - self.weighted_successes

    def sample(self, size=1):
        return beta_dist.rvs(size, self.alpha, self.beta)


class VaccineAllocationTS:
    def __init__(self, arms, decay_factor, num_samples, reward_function):
        self.arms = arms
        self.weighted_distributions = {
            arm: WeightedBetaBinomial(1, 1, decay_factor) for arm in arms
        }
        self.num_samples = num_samples
        self.reward_function = reward_function

    def sample_arms(self):
        samples = {
            arm: self.weighted_distributions[arm].sample(self.num_samples)
            for arm in self.arms
        }
        return samples

    def update_distributions(self, arm, successes, failures):
        self.weighted_distributions[arm].update(
            successes, successes + failures
        )

    def allocate_vaccines(self, total_vaccines):
        # Sample arms to estimate distribution over the best arm
        samples = self.sample_arms()

        # Calculate rewards for each arm based on simulated outcomes
        rewards = {
            arm: self.reward_function(outcome)
            for arm, outcome in samples.items()
        }

        # Calculate probabilities from maximum rewards
        rewards_flat = np.hstack(
            [reward for _, reward in rewards.items()]
        )  # (self.num_samples, n_arms)
        max_rewards = (
            np.argmax(rewards_flat, axis=-1)
            == np.arange(len(self.arms))[:, None]
        )
        rewards_prob = max_rewards.sum(axis=-1)

        # Allocate vaccines based on the probabilities of each arm being best
        vaccine_allocation = np.random.multinomial(
            total_vaccines, rewards_prob
        )
        return {
            arm: allocation
            for arm, allocation in zip(self.arms, vaccine_allocation)
        }


# Function to simulate outcomes
def simulate_outcomes(prevalence, vaccinated, unvaccinated):
    outcomes = {}
    for arm, p in prevalence.items():
        # Calculate the number of infections and hospitalizations based on prevalence, vaccinated, and unvaccinated
        num_infected_vaccinated = np.random.binomial(
            vaccinated[arm], p / 2
        )  # Assuming half the risk for vaccinated
        num_infected_unvaccinated = np.random.binomial(unvaccinated[arm], p)
        num_infected = num_infected_vaccinated + num_infected_unvaccinated

        # Calculate the number of hospitalizations (just for illustration)
        num_hospitalized = np.random.binomial(
            num_infected, 0.1
        )  # Assuming 10% of infections lead to hospitalization

        outcomes[arm] = {
            "infections": num_infected,
            "hospitalizations": num_hospitalized,
        }
    return outcomes


# Reward function that values hospitalizations as more negative than infections
def reward_function(outcome):
    return -10 * outcome["hospitalizations"] - outcome["infections"]


# Example usage
arms = ["Group A", "Group B"]
decay_factor = 0.9  # Example decay factor
num_samples = (
    1000  # Number of samples used to estimate the probability distribution
)
vaccine_allocation = VaccineAllocationTS(
    arms, decay_factor, num_samples, reward_function
)

# Simulate outcomes and update distributions iteratively
for _ in range(10):
    # Simulate prevalence, vaccinated, and unvaccinated for each arm (example values)
    prevalence = {"Group A": 0.5, "Group B": 0.3}
    vaccinated = {"Group A": 500, "Group B": 300}
    unvaccinated = {"Group A": 5000, "Group B": 7000}

    # Simulate outcomes based on prevalence, vaccinated, and unvaccinated
    outcomes = simulate_outcomes(prevalence, vaccinated, unvaccinated)

    # Update distributions based on observed outcomes
    for arm, outcome in outcomes.items():
        vaccine_allocation.update_distributions(
            arm, outcome["infections"], outcome["hospitalizations"]
        )

    # Allocate vaccines based on updated distributions
    best_vaccine_allocation = vaccine_allocation.allocate_vaccines(1000)
    print("Vaccines allocation:", best_vaccine_allocation)
