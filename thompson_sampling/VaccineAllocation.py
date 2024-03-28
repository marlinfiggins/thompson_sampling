from .UtilityTS import BetaBinomial, ThompsonSampling
import numpy as np


N = 1000  # Population size in each arm
K = 50  # Number of vaccines available per day
days = 30  # Simulate for 30 days
n_arms = 2  # Number of arms

# Initialize arms with different base infection risks
arms = [[BetaBinomial(alpha=1, beta=1)] for _ in range(n_arms)]
at_risk = [N for _ in range(n_arms)]
base_infection_risks = [0.01, 0.02]  # Base infection risks for each arm
base_hospitalization_risk = 0.1  # Base hospitalization risk


def utility_function(outcomes) -> float:
    # Assuming outcomes are structured as [infections, hospitalizations]
    infections, hospitalizations = outcomes
    # Negative impact to reflect that we want to minimize these
    return -(infections + hospitalizations)


ts = ThompsonSampling(arms=arms, utility_function=utility_function)

# Track outcomes for each arm
infections = np.zeros((days, n_arms))
hospitalizations = np.zeros((days, n_arms))

for day in range(days):
    vaccine_allocation = ts.select_arms(K)

    for arm_index in range(n_arms):
        vaccinated = vaccine_allocation[arm_index]
        base_infection_risk = base_infection_risks[arm_index]

        # Simulate infections, half risk if vaccinated
        infected_unvaccinated = np.random.binomial(
            N - vaccinated, base_infection_risk
        )
        infected_vaccinated = np.random.binomial(
            vaccinated, base_infection_risk / 2
        )
        total_infected = infected_unvaccinated + infected_vaccinated

        # Simulate hospitalizations, half risk if vaccinated
        hospitalization_unvaccinated = np.random.binomial(
            infected_unvaccinated, base_hospitalization_risk
        )
        hospitalization_vaccinated = np.random.binomial(
            infected_vaccinated, base_hospitalization_risk / 2
        )
        total_hospitalizations = (
            hospitalization_unvaccinated + hospitalization_vaccinated
        )

        infections[day, arm_index] = total_infected
        hospitalizations[day, arm_index] = total_hospitalizations

        # Update arm based on observed outcomes
        arms[arm_index][0].update(total_infected, N)
        arms[arm_index][1].update(total_hospitalizations, N)

    print(f"Day {day + 1}:")
    for arm_index in range(n_arms):
        print(
            f"  Arm {arm_index + 1}: Vaccinated = {vaccine_allocation[arm_index]}, Infections = {infections[arm_index]}, Hospitalizations = {hospitalizations[arm_index]}"
        )
