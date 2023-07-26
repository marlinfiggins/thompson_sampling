import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as Normal
from scipy.stats import bernoulli as Bernoulli

FeatureWeights = np.ndarray  # Should have shape (n_features, n_arms)


class ContextualBernoulli:
    def __init__(self, weights: FeatureWeights):
        self.weights = weights
        self.n_feature, self.n_arms = self.weights.shape

    def theta(self, arm, features):
        if features.ndim == 1:
            _features = features[None, :]
        else:
            _features = features

        eta = np.dot(_features, self.weights[:, arm])  # For given feature
        return 1.0 / (1.0 + np.exp(-eta))

    def sample(self, arm, features):
        # Feature values for logistic regression
        prob = self.theta(arm, features)
        return Bernoulli(prob).rvs()


class BayesOnlineLogistic:
    def __init__(self, n_features, lam):
        self.n_features = n_features
        self.lam = lam

        # Initilize parameters to be estimated
        self.means = np.zeros(self.n_features)
        self.precisions = np.ones(self.n_features) * self.lam

        # Sample initial weights from prior
        self.weights = self.sample_weights()

    @property
    def weight_dists(self):
        return [Normal(m, 1 / q) for m, q in zip(self.means, self.precisions)]

    def sample_weights(self):
        return np.random.normal(
            self.means, np.power(self.precisions, -1), size=self.n_features
        )

    def loss(self, weights, features, rewards):
        reg_term = (
            0.5 * (self.precisions * np.square(weights - self.means)).sum()
        )
        eta = np.dot(features, weights)
        log_lik = np.log(1.0 + np.exp(eta)) - rewards * eta
        return reg_term + log_lik.sum()

    def grad(self, weights, features, rewards):
        # TODO: Fix gradient
        reg_term = self.precisions * (weights - self.means)
        log_like = (
            features
            * np.power(1.0 + np.exp(np.dot(features, weights)), -1)[:, None]
            - rewards[:, None] * features
        ).sum(axis=0)
        return reg_term + log_like

    def fit(self, features, rewards):

        # Fit weights and set to means
        self.minimizer = minimize(
            lambda w: self.loss(w, features, rewards),
            x0=self.weights,
        )
        self.weights = self.minimizer.x
        self.means = self.weights

        # Update precisions
        p = np.power(1 + np.exp(-np.dot(features, self.weights)), -1)
        self.precisions += np.dot(p * (1 - p), np.square(features))

    def predict_expect(self, features):
        return np.power(1 + np.exp(-np.dot(features, self.means)), -1)

    def predict(self, features):
        self.weights = self.sample_weights()
        return np.power(1 + np.exp(-np.dot(features, self.weights)), -1)


class ThompsonCB:
    def __init__(
        self,
        n_arms,
        n_features,
        lam,
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.lam = lam
        self.thetas = [
            BayesOnlineLogistic(self.n_features, self.lam)
            for _ in range(self.n_arms)
        ]

    def sample(self, features):
        # Draw posterior prob of success
        theta_sample = np.array(
            [post.predict(features) for post in self.thetas]
        )

        # Select highest theta arm
        selected_arm = np.argmax(theta_sample)

        return selected_arm
