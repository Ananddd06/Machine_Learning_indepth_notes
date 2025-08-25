import numpy as np

class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Bernoulli Naive Bayes Classifier
        :param alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.phi_y = None          # P(y=1)
        self.phi_j_y1 = None       # P(x_j=1|y=1)
        self.phi_j_y0 = None       # P(x_j=1|y=0)

    def fit(self, X, y):
        """
        Fit the Bernoulli Naive Bayes model.
        :param X: Binary feature matrix (n_samples, n_features)
        :param y: Labels (0 or 1)
        """
        n_samples, n_features = X.shape

        # Prior probability of y=1
        self.phi_y = np.mean(y)

        # For y=1
        X_y1 = X[y == 1]
        self.phi_j_y1 = (np.sum(X_y1, axis=0) + self.alpha) / (X_y1.shape[0] + 2*self.alpha)

        # For y=0
        X_y0 = X[y == 0]
        self.phi_j_y0 = (np.sum(X_y0, axis=0) + self.alpha) / (X_y0.shape[0] + 2*self.alpha)

    def _likelihood(self, x, y_val):
        """
        Compute likelihood P(x|y)
        :param x: Feature vector
        :param y_val: Class value (0 or 1)
        """
        if y_val == 1:
            phi = self.phi_j_y1
        else:
            phi = self.phi_j_y0

        # Bernoulli likelihood: product over j of phi^xj * (1-phi)^(1-xj)
        likelihood = np.prod(np.power(phi, x) * np.power(1 - phi, 1 - x))
        return likelihood

    def predict_proba(self, X):
        """
        Compute posterior probabilities P(y|x).
        :param X: Feature matrix
        :return: Array of shape (n_samples, 2)
        """
        probs = []
        for x in X:
            # Likelihoods
            p_x_y1 = self._likelihood(x, 1)
            p_x_y0 = self._likelihood(x, 0)

            # Priors
            p_y1 = self.phi_y
            p_y0 = 1 - self.phi_y

            # Posterior (normalized)
            numerator1 = p_x_y1 * p_y1
            numerator0 = p_x_y0 * p_y0
            denom = numerator1 + numerator0

            probs.append([numerator0 / denom, numerator1 / denom])

        return np.array(probs)

    def predict(self, X):
        """
        Predict class labels for input samples.
        :param X: Feature matrix
        :return: Predicted labels
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Toy dataset (spam = 1, not spam = 0)
    # Features: [contains_word1, contains_word2, contains_word3]
    X = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])
    y = np.array([1, 1, 0, 0, 1])  # Labels

    # Train model
    nb = BernoulliNaiveBayes(alpha=1.0)
    nb.fit(X, y)

    # Predict on new sample
    X_test = np.array([[1, 0, 0], [0, 1, 1]])
    print("Predicted probabilities:\n", nb.predict_proba(X_test))
    print("Predicted labels:", nb.predict(X_test))
