import numpy as np

class LinearRegressionClosedForm:
    def __init__(self):
        self.theta = None  # [weights..., bias]

    def fit(self, X, y):
        """
        Fits the model using the Normal Equation (no gradient descent).
        """
        # Add a column of 1s to X for the intercept (bias)
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))  # shape: (n_samples, n_features + 1)

        # Closed-form solution (Normal Equation)
        XTX = X_b.T.dot(X_b)
        XTy = X_b.T.dot(y)
        self.theta = np.linalg.inv(XTX).dot(XTy)

    def predict(self, X):
        """
        Predicts output using learned parameters.
        """
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        return X_b.dot(self.theta)

    def score(self, X, y):
        """
        Computes the R² score.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

if __name__ == "__main__":
    # Data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])

    model = LinearRegressionClosedForm()
    model.fit(X, y)

    predictions = model.predict(X)

    print("Weights (slope):", model.theta[1:])
    print("Bias (intercept):", model.theta[0])
    print("R² score:", model.score(X, y))
