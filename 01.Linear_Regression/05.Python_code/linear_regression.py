import numpy as np

class LinearRegressionClosedForm:
    def __init__(self):
        self.theta = None  # includes bias

    def fit(self, X, y):
        # Add intercept term
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        # Normal equation: theta = (X^T X)^-1 X^T y
        XTX = X_b.T.dot(X_b)
        XTy = X_b.T.dot(y)
        self.theta = np.linalg.inv(XTX).dot(XTy)

    def predict(self, X):
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        return X_b.dot(self.theta)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(self.mean_squared_error(y_true, y_pred))

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "MSE": self.mean_squared_error(y, y_pred),
            "RMSE": self.root_mean_squared_error(y, y_pred),
            "MAE": self.mean_absolute_error(y, y_pred),
            "R2 Score": self.r2_score(y, y_pred)
        }

if __name__ == "__main__":
    # Example data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # perfect fit

    model = LinearRegressionClosedForm()
    model.fit(X, y)

    y_pred = model.predict(X)

    print("Weights (slope):", model.theta[1:])
    print("Bias (intercept):", model.theta[0])

    metrics = model.evaluate(X, y)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
