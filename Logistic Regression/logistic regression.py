import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    # Hypothesis function (sigmoid)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Cost function (binary cross-entropy)
    def compute_cost(self, y, y_hat):
        m = len(y)
        cost = -(1/m) * np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))
        return cost
    
    # Training with Gradient Descent
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Hypothesis (sigmoid)
            y_hat = self.sigmoid(linear_model)

            # Gradients
            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum(y_hat - y)

            # Update rule (Gradient Descent → subtract)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self
    
    # Prediction probabilities
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    # Prediction classes
    def predict(self, X, threshold=0.5):
        y_prob = self.predict_proba(X)
        return np.where(y_prob >= threshold, 1, 0)


# Example usage
if __name__ == "__main__":
    # Dummy dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, epochs=1000)
    model.fit(X, y)

    preds = model.predict(X)
    print("Predictions:", preds)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
