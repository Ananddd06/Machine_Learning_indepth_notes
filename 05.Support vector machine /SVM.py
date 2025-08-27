import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the SVM parameters
        - learning_rate: step size for gradient descent
        - lambda_param: regularization strength
        - n_iters: number of iterations for training
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVM using gradient descent
        - X: input features (num_samples x num_features)
        - y: labels (-1 or 1)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # weight vector
        self.b = 0                     # bias term

        # Gradient Descent loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # If correctly classified and outside margin
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    # If inside margin or misclassified
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y[idx])
                    db = -y[idx]

                # Update rule
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        """
        Predict class labels for input samples
        - X: input features
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Toy dataset (linearly separable)
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [3, 2]
    ])
    y = np.array([-1, -1, -1, 1, 1])  # labels must be -1 or +1

    svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)
    predictions = svm.predict(X)

    print("Weights:", svm.w)
    print("Bias:", svm.b)
    print("Predictions:", predictions)
