import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionClosedForm:
    """
    Implements Linear Regression using the Normal Equation (closed-form solution).
    """
    def __init__(self):
        self.theta = None  # Model parameters (weights + bias)

    def fit(self, X, y):
        """
        Fits the linear model to the training data.

        Args:
            X (np.ndarray): Training data features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        # Add a bias term (intercept) to the feature matrix
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate theta using the Normal Equation: θ = (X^T * X)^(-1) * X^T * y
        try:
            XTX = X_b.T.dot(X_b)
            XTX_inv = np.linalg.inv(XTX)
            XTy = X_b.T.dot(y)
            self.theta = XTX_inv.dot(XTy)
        except np.linalg.LinAlgError:
            print("Error: Could not compute the inverse of X^T*X. The matrix might be singular.")
            self.theta = None

    def predict(self, X):
        """
        Makes predictions using the fitted linear model.

        Args:
            X (np.ndarray): Test data features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        if self.theta is None:
            raise RuntimeError("The model has not been fitted yet. Call fit() first.")
        
        # Add a bias term to the feature matrix
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predict: y_pred = X_b * θ
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
        if ss_total == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_total)

    def evaluate(self, X, y):
        """
        Evaluates the model on the given data using multiple metrics.

        Args:
            X (np.ndarray): Test data features.
            y (np.ndarray): True target values.

        Returns:
            dict: A dictionary containing MSE, RMSE, MAE, and R2 Score.
        """
        y_pred = self.predict(X)
        return {
            "MSE": self.mean_squared_error(y, y_pred),
            "RMSE": self.root_mean_squared_error(y, y_pred),
            "MAE": self.mean_absolute_error(y, y_pred),
            "R2 Score": self.r2_score(y, y_pred)
        }

# This block runs when the script is executed directly
if __name__ == "__main__":
    # 1. Create example data: y = 2x + 1
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # A perfect linear relationship

    # 2. Initialize and train the model
    model = LinearRegressionClosedForm()
    model.fit(X, y)

    # 3. Make predictions
    y_pred = model.predict(X)

    # 4. Print the learned parameters
    print("✨ Model Parameters ✨")
    # The first element of theta is the bias (intercept), the rest are the weights (slopes)
    print(f"Weights (slope): {model.theta[1:]}")
    print(f"Bias (intercept): {model.theta[0]:.4f}")

    # 5. Evaluate the model's performance
    metrics = model.evaluate(X, y)
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 6. Plot the results using Matplotlib
    print("\n📈 Generating plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', zorder=5)
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title('Linear Regression Fit')
    plt.xlabel('Feature (X)')
    plt.ylabel('Target (y)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("✨ Plot generated! ✨")