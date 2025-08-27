# SVM Implementation with Gradient Descent and Hinge Loss

Below is a Python implementation of a **Support Vector Machine (SVM)** trained using **gradient descent** and the **hinge loss function**.

We will go through each function in the class one by one, explain the code **immediately after the function**, and then move to the next function.

---

## Code Implementation with Explanations

### 1. `__init__`

```python
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
```

#### Explanation

- This is the **constructor function** of the class.
- It sets the hyperparameters and placeholders for weights (`w`) and bias (`b`).
- **learning_rate (η)**: Controls how fast/slow gradient descent updates happen.
- **lambda_param (λ)**: Regularization strength that prevents overfitting by penalizing large weights.
- **n_iters**: Number of training iterations (epochs).
- **w** and **b** are initialized as `None` here and will be set later in `fit`.

Mathematically, SVM minimizes the following cost function:
$J(w, b) = \frac{1}{2} ||w||^2 + \lambda \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b))$

---

### 2. `fit`

```python
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
```

#### Explanation

- **Purpose**: This function trains the SVM by updating weights and bias using **gradient descent**.
- First, weights are initialized as zeros, bias as 0.
- Loop runs for `n_iters` (epochs). Inside each epoch, we iterate through all training samples.

**Key logic:**

- Compute margin condition:
  $y_i (w^T x_i + b) \geq 1$

  - If this holds, the point is correctly classified and outside the margin.
  - If not, it’s either inside the margin or misclassified.

**Gradients:**

- If condition satisfied:

  - $dw = 2\lambda w$ (only regularization term contributes)
  - $db = 0$

- If condition violated:

  - $dw = 2\lambda w - y_i x_i$
  - $db = -y_i$

**Update rule:**

- $w \leftarrow w - \eta dw$
- $b \leftarrow b - \eta db$

This ensures that:

1. The margin is maximized.
2. Misclassified points get corrected by pulling hyperplane.

---

### 3. `predict`

```python
    def predict(self, X):
        """
        Predict class labels for input samples
        - X: input features
        """
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)
```

#### Explanation

- **Purpose**: To classify unseen data after training.
- Compute the linear decision function:
  $f(x) = w^T x + b$
- Apply the sign function:

  - If result > 0 → predict `+1`
  - If result < 0 → predict `-1`

This outputs whether a sample belongs to the positive or negative class.

---

## Summary of Workflow

1. `__init__` → Initializes hyperparameters and placeholders.
2. `fit` → Trains using gradient descent on hinge loss.
3. `predict` → Uses learned hyperplane to classify new samples.

This is a **linear SVM** built from scratch using **hinge loss** and **gradient descent**.
