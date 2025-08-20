# Logistic Regression — From Scratch (Senior Applied Scientist Perspective)

---

## 1. Why Not Linear Regression for Classification?

- Linear regression outputs predictions in the range $(-\infty, +\infty)$.
- For binary classification, we need probabilities in $[0,1]$.
- If we threshold linear regression output at 0.5, predictions may work, but probabilities are **not bounded** and lack probabilistic interpretation.

**Example:**

- Predicting whether a customer clicks on an ad (1) or not (0).
- A probability > 1 or < 0 makes no sense.

Thus, we need a function that maps real values to $[0,1]$: the **sigmoid function**.

---

## 2. The Sigmoid Function

The **sigmoid (logistic) function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} , \quad z = \theta^T x
$$

### Properties:

- Range: (0,1) → interpretable as a probability.
- Differentiable → good for optimization.
- Symmetry → $\sigma(-z) = 1 - \sigma(z)$.

---

## 3. Probabilistic Model

We model the probability of class label $y \in \{0,1\}$ given input $x$:

$$
P(y=1|x;\theta) = \sigma(\theta^T x)
$$

$$
P(y=0|x;\theta) = 1 - \sigma(\theta^T x)
$$

Thus, the conditional probability:

$$
P(y|x;\theta) = (\sigma(\theta^T x))^y (1 - \sigma(\theta^T x))^{1-y}
$$

This single formula handles both cases (y=0 or y=1).

---

## 4. Log-Likelihood Function

Given dataset $D = \{(x^{(i)}, y^{(i)})\}_{i=1}^m$:

The **likelihood**:

$$
L(\theta) = \prod_{i=1}^m P(y^{(i)}|x^{(i)};\theta)
$$

Since products can underflow, we take log:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^m \Big[ y^{(i)} \log(\sigma(\theta^T x^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(\theta^T x^{(i)})) \Big]
$$

This is the **log-likelihood function**.

---

## 5. Maximum Likelihood Estimation (MLE)

We want to find parameters $\theta$ that maximize log-likelihood:

$$
\hat{\theta} = \arg\max_{\theta} \; \ell(\theta)
$$

Equivalently, minimize the **negative log-likelihood (NLL)**, which is the **Binary Cross-Entropy Loss**:

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]
$$

where $\hat{y}^{(i)} = \sigma(\theta^T x^{(i)})$.

---

## 6. Gradient Descent Update

To minimize cost, we use Gradient Descent.

The gradient of cost wrt weights:

$$
\nabla_\theta J(\theta) = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x^{(i)}
$$

Update rule:

$$
\theta := \theta - \eta \cdot \nabla_\theta J(\theta)
$$

where $\eta$ is learning rate.

---

## 7. Why Sigmoid Function?

**Q1: Why not linear function?**

- Because probabilities must be bounded in $[0,1]$.

**Q2: Why not step function?**

- Step function is non-differentiable → can’t optimize with gradient descent.

**Q3: Why sigmoid specifically?**

- Smooth, differentiable.
- Monotonic.
- Maps real line to (0,1).
- Historical roots in statistics and biology.

**Q4: Why not softmax?**

- Softmax is used for multi-class (K > 2) classification.
- Sigmoid is enough for binary classification.

---

## 8. Key Takeaways

- Logistic Regression is a **probabilistic linear classifier**.
- Uses **sigmoid** to map scores into probabilities.
- Parameters estimated using **Maximum Likelihood Estimation**.
- Loss function = **Binary Cross-Entropy**.
- Optimization via **Gradient Descent** (or variants like SGD, Adam).

---

## 9. Interview-Style Questions

1. **Why logistic regression instead of linear regression for classification?**

   - Linear regression does not produce valid probabilities.

2. **What is the cost function in logistic regression?**

   - Binary Cross-Entropy (Negative Log-Likelihood).

3. **Why use log-likelihood instead of likelihood directly?**

   - Numerical stability and converts product → sum.

4. **Why is sigmoid function used?**

   - Maps to (0,1), differentiable, probabilistic interpretation.

5. **Is logistic regression a linear or nonlinear classifier?**

   - Linear in parameters (decision boundary is linear).

---
