# Logistic Regression: A Probabilistic Perspective

## 1. Why Logistic Regression?

- Logistic regression is used for **binary classification**: predicting whether an outcome is **0** or **1**.
- Unlike linear regression, which outputs any real number, logistic regression models **probabilities** in the range $[0, 1]$.

---

## 2. Hypothesis Function

We cannot use a linear function directly (since probabilities must be bounded).  
So, we pass the linear combination through a **sigmoid function**:

$$
h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

- $h_\theta(x)$ = predicted probability that $y = 1$ given $x$.
- $\theta^T x$ = linear score (weights + bias).

---

## 3. Probabilistic Model

We model the probability of each class as:

$$
P(y = 1 \mid x; \theta) = h_\theta(x)
$$

$$
P(y = 0 \mid x; \theta) = 1 - h_\theta(x)
$$

So the combined expression is:

$$
P(y \mid x; \theta) = \big(h_\theta(x)\big)^y \cdot \big(1 - h_\theta(x)\big)^{(1-y)}
$$

---

## 4. Likelihood Function

Given $m$ independent training examples $(x^{(i)}, y^{(i)})$, the likelihood is:

$$
L(\theta) = \prod_{i=1}^{m} P\big(y^{(i)} \mid x^{(i)}; \theta\big)
$$

Expanding:

$$
L(\theta) = \prod_{i=1}^{m} \Big(h_\theta(x^{(i)})\Big)^{y^{(i)}} \cdot \Big(1 - h_\theta(x^{(i)})\Big)^{1-y^{(i)}}
$$

---

## 5. Log-Likelihood

We take the **logarithm** of the likelihood (for numerical stability and easier differentiation):

$$
\ell(\theta) = \log L(\theta)
$$

$$
\ell(\theta) = \sum_{i=1}^{m} \Big[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log \big(1 - h_\theta(x^{(i)})\big) \Big]
$$

---

## 6. Why Log Instead of Raw Probability?

1. **Numerical stability**: Products of probabilities become very small $\to$ logs prevent underflow.
2. **Simplifies derivatives**: Log converts product $\to$ sum, making optimization tractable.
3. **Interpretability**: Log-likelihood directly relates to information theory (cross-entropy).

---

## 7. Cost Function (Negative Log-Likelihood)

In optimization, we usually **minimize** a cost function.  
So we take the **negative log-likelihood (NLL)**:

$$
J(\theta) = -\ell(\theta)
$$

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log \big(1 - h_\theta(x^{(i)})\big) \Big]
$$

This is also known as the **Binary Cross-Entropy Loss**.

---

## 8. Gradient Descent Updates

To minimize $J(\theta)$, we use gradient descent:

For each weight $w_j$:

$$
w_j := w_j - \alpha \frac{\partial J(\theta)}{\partial w_j}
$$

Bias term $b$:

$$
b := b - \alpha \frac{\partial J(\theta)}{\partial b}
$$

Where $\alpha$ = learning rate.

---

## 9. Why Sigmoid?

- Ensures output $\in [0,1]$ (valid probability).
- Smooth and differentiable.
- Natural probabilistic interpretation.
- Derived from the odds ratio:

$$
\text{Odds} = \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = e^{\theta^T x}
$$

Taking log:

$$
\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = \theta^T x
$$

This shows that logistic regression is a **linear model for the log-odds**.

---

## 10. Key Takeaways

- Logistic regression is a **probabilistic linear classifier**.
- Uses **sigmoid** to map scores to probabilities.
- Optimized via **maximum likelihood estimation (MLE)**.
- Loss function = **binary cross-entropy** (negative log-likelihood).
- Log is used for **stability, simplicity, and interpretability**.
