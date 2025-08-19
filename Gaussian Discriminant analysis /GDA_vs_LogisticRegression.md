# Gaussian Discriminant Analysis (GDA) vs Logistic Regression

## 1. Introduction

Both **Logistic Regression** and **Gaussian Discriminant Analysis (GDA)** are popular algorithms for **binary classification** (e.g., spam vs. ham).

- Logistic Regression is **discriminative** (models $P(y|x)$ directly).
- GDA is **generative** (models $P(x|y)$ and $P(y)$, then applies Bayes’ rule).

---

## 2. Logistic Regression

### Model Assumption

We assume the probability of class $y \in \{0,1\}$ given input $x$ follows a **sigmoid function**:

\[
P(y=1 \mid x; \theta) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
\]

\[
P(y=0 \mid x; \theta) = 1 - P(y=1 \mid x; \theta)
\]

---

### Decision Boundary

The decision rule is:

\[
\hat{y} =
\begin{cases}
1 & \text{if } \theta^T x \geq 0 \\
0 & \text{otherwise}
\end{cases}
\]

---

### Training (Maximum Likelihood Estimation)

We maximize the log-likelihood:

\[
\ell(\theta) = \sum\_{i=1}^{m} \Big[ y^{(i)} \log P(y^{(i)} \mid x^{(i)}; \theta) + (1-y^{(i)}) \log (1 - P(y^{(i)} \mid x^{(i)}; \theta)) \Big]
\]

Parameters $\theta$ are optimized via **gradient descent** (or variants like Adam, RMSProp, etc.).

---

## 3. Gaussian Discriminant Analysis (GDA)

### Model Assumptions

1. Prior:
   \[
   P(y=1) = \phi, \quad P(y=0) = 1 - \phi
   \]

2. Conditional distribution of features given label:
   \[
   x \mid y=0 \sim \mathcal{N}(\mu_0, \Sigma), \quad
   x \mid y=1 \sim \mathcal{N}(\mu_1, \Sigma)
   \]

   - $\mu_0, \mu_1$ are class means
   - $\Sigma$ is the shared covariance matrix

---

### Posterior via Bayes’ Rule

\[
P(y=1 \mid x) = \frac{P(x \mid y=1) P(y=1)}{P(x \mid y=0) P(y=0) + P(x \mid y=1) P(y=1)}
\]

---

### Decision Boundary

The log-odds is **linear** in $x$:

\[
\log \frac{P(y=1 \mid x)}{P(y=0 \mid x)} = \theta^T x + \theta_0
\]

Thus, like logistic regression, GDA also results in a **linear decision boundary** (when covariances are shared).

---

### Parameter Estimation (Closed-Form)

Parameters are estimated using **maximum likelihood** with closed-form solutions:

\[
\phi = \frac{1}{m} \sum\_{i=1}^m 1\{ y^{(i)} = 1 \}
\]

\[
\mu*0 = \frac{\sum*{i=1}^m 1\{ y^{(i)}=0 \} x^{(i)}}{\sum\_{i=1}^m 1\{ y^{(i)}=0 \}}
\]

\[
\mu*1 = \frac{\sum*{i=1}^m 1\{ y^{(i)}=1 \} x^{(i)}}{\sum\_{i=1}^m 1\{ y^{(i)}=1 \}}
\]

\[
\Sigma = \frac{1}{m} \sum*{i=1}^m \big( x^{(i)} - \mu*{y^{(i)}} \big) \big( x^{(i)} - \mu\_{y^{(i)}} \big)^T
\]

---

## 4. Comparison

| Aspect                   | Logistic Regression                        | Gaussian Discriminant Analysis                              |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| **Type**                 | Discriminative                             | Generative                                                  |
| **What it models**       | $P(y \mid x)$                              | $P(x \mid y)$ and $P(y)$                                    |
| **Assumptions**          | No distributional assumptions on $x$       | Features follow class-conditional Gaussian                  |
| **Decision Boundary**    | Linear in features                         | Linear if shared $\Sigma$, quadratic if different $\Sigma$  |
| **Parameter Estimation** | Gradient descent on log-likelihood         | Closed-form MLE (means, covariance, priors)                 |
| **Data Requirement**     | Works well with large data                 | Works well with small data (uses distributional assumption) |
| **Robustness**           | Robust to wrong distributional assumptions | Sensitive if Gaussian assumption is violated                |

---

## 5. Practical Guidelines

- Use **Logistic Regression** when:

  - You have **large labeled datasets**
  - You do **not** want to assume Gaussian feature distributions
  - Training speed & robustness are important

- Use **GDA** when:
  - You have **small datasets** (distributional assumption helps generalization)
  - The features are approximately **Gaussian-distributed** within each class
  - You want closed-form parameter estimates without iterative optimization

---

## 6. Example: Spam vs Ham Emails

- **Logistic Regression**: Directly learns decision boundary between spam and ham from labeled data. Works well with large datasets and diverse feature distributions (e.g., word embeddings, TF-IDF).

- **GDA**: Models distribution of words given spam vs ham separately (as Gaussians). Performs well if data really follows Gaussian-like clusters, but fails if distributions are highly skewed (e.g., sparse text features).

---

## 7. Key Takeaway

- **Logistic Regression**: safer, more general-purpose, robust in real-world tasks.
- **GDA**: powerful if Gaussian assumptions hold, especially with small data.
