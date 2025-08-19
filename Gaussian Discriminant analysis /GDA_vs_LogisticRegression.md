# Logistic Regression vs Gaussian Discriminant Analysis (GDA)

This document provides a structured comparison of **Logistic Regression** and **Gaussian Discriminant Analysis (GDA)** for interview preparation.

---

## 1. Model Type

- **Logistic Regression (Discriminative):**  
  Directly models

  $$
  P(y|x; \theta)
  $$

  without assuming a distribution for $x$.

- **GDA (Generative):**  
  Models the **joint distribution**:
  $$
  P(x, y) = P(x|y)P(y)
  $$

---

## 2. Core Idea

- **Logistic Regression:** Learn the **decision boundary** directly.
- **GDA:** Model the data generation process and use Bayes’ rule:
  $$
  P(y|x) = \frac{P(x|y)P(y)}{\sum_{y'} P(x|y')P(y')}
  $$

---

## 3. Assumptions

- **Logistic Regression:**  
  No assumption on the distribution of $x$.

- **GDA:**  
  Assumes Gaussian distribution for features:
  $$
  x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
  $$
  $$
  x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
  $$

---

## 4. Decision Boundary

- **Logistic Regression:** Always linear:

  $$
  P(y=1|x) = \frac{1}{1 + e^{-\theta^T x}}
  $$

- **GDA:**
  - If covariance matrices are equal ($\Sigma_0 = \Sigma_1$): **Linear boundary**
  - If covariance matrices differ ($\Sigma_0 \neq \Sigma_1$): **Quadratic boundary**

---

## 5. Parameter Estimation

- **Logistic Regression:**

  - Parameters estimated using **Maximum Likelihood Estimation (MLE)**.
  - Requires iterative optimization (e.g., **Gradient Descent**, **Newton’s Method**).

- **GDA:**  
  Parameters estimated in **closed form**:

  Prior:

  $$
  \phi = \frac{1}{m} \sum_{i=1}^m 1\{ y^{(i)}=1 \}
  $$

  Class means:

  $$
  \mu_k = \frac{\sum_{i=1}^m 1\{ y^{(i)}=k \} x^{(i)}}{\sum_{i=1}^m 1\{ y^{(i)}=k \}}
  $$

  Shared covariance:

  $$
  \Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
  $$

---

## 6. When to Use

- **Logistic Regression:**

  - Works well with large datasets.
  - Robust when data is **not Gaussian**.
  - Preferred when decision boundary is **linear**.

- **GDA:**
  - Useful when features are approximately Gaussian.
  - Performs well with **smaller datasets** (fewer parameters to estimate).
  - Provides **quadratic boundaries** when covariance differs.

---

## 7. Summary Table

| Aspect                  | Logistic Regression (Discriminative)  | GDA (Generative)                 |
| ----------------------- | ------------------------------------- | -------------------------------- | ---- | ------- |
| Models                  | $P(y                                  | x)$                              | $P(x | y)P(y)$ |
| Distribution assumption | None                                  | Gaussian                         |
| Boundary                | Linear                                | Linear or Quadratic              |
| Estimation              | Iterative (MLE + GD/Newton)           | Closed form (MLE)                |
| Best when               | Large dataset, arbitrary distribution | Small dataset, Gaussian features |

---
