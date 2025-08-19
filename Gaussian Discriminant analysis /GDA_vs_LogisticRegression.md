# Gaussian Discriminant Analysis (GDA) vs Logistic Regression

This document provides a detailed comparison of **GDA** and **Logistic Regression**, focusing on assumptions, decision boundaries, parameter estimation, and use cases.

---

## 1. Model Type

- **Logistic Regression (LR)**:  
  Discriminative model.  
  Directly models the conditional probability:

  $$
  P(y|x; \theta)
  $$

- **Gaussian Discriminant Analysis (GDA)**:  
  Generative model.  
  Models the joint distribution of $(x, y)$ and then applies Bayes' rule:

  $$
  P(y|x) = \frac{P(x|y) \, P(y)}{P(x)}
  $$

---

## 2. Assumptions

- **Logistic Regression**:  
  Makes **no assumption** about the distribution of $x$.

- **GDA**:  
  Assumes **Gaussian distribution** for features conditioned on class:

  $$
  x \mid y=0 \sim \mathcal{N}(\mu_0, \Sigma)
  $$

  $$
  x \mid y=1 \sim \mathcal{N}(\mu_1, \Sigma)
  $$

---

## 3. Decision Boundary

- **Logistic Regression**:  
  Always produces a **linear decision boundary**:

  $$
  P(y=1 \mid x) = \frac{1}{1 + e^{-\theta^T x}}
  $$

- **GDA**:
  - If covariance matrices are **equal** ($\Sigma_0 = \Sigma_1$): boundary is **linear**.
  - If covariance matrices are **different** ($\Sigma_0 \neq \Sigma_1$): boundary is **quadratic**.

---

## 4. Parameter Estimation

- **Logistic Regression**:  
  Parameters are estimated using **Maximum Likelihood Estimation (MLE)**.  
  Requires iterative optimization methods such as:

  - Gradient Descent
  - Newton’s Method

- **GDA**:  
  Parameters are estimated in **closed form**:

  - Prior (class probability):

    $$
    \phi = \frac{1}{m} \sum_{i=1}^m 1\{ y^{(i)} = 1 \}
    $$

  - Class means:

    $$
    \mu_k = \frac{\sum_{i=1}^m 1\{ y^{(i)} = k \} \, x^{(i)}}{\sum_{i=1}^m 1\{ y^{(i)} = k \}}
    $$

  - Shared covariance matrix:

    $$
    \Sigma = \frac{1}{m} \sum_{i=1}^m \big( x^{(i)} - \mu_{y^{(i)}} \big) \big( x^{(i)} - \mu_{y^{(i)}} \big)^T
    $$

---

## 5. Advantages & Disadvantages

### Logistic Regression

✅ Robust to model misspecification (no strong distribution assumptions).  
✅ Always gives a linear boundary → efficient for high dimensions.  
❌ Requires iterative optimization (no closed-form solution).

### GDA

✅ If Gaussian assumption holds → more **statistically efficient** (needs less data).  
✅ Closed-form parameter estimation.  
✅ Can capture quadratic boundaries (if $\Sigma_0 \neq \Sigma_1$).  
❌ If Gaussian assumption is violated → performs poorly.

---

## 6. When to Use Which?

- **Use Logistic Regression if**:

  - You don’t want to assume a distribution for $x$.
  - Features are high-dimensional and sparse (e.g., text classification).
  - You prefer robustness over data efficiency.

- **Use GDA if**:
  - Data is well-modeled by Gaussian distributions per class.
  - You have limited training data (closed-form estimates are efficient).
  - You need quadratic decision boundaries.

---

## 7. Summary Table

| Aspect               | Logistic Regression               | GDA                                                                      |
| -------------------- | --------------------------------- | ------------------------------------------------------------------------ |
| Type                 | Discriminative                    | Generative                                                               |
| Models               | $P(y \mid x)$                     | $P(x \mid y)$ and $P(y)$                                                 |
| Distribution Assum.  | None                              | Gaussian                                                                 |
| Decision Boundary    | Linear                            | Linear (if $\Sigma_0=\Sigma_1$), Quadratic (if $\Sigma_0 \neq \Sigma_1$) |
| Parameter Estimation | Iterative (MLE + GD/Newton)       | Closed form                                                              |
| Data Efficiency      | Less efficient if data is limited | More efficient if Gaussian assumption holds                              |
| Robustness           | High                              | Low (sensitive to wrong distributional assumptions)                      |

---
