# Gaussian Discriminant Analysis (GDA) vs Logistic Regression

## 1. Introduction
Both **Gaussian Discriminant Analysis (GDA)** and **Logistic Regression** are classification algorithms, but they differ in their **assumptions**, **mathematical formulation**, and **applicability**.

---

## 2. Core Idea

- **Logistic Regression (Discriminative Model):**
  Models the conditional probability directly:
  $$ P(y|x; \theta) $$

- **Gaussian Discriminant Analysis (Generative Model):**
  Models the joint probability of input and output:
  $$ P(x, y) = P(x|y; \theta) P(y) $$

  Then uses Bayes' rule:
  $$ P(y|x) = \frac{P(x|y) P(y)}{P(x)} $$

---

## 3. Assumptions

- **Logistic Regression:**
  - No distributional assumptions on $x$.
  - Only assumes the **log-odds** is linear in $x$:
    $$ \log \frac{P(y=1|x)}{P(y=0|x)} = \theta^T x $$

- **GDA:**
  - Assumes **Gaussian distribution** for $x|y$:
    $$ x|y=0 \sim \mathcal{N}(\mu_0, \Sigma) $$
    $$ x|y=1 \sim \mathcal{N}(\mu_1, \Sigma) $$

---

## 4. Decision Boundary

- **Logistic Regression:**
  Always linear:
  $$ P(y=1|x) = \frac{1}{1 + e^{-\theta^T x}} $$

- **GDA:**
  - If covariance matrices are equal ($\Sigma_0 = \Sigma_1$): Linear boundary.
  - If covariance matrices differ ($\Sigma_0 \neq \Sigma_1$): Quadratic boundary.

---

## 5. Parameter Estimation

- **Logistic Regression:**
  - Estimated using **Maximum Likelihood Estimation (MLE)**.
  - Requires iterative optimization (e.g., Gradient Descent, Newton’s Method).

- **GDA:**
  - Parameters estimated in **closed form**:
    - Prior: $$ \phi = \frac{1}{m} \sum_{i=1}^m 1\{ y^{(i)}=1 \} $$
    - Means: $$ \mu_k = \frac{\sum_{i=1}^m 1\{ y^{(i)}=k \} x^{(i)}}{\sum_{i=1}^m 1\{ y^{(i)}=k \}} $$
    - Covariance: $$ \Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T $$

---

## 6. When to Use

- **Use Logistic Regression when:**
  - You don’t want to assume Gaussian features.
  - Data distribution is unknown or not Gaussian.
  - You want robustness to model mis-specification.

- **Use GDA when:**
  - Data is approximately Gaussian.
  - Small dataset (GDA is more **data-efficient** because of distributional assumptions).
  - You want probabilistic generative modeling.

---

## 7. Advantages & Disadvantages

| Aspect | Logistic Regression | GDA |
|--------|----------------------|-----|
| **Model Type** | Discriminative | Generative |
| **Decision Boundary** | Linear | Linear / Quadratic |
| **Distribution Assumption** | None | Gaussian |
| **Estimation** | Iterative | Closed-form |
| **Small Dataset Performance** | Worse | Better (if assumptions hold) |
| **Large Dataset Performance** | Very good | Converges to Logistic Regression |

---

## 8. Summary

- Logistic Regression is **more robust** when Gaussian assumption does not hold.  
- GDA is **more efficient with small datasets** if Gaussian assumption is reasonable.  
- Both converge to similar performance with very large datasets.

---
