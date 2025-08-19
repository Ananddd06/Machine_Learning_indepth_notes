# 📌 Difference Between GDA and Logistic Regression

---

## 🔹 1. Type of Model

- **Logistic Regression**

  - **Discriminative model** → directly models the conditional probability:
    $$
    P(y|x;\theta)
    $$
  - Learns decision boundary without modeling the data distribution.

- **Gaussian Discriminant Analysis (GDA)**
  - **Generative model** → models the joint distribution:
    $$
    P(x, y) = P(x|y) \, P(y)
    $$
  - Learns class-conditional densities and priors, then uses Bayes’ rule.

---

## 🔹 2. Distributional Assumptions

- **Logistic Regression**

  - Makes **no assumption** about the distribution of features \(x\).
  - Only assumes the log-odds are linear in \(x\):
    $$
    \log \frac{P(y=1|x)}{P(y=0|x)} = \theta^T x
    $$

- **GDA**
  - Assumes features follow a **Gaussian distribution** conditioned on class:
    $$
    x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
    $$
    $$
    x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
    $$

---

## 🔹 3. Decision Boundary

- **Logistic Regression**

  - Always produces a **linear boundary**:
    $$
    P(y=1|x) = \frac{1}{1 + e^{-\theta^T x}}
    $$

- **GDA**
  - If \(\Sigma_0 = \Sigma_1\) → **linear boundary**.
  - If \(\Sigma_0 \neq \Sigma_1\) → **quadratic boundary**.

---

## 🔹 4. Parameter Estimation

- **Logistic Regression**

  - Parameters estimated using **Maximum Likelihood Estimation (MLE)**.
  - Requires **iterative optimization** (Gradient Descent, Newton’s Method).

- **GDA**
  - Parameters estimated by **closed-form MLE**:
    - Class priors:
      $$
      \phi = \frac{1}{m} \sum_{i=1}^m 1\{y^{(i)}=1\}
      $$
    - Class means:
      $$
      \mu_0 = \frac{\sum_{i: y^{(i)}=0} x^{(i)}}{\sum_{i: y^{(i)}=0} 1},
      \quad
      \mu_1 = \frac{\sum_{i: y^{(i)}=1} x^{(i)}}{\sum_{i: y^{(i)}=1} 1}
      $$
    - Shared covariance:
      $$
      \Sigma = \frac{1}{m} \sum_{i=1}^m \left(x^{(i)} - \mu_{y^{(i)}}\right) \left(x^{(i)} - \mu_{y^{(i)}}\right)^T
      $$

---

## 🔹 5. Data Efficiency & Performance

- **Logistic Regression**

  - Works better when data distribution **does not follow Gaussian assumptions**.
  - More **robust** to model mis-specification.

- **GDA**
  - More **data efficient** if Gaussian assumption is approximately correct.
  - Can perform better than logistic regression with small datasets.

---

## 🔹 6. Summary Table

| Aspect               | Logistic Regression          | GDA                                |
| -------------------- | ---------------------------- | ---------------------------------- | ----- | --------- |
| Model type           | Discriminative               | Generative                         |
| Probability modeled  | \(P(y                        | x)\)                               | \(P(x | y) P(y)\) |
| Assumption           | Linear log-odds              | Gaussian features                  |
| Decision boundary    | Linear                       | Linear or Quadratic                |
| Parameter estimation | Iterative (MLE)              | Closed-form (MLE)                  |
| Works best when      | No clear Gaussian assumption | Data follows Gaussian distribution |

---
