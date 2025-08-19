# Gaussian Discriminant Analysis (GDA)

> **Goal:** Build a principled classifier for continuous features by modeling the **joint** distribution \( p(x, y) \) with Gaussians and using Bayes’ rule to obtain \( p(y \mid x) \).  
> **Setting:** Binary classification \( y \in \{0,1\} \), features \( x \in \mathbb{R}^d \).

---

## 1) Generative vs. Discriminative: Why GDA?

- **Discriminative** models (e.g., Logistic Regression) model \( p(y \mid x) \) **directly**.
- **Generative** models (e.g., GDA) model the **joint** \( p(x, y) = p(y)\,p(x \mid y) \), then use Bayes’ rule to get \( p(y \mid x) \).

**Pros of GDA:**

- Closed-form **maximum likelihood** estimates (MLE) for parameters.
- Naturally handles **class priors** and yields calibrated **posteriors** (under correct model).
- When class-conditional densities are near-Gaussian, GDA can be very **data-efficient**.

**Cons of GDA:**

- Sensitive to **distributional mis-specification** (non-Gaussian tails, multimodality).
- Requires estimating a **covariance matrix** (can be ill-conditioned in high-d).

---

## 2) Model Specification

We assume:

- **Prior**

  $$
  y \sim \mathrm{Bernoulli}(\phi), \quad \Rightarrow \quad p(y) = \phi^y (1-\phi)^{1-y}, \quad \phi = p(y=1).
  $$

- **Class-conditional densities (shared covariance \(\Sigma\))**

  $$
  x \mid y=0 \sim \mathcal{N}(\mu_0, \Sigma), \qquad
  x \mid y=1 \sim \mathcal{N}(\mu_1, \Sigma).
  $$

- **Multivariate Gaussian pdf**
  $$
  p(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}
  \exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right).
  $$

Thus,

$$
p(x \mid y=i) = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}
\exp\!\left(-\tfrac{1}{2}(x-\mu_i)^\top \Sigma^{-1}(x-\mu_i)\right),\quad i\in\{0,1\}.
$$

---

## 3) Bayes’ Rule and the Posterior \( p(y=1\mid x) \)

Bayes’ rule:

$$
p(y=1\mid x) = \frac{p(x\mid y=1)\,p(y=1)}{p(x\mid y=0)\,p(y=0) + p(x\mid y=1)\,p(y=1)}.
$$

Equivalently, the **log-posterior-odds**:

$$
\log\frac{p(y=1\mid x)}{p(y=0\mid x)}
= \log\frac{p(x\mid y=1)\,p(y=1)}{p(x\mid y=0)\,p(y=0)}.
$$

Substitute the Gaussians and expand the quadratic forms:

$$
\begin{aligned}
\log\frac{p(y=1\mid x)}{p(y=0\mid x)}
&= \log\frac{\phi}{1-\phi}
-\tfrac{1}{2}\big[(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1) - (x-\mu_0)^\top\Sigma^{-1}(x-\mu_0)\big] \\
&= \underbrace{\big(\Sigma^{-1}(\mu_1-\mu_0)\big)^\top}_{\theta^\top} x \;+\;
\underbrace{\left(
-\tfrac{1}{2}\mu_1^\top\Sigma^{-1}\mu_1
+\tfrac{1}{2}\mu_0^\top\Sigma^{-1}\mu_0
+\log\frac{\phi}{1-\phi}
\right)}_{\theta_0}.
\end{aligned}
$$

Therefore,

$$
p(y=1\mid x) = \sigma\!\left(\theta^\top x + \theta_0\right), \quad
\theta = \Sigma^{-1}(\mu_1 - \mu_0),
\quad
\theta_0 = -\tfrac{1}{2}\mu_1^\top\Sigma^{-1}\mu_1
+\tfrac{1}{2}\mu_0^\top\Sigma^{-1}\mu_0
+\log\frac{\phi}{1-\phi},
$$

with \( \sigma(t) = \frac{1}{1+e^{-t}} \).  
**Key:** Under shared \(\Sigma\), the decision boundary is **linear** in \(x\).

---

## 4) Likelihood and Joint Log-Likelihood

Given i.i.d. data \( \mathcal{D}=\{(x^{(i)}, y^{(i)})\}\_{i=1}^m \), the **likelihood** is

$$
L(\phi,\mu_0,\mu_1,\Sigma)
= \prod_{i=1}^m p\big(y^{(i)}\big)\; p\big(x^{(i)} \mid y^{(i)}\big).
$$

The **joint log-likelihood**:

$$
\begin{aligned}
\ell(\phi,\mu_0,\mu_1,\Sigma)
&= \sum_{i=1}^m \log p\big(y^{(i)}\big)
+ \sum_{i=1}^m \log p\big(x^{(i)} \mid y^{(i)}\big) \\[4pt]
&= \sum_{i=1}^m \Big[ y^{(i)} \log\phi + (1-y^{(i)})\log(1-\phi) \Big] \\
&\quad - \frac{m d}{2}\log(2\pi) - \frac{m}{2}\log|\Sigma|
- \frac{1}{2}\sum_{i=1}^m \big(x^{(i)}-\mu_{y^{(i)}}\big)^\top \Sigma^{-1} \big(x^{(i)}-\mu_{y^{(i)}}\big).
\end{aligned}
$$

Up to constants, the objective to **maximize** is

$$
\ell = \sum_{i=1}^m \Big[ y^{(i)} \log\phi + (1-y^{(i)})\log(1-\phi) \Big]
- \frac{m}{2}\log|\Sigma|
- \frac{1}{2}\sum_{i=1}^m \big(x^{(i)}-\mu_{y^{(i)}}\big)^\top \Sigma^{-1} \big(x^{(i)}-\mu_{y^{(i)}}\big).
$$

---

## 5) Maximum Likelihood Estimation (Closed Forms)

Let \( m_1=\sum_i \mathbf{1}\{y^{(i)}=1\} \), \( m_0=m-m_1 \).

### (a) Prior

$$
\frac{\partial \ell}{\partial \phi}
= \frac{m_1}{\phi} - \frac{m_0}{1-\phi} = 0
\quad \Rightarrow \quad
\boxed{\;\phi^\star = \frac{m_1}{m}\;}
$$

### (b) Class Means

Take derivative w.r.t. \( \mu_1 \) (and analogously \( \mu_0 \)):

$$
\frac{\partial}{\partial \mu_1}
\left[-\frac{1}{2}\sum_{i: y^{(i)}=1}
(x^{(i)}-\mu_1)^\top \Sigma^{-1} (x^{(i)}-\mu_1)\right]
= \sum_{i: y^{(i)}=1} \Sigma^{-1}\big(x^{(i)}-\mu_1\big) = 0.
$$

Multiply by \( \Sigma \) and rearrange:

$$
\boxed{\;\mu_1^\star = \frac{1}{m_1}\sum_{i: y^{(i)}=1} x^{(i)}\;}, \qquad
\boxed{\;\mu_0^\star = \frac{1}{m_0}\sum_{i: y^{(i)}=0} x^{(i)}\;}
$$

### (c) Shared Covariance

Use matrix calculus identities \( \partial \log|\Sigma|/\partial \Sigma = \Sigma^{-1} \) and
\( \partial \,\mathrm{tr}(\Sigma^{-1}S)/\partial \Sigma = -\Sigma^{-1} S \Sigma^{-1} \):

$$
\frac{\partial \ell}{\partial \Sigma}
= -\frac{m}{2}\Sigma^{-1}
+\frac{1}{2}\Sigma^{-1}\!\left[\sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^\top\right]\!\Sigma^{-1} = 0.
$$

Multiply by \( \Sigma \) on both sides and rearrange:

$$
\boxed{\;\Sigma^\star = \frac{1}{m}\sum_{i=1}^m
\big(x^{(i)}-\mu_{y^{(i)}}^\star\big)\big(x^{(i)}-\mu_{y^{(i)}}^\star\big)^\top\;}
$$

**Summary of MLEs:**

$$
\phi^\star=\frac{m_1}{m},\quad
\mu_0^\star=\frac{1}{m_0}\sum_{i: y^{(i)}=0} x^{(i)},\quad
\mu_1^\star=\frac{1}{m_1}\sum_{i: y^{(i)}=1} x^{(i)},\quad
\Sigma^\star=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}}^\star)(x^{(i)}-\mu_{y^{(i)}}^\star)^\top.
$$

---

## 6) Decision Rule and Boundary

Classify to the class with larger posterior:

$$
\hat{y}(x) = \arg\max_{y\in\{0,1\}} \; p(y\mid x).
$$

Under shared \(\Sigma\), the **log-odds** is **linear** in \(x\):

$$
\log\frac{p(y=1\mid x)}{p(y=0\mid x)} = \theta^\top x + \theta_0,
$$

so the decision boundary is the hyperplane \( \{x : \theta^\top x + \theta_0 = 0\} \).

If each class has its own covariance \( \Sigma_0, \Sigma_1 \) (QDA), the boundary becomes **quadratic** in \(x\).

---

## 7) Multiclass Extension (LDA)

For \( y\in\{1,\dots,K\} \) with shared \(\Sigma\):

$$
x \mid y=k \sim \mathcal{N}(\mu_k, \Sigma),\quad p(y=k)=\pi_k.
$$

MLE:

$$
\pi_k=\frac{m_k}{m},\quad
\mu_k=\frac{1}{m_k}\sum_{i: y^{(i)}=k} x^{(i)},\quad
\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^\top.
$$

Posterior is **softmax** over linear scores:

$$
p(y=k\mid x)=\frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)},\quad
w_k=\Sigma^{-1}\mu_k,\;\; b_k=-\tfrac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k.
$$

---

## 8) Practical Considerations

- **Standardization**: Center/scale features to stabilize \(\Sigma\) estimation.
- **Ill-conditioning**: If \( d \) is large or \( m \) is small, \(\Sigma\) may be singular. Use **regularization**:
  $$
  \Sigma_\lambda = \Sigma + \lambda I \quad \text{(ridge/Tikhonov)},
  $$
  or **shrinkage** toward diagonal.
- **Class imbalance**: The prior \( \phi \) carries class frequency; you can override it with domain priors.
- **Outliers / heavy tails**: Gaussians can be brittle; consider robust covariance or mixture models.
- **Naive Bayes connection**: Diagonal \(\Sigma\) \(\Rightarrow\) conditional independence of features given \(y\).

---

## 9) From-Scratch NumPy Implementation (Binary GDA)

```python
import numpy as np

class GaussianDiscriminantAnalysis:
    def __init__(self, reg=0.0):
        self.reg = reg  # ridge regularization added to Sigma
        self.phi = None
        self.mu0 = None
        self.mu1 = None
        self.Sigma = None
        self.Sigma_inv = None

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y).astype(int)
        m, d = X.shape

        # Priors and means
        self.phi = y.mean()
        self.mu0 = X[y == 0].mean(axis=0)
        self.mu1 = X[y == 1].mean(axis=0)

        # Shared covariance (pooled)
        Sigma = np.zeros((d, d))
        for i in range(m):
            mui = self.mu1 if y[i] == 1 else self.mu0
            diff = (X[i] - mui).reshape(-1, 1)
            Sigma += diff @ diff.T
        Sigma /= m

        # Regularize for stability
        if self.reg > 0:
            Sigma = Sigma + self.reg * np.eye(d)

        self.Sigma = Sigma
        self.Sigma_inv = np.linalg.pinv(Sigma)  # robust inverse

    def _logit_params(self):
        # theta, theta0 for the logistic form
        theta = self.Sigma_inv @ (self.mu1 - self.mu0)
        theta0 = (-0.5 * self.mu1 @ self.Sigma_inv @ self.mu1
                  + 0.5 * self.mu0 @ self.Sigma_inv @ self.mu0
                  + np.log(self.phi / (1 - self.phi)))
        return theta, theta0

    def predict_proba(self, X):
        X = np.asarray(X)
        theta, theta0 = self._logit_params()
        logits = X @ theta + theta0
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 10) Complexity, Diagnostics, and Tips

- **Compute cost**: Estimating \(\Sigma\) is \( \mathcal{O}(md^2) \); inverting is \( \mathcal{O}(d^3) \).
- **Numerical stability**: Prefer `np.linalg.pinv` or Cholesky; add small **jitter** to the diagonal.
- **Model diagnostics**: Plot class-conditional **QQ-plots**; inspect **Mahalanobis distances**; check **residuals**.
- **When to use GDA**: Moderate \(d\), Gaussian-like classes, limited data (generative gains). Otherwise, logistic regression or regularized LDA can be safer.

---

## 11) Cheat Sheet (Binary, Shared \(\Sigma\))

- **MLEs**

  $$
  \phi=\frac{m_1}{m},\quad
  \mu_0=\frac{1}{m_0}\sum_{i:y^{(i)}=0}x^{(i)},\quad
  \mu_1=\frac{1}{m_1}\sum_{i:y^{(i)}=1}x^{(i)},\quad
  \Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^\top.
  $$

- **Posterior**

  $$
  p(y=1\mid x) = \sigma\!\left(\underbrace{\Sigma^{-1}(\mu_1-\mu_0)}_{\theta}\!^\top x
  + \underbrace{\Big(-\tfrac{1}{2}\mu_1^\top\Sigma^{-1}\mu_1 + \tfrac{1}{2}\mu_0^\top\Sigma^{-1}\mu_0
  + \log\frac{\phi}{1-\phi}\Big)}_{\theta_0}\right).
  $$

- **Decision boundary**: \( \theta^\top x + \theta_0 = 0 \) (linear).
- **QDA**: Allow \(\Sigma_0 \neq \Sigma_1\) \(\Rightarrow\) quadratic boundary.

---

## 12) FAQ

- **Why multivariate Gaussian instead of univariate?**  
  To capture **correlations** between features via \(\Sigma\). Univariate (or diagonal \(\Sigma\)) assumes conditional independence (Naive Bayes).

- **What if \(\Sigma\) is singular?**  
  Use **regularization** \( \Sigma + \lambda I \), reduce dimensionality (PCA), or use diagonal/shrinkage estimators.

- **How does this relate to Logistic Regression?**  
  With shared \(\Sigma\), the posterior is **logistic in \(x\)**. Logistic regression learns \(\theta\) **discriminatively**; GDA derives \(\theta\) from generative parameters.

---

_End of notes._
