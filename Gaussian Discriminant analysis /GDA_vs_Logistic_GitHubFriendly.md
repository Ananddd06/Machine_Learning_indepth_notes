# Logistic Regression vs Gaussian Discriminant Analysis (GDA)

This document provides a structured, **GitHub-friendly** comparison of **Logistic Regression** and **Gaussian Discriminant Analysis (GDA)** for interview preparation.  
All mathematical formulas are embedded as images (hosted by CodeCogs) so they render across platforms (GitHub, plain Markdown viewers, etc.).

---

## 1. Model Type

- **Logistic Regression (Discriminative):**  
  Directly models

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20P%28y%7Cx%3B%20%5Ctheta%29)

  without assuming a distribution for $x$.

- **GDA (Generative):**  
  Models the **joint distribution**:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20P%28x%2Cy%29%3DP%28x%7Cy%29P%28y%29)

---

## 2. Core Idea

- **Logistic Regression:** Learn the **decision boundary** directly.
- **GDA:** Model the data generation process and use Bayes’ rule:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20P%28y%7Cx%29%3D%5Cfrac%7BP%28x%7Cy%29P%28y%29%7D%7B%5Csum_%7By%27%7DP%28x%7Cy%27%29P%28y%27%29%7D)

---

## 3. Assumptions

- **Logistic Regression:**  
  No assumption on the distribution of $x$.

- **GDA:**  
  Assumes Gaussian distribution for features:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20x%5Cmid%20y%3D0%5Csim%5Cmathcal%7BN%7D%28%5Cmu_0%2C%5CSigma%29)

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20x%5Cmid%20y%3D1%5Csim%5Cmathcal%7BN%7D%28%5Cmu_1%2C%5CSigma%29)

---

## 4. Decision Boundary

- **Logistic Regression:** Always linear:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20P%28y%3D1%5Cmid%20x%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5ET%20x%7D%7D)

- **GDA:**

  - If covariance matrices are equal ($\Sigma_0 = \Sigma_1$): **Linear boundary**
  - If covariance matrices differ ($\Sigma_0 \neq \Sigma_1$): **Quadratic boundary**

  The log-odds for GDA (shared covariance) is linear:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5Clog%5Cfrac%7BP%28y%3D1%5Cmid%20x%29%7D%7BP%28y%3D0%5Cmid%20x%29%7D%3D%5Ctheta%5ET%20x%20%2B%20%5Ctheta_0)

---

## 5. Parameter Estimation

- **Logistic Regression:**

  - Parameters estimated using **Maximum Likelihood Estimation (MLE)**.
  - Requires iterative optimization (e.g., **Gradient Descent**, **Newton’s Method**).

  Log-likelihood:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5Cell%28%5Ctheta%29%3D%5Csum_%7Bi%3D1%7D%5Em%5Cleft%5By%5E%7B%28i%29%7D%5Clog%20P%28y%5E%7B%28i%29%7D%5Cmid%20x%5E%7B%28i%29%7D%3B%5Ctheta%29%2B%281-y%5E%7B%28i%29%7D%29%5Clog%281-P%28y%5E%7B%28i%29%7D%5Cmid%20x%5E%7B%28i%29%7D%3B%5Ctheta%29%29%5Cright%5D)

- **GDA:**  
  Parameters estimated in **closed form**:

  Prior:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5Cphi%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%201%5C%7B%20y%5E%7B%28i%29%7D%3D1%20%5C%7D)

  Class means:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5Cmu_k%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5Em%201%5C%7B%20y%5E%7B%28i%29%7D%3Dk%20%5C%7D%20x%5E%7B%28i%29%7D%7D%7B%5Csum_%7Bi%3D1%7D%5Em%201%5C%7B%20y%5E%7B%28i%29%7D%3Dk%20%5C%7D%7D)

  Shared covariance:

  ![formula](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5CSigma%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%20%28x%5E%7B%28i%29%7D-%5Cmu_%7By%5E%7B%28i%29%7D%7D%29%28x%5E%7B%28i%29%7D-%5Cmu_%7By%5E%7B%28i%29%7D%7D%29%5ET)

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
| ----------------------- | ------------------------------------- | -------------------------------- |
| Models                  | $P(y \\mid x)$                        | $P(x \\mid y)P(y)$               |
| Distribution assumption | None                                  | Gaussian                         |
| Boundary                | Linear                                | Linear or Quadratic              |
| Estimation              | Iterative (MLE + GD/Newton)           | Closed form (MLE)                |
| Best when               | Large dataset, arbitrary distribution | Small dataset, Gaussian features |

---

## 8. Notes

- This Markdown uses external images (CodeCogs) to render LaTeX. If you want a fully offline version, I can:
  - Generate the SVG images and embed them in a ZIP with the `.md`, or
  - Produce a version with LaTeX blocks (`$$ ... $$`) for MathJax/KateX-enabled viewers (e.g., Jupyter, Obsidian).
