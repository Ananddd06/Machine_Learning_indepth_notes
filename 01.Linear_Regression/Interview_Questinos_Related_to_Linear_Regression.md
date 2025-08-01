# 📘 Linear Regression: Full Interview & Research-Oriented Question Bank (with Math Intuition)

This document contains every possible angle from which **Linear Regression** can be asked in **interviews (Amazon, DeepMind, Google)**, **research discussions**, and **GATE DA/AI exams**, including **math concepts**, **model behavior**, **metrics**, and **pitfalls**. Mastering this ensures complete readiness.

---

## Sources / Links for the inteview Quetsions :

[Linear Regression Interview Questions](https://devinterview.io/questions/machine-learning-and-data-science/logistic-regression-interview-questions/)

---

## 🔹 I. Core Math & Optimization Questions

1. **Define the hypothesis function.**
   $h_\theta(x) = \theta_0 + \theta_1 x$

2. **Derive the normal equation.**
   $\theta = (X^TX)^{-1}X^Ty$

3. **Why is the cost function convex?**
   Because it's quadratic in $\theta$; Hessian is positive semi-definite.

4. **Derive the gradient descent update rules.**
   $\theta_j := \theta_j - \alpha \cdot \partial J / \partial \theta_j$

5. **Difference between closed-form and gradient descent solutions.**

6. **What if $X^TX$ is not invertible?**
   Multicollinearity; use pseudo-inverse or Ridge regression.

7. **What is the geometric interpretation of regression?**
   Projection of $y$ onto the column space of $X$.

---

## 🔸 II. Statistical Assumptions & Interpretation

1. **List assumptions of linear regression:**

   - Linearity
   - Independence of errors
   - Homoscedasticity
   - Normality of residuals

2. **What is the Gauss-Markov theorem?**
   OLS is the BLUE (Best Linear Unbiased Estimator).

3. **Effect of violating assumptions:**

   - Heteroscedasticity → inefficient estimates
   - Multicollinearity → unstable coefficients

4. **OLS vs MLE:**

   - OLS minimizes squared error
   - MLE assumes Gaussian noise and maximizes likelihood

5. **What happens if residuals are heteroscedastic?**
6. **What if residuals are autocorrelated (esp. in time series)?**
7. **What happens if your error terms are not normally distributed?**
8. **Why can linear regression still work without Gaussian noise?**

---

## 🔍 III. Model Evaluation Metrics

| Metric                 | Formula                                | Use                  |     |                         |
| ---------------------- | -------------------------------------- | -------------------- | --- | ----------------------- |
| **MSE**                | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Training loss        |     |                         |
| **RMSE**               | $\sqrt{MSE}$                           | Same units as output |     |                         |
| **MAE**                | ( \frac{1}{n} \sum                     | y_i - \hat{y}\_i     | )   | More robust to outliers |
| **R-squared**          | $1 - \frac{SS_{res}}{SS_{tot}}$        | Goodness of fit      |     |                         |
| **Adjusted R-squared** | Penalizes extra features               | Feature evaluation   |     |                         |

---

## 🔧 IV. Problem Scenarios & Challenges

1. **Overfitting:** Too many features → high variance
2. **Underfitting:** Linear model can't capture non-linearity
3. **Outliers:** Heavily influence OLS estimates
4. **Multicollinearity:** $X^TX$ nearly singular → unstable $\theta$
5. **Feature Scaling:** Needed for gradient descent to converge quickly
6. **Large n, small d:** Easy to solve; small n, large d → overfitting risk

---

## 💬 V. Intuition-Based Questions

1. **Why is linear regression interpretable?**
2. **What does the slope $\theta_1$ mean?**
3. **Why can R^2 be negative?**
4. **Why does OLS force residuals to sum to 0?**
5. **Why do we use MSE instead of MAE in OLS?**

---

## 🎩 VI. Trick / Non-Obvious Questions

1. **What if the variance of residuals increases with x?** (heteroscedasticity)
2. **How do you debug a linear model with low R² but good test performance?**
3. **Can linear regression be used for classification?** (Why not?)
4. **Why do we one-hot encode categorical variables?**

---

## 🧠 VII. Extensions & Advanced Thinking

1. **Polynomial regression as a special case of linear regression**
2. **Linear regression with regularization (Ridge, Lasso)**
3. **Linear regression from a probabilistic standpoint (Bayesian Linear Regression)**
4. **Linear regression in high-dimensional spaces**

---

## 🧾 VIII. Linear Algebra Pre-requisites

| Concept               | Why It Matters                  |
| --------------------- | ------------------------------- |
| Matrix multiplication | $X^T X \theta = X^T y$ form     |
| Rank of matrix        | Whether $X^T X$ is invertible   |
| Eigenvalues / PSD     | Convexity of cost function      |
| Projection matrices   | Geometric interpretation of OLS |
| Vector norms          | Cost functions (L2, L1)         |
| Pseudo-inverse        | When $X^T X$ is not invertible  |

---

## ⚖️ IX. Advantages and Disadvantages

### ✅ Advantages

- Simple and interpretable
- Fast to train and predict
- Well-understood statistically
- Closed-form solution available

### ❌ Disadvantages

- Assumes linear relationship
- Sensitive to outliers
- Struggles with multicollinearity
- Not robust to non-linear data

---

## 🎯 X. Metrics to Evaluate Linear Model Accuracy

- **MSE / RMSE** → For loss/error
- **MAE** → Robust to outliers
- **R² / Adjusted R²** → Goodness of fit
- **AIC/BIC** → Model comparison (penalize complexity)

---

## 🔚 Summary:

Linear regression is simple yet powerful. Mastering it requires full knowledge of:

- 📐 The math (derivations, geometry)
- 🧮 The statistics (assumptions, inference)
- 🔧 The pitfalls (real-world limitations)
- 🔬 The extensions (regularization, Bayesian, GLMs)
- 📊 The metrics (R², RMSE, MAE)

Understand it deeply, and it becomes a launchpad for mastering all other ML models.

---
