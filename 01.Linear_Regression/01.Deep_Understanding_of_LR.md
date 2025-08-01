# 📘 Linear Regression: A Deep Dive

Linear regression is a foundational **statistical and machine learning technique** that models the relationship between a **dependent variable** (the outcome you want to predict) and one or more **independent variables** (factors you use to make the prediction) by fitting a **linear equation** to the observed data.

---

## 🎯 Core Idea: Finding the “Best Fit” Line (or Hyperplane)

At its heart, linear regression seeks to find a:

- 📈 **Straight line** in simple linear regression (1 input)
- 🧮 **Hyperplane** in multiple linear regression (multi-input)

This "best fit" is determined by **minimizing the discrepancies** between the actual observed values and the predicted values.

---

## 🧠 Mathematical Formulation

The linear regression hypothesis is:

$$
h\_\theta(x) = \theta_0 + \theta_1 x
$$

Where:

- \( h\_\theta(x) \): Predicted output
- \( \theta_0 \): Intercept (bias)
- \( \theta_1 \): Slope (weight)

In the multivariate case:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_n X_n + \epsilon
$$

- **Y**: Dependent variable
- **\( X_1, X_2, \dots, X_n \)**: Independent variables (features)
- **\( \beta_0 \)**: Intercept (when all X's are 0)
- **\( \beta_i \)**: Coefficient for each \( X_i \)
- **\( \epsilon \)**: Error term (residual)

---

## 🔍 How the "Best Fit" is Determined: Least Squares

The most common way to estimate the coefficients \( \beta \) is:

### ✅ Ordinary Least Squares (OLS)

1. 🔹 **Residual** for a point:

   $$
   \text{e_i} = y_i - \hat{y}\_i
   $$

2. 🔹 **Sum of Squared Residuals (SSR)**:
   $$
   \text{SSR} = \sum\_{i=1}^{m} (y_i - \hat{y}\_i)^2
   $$

OLS finds values of \( \beta_0, \beta_1, ..., \beta_n \) that **minimize SSR**, thus ensuring the best fit line.

---

## ⚠️ Key Assumptions You MUST Know

| Assumption 💡               | Meaning 📘                                                 |
| --------------------------- | ---------------------------------------------------------- |
| 📏 **Linearity**            | The relationship between inputs and output is linear.      |
| 🔗 **Independence**         | Observations (and residuals) are not correlated.           |
| 📉 **Homoscedasticity**     | The variance of errors is constant across all values of X. |
| 📊 **Normality of Errors**  | Residuals should be normally distributed for inference.    |
| 🔀 **No Multicollinearity** | Features should not be highly correlated with each other.  |
| 🚫 **No Endogeneity**       | Features should not correlate with the error term.         |

---

## 💎 Why is Linear Regression Important?

- 🧠 **Simplicity & Interpretability**  
  Easy to understand and interpret coefficients.

- 🏗️ **Foundational for Complex Models**  
  Many advanced models extend from linear regression.

- 📊 **Good Predictive Power**  
  When assumptions hold, predictions can be very accurate.

- 🧪 **Supports Hypothesis Testing**  
  Enables statistical analysis of relationships between variables.

---

## 🧠 Summary Formula (One Feature):

$$
h\_\theta(x) = \theta_0 + \theta_1 x
$$

- \( \theta_0 \): Intercept
- \( \theta_1 \): Slope (coefficient)
- \( x \): Independent variable
- \( h\_\theta(x) \): Prediction (dependent variable)
