
# 📘 Machine Learning Core Concepts – Extended Interview Guide

---

## 🔢 Linear Regression

### **Definition**
Linear Regression is a **supervised learning algorithm** used for **predicting a continuous dependent variable** based on one or more independent variables.

### **Mathematical Form**
\[
y = X \theta + \epsilon
\]
- \( y \): target
- \( X \): input features
- \( \theta \): model parameters
- \( \epsilon \): error term

### **Objective**
Minimize the **Mean Squared Error (MSE)**:
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
\]

---

## ⚙️ Optimization Algorithms

### 1. **Gradient Descent (GD)**
- Uses the full dataset to compute gradient.
- Update Rule:
\[
\theta = \theta - \eta \cdot \nabla_\theta J(\theta)
\]

### 2. **Stochastic Gradient Descent (SGD)**
- Uses a **single data point** per iteration.
- Faster but noisy updates.

### 3. **Mini-Batch Gradient Descent**
- Uses a **subset (mini-batch)** of data.
- Balances speed and stability.

---

### 4. **Momentum**
- Accelerates updates in consistent directions using velocity.
- Update Rule:
\[
v_t = \gamma v_{t-1} + \eta \nabla J(\theta)
\quad ; \quad \theta = \theta - v_t
\]
- Helps avoid local minima and oscillations.

---

### 5. **RMSProp**
- Adapts learning rate per parameter using moving average of squared gradients.
- Update Rule:
\[
s_t = \beta s_{t-1} + (1 - \beta) g_t^2
\quad ; \quad \theta = \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
\]
- Reduces oscillations in steep directions.

---

### 6. **Adam (Adaptive Moment Estimation)**
- Combines Momentum and RMSProp.
- Uses moving averages of gradients and squared gradients.
- Update Rule:
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta = \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]

---

## 🧮 Regularization Techniques

### 1. **Ridge Regression (L2 Regularization)**

#### **Definition**
- Adds a penalty term \( \lambda \sum \theta^2 \) to the loss function to **shrink coefficients**.
- Cost Function:
\[
J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} \theta_j^2
\]

#### **Why Use It?**
- Prevents overfitting by penalizing large weights.
- Keeps all features but shrinks them.

---

### 2. **Lasso Regression (L1 Regularization)**

#### **Definition**
- Adds a penalty term \( \lambda \sum |\theta| \) to the loss.
- Cost Function:
\[
J(\theta) = \text{MSE} + \lambda \sum_{j=1}^{n} |\theta_j|
\]

#### **Why Use It?**
- Shrinks some coefficients to **zero**, effectively performing **feature selection**.

---

## 🧠 When to Use What?

| Situation | Technique |
|----------|-----------|
| Large number of irrelevant features | Lasso (L1) |
| Correlated features | Ridge (L2) |
| Want sparsity and feature selection | Lasso |
| Want stable, small coefficients | Ridge |
| Need both L1 and L2 | ElasticNet |

---

## 📌 Summary Table

| Optimizer | Learns Rate | Adaptive | Momentum | Use Case |
|----------|-------------|----------|----------|----------|
| GD | Fixed | ❌ | ❌ | Small data |
| SGD | Fixed | ❌ | ❌ | Online learning |
| Mini-batch GD | Fixed | ❌ | ❌ | Large datasets |
| Momentum | Fixed | ❌ | ✅ | Faster convergence |
| RMSProp | Adaptive | ✅ | ❌ | RNNs, noisy gradients |
| Adam | Adaptive | ✅ | ✅ | Default for deep learning |

---
