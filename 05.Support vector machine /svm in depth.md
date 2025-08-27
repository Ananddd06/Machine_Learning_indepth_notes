# Support Vector Machine (SVM) — In-Depth Explanation

## 1. The Goal of SVM

Support Vector Machines (SVM) are supervised learning algorithms used for **binary classification**. The central idea is:

- Find a **hyperplane** that separates the two classes.
- The hyperplane should maximize the **margin** (the distance between the hyperplane and the closest data points).
- Data points that lie closest to the hyperplane are called **support vectors**.

### Hyperplane Equation

The hyperplane is defined as:

$$
 h(x) = w^T x + b
$$

Where:

- \$w\$ = weight vector (normal to the hyperplane)
- \$b\$ = bias (intercept)
- \$x\$ = input feature vector

A point is classified as:

$$
\hat{y} = \text{sign}(w^T x + b)
$$

---

## 2. Margin Intuition

For a data point \$(x_i, y_i)\$, where \$y_i \in {-1, +1}\$:

- If \$y_i (w^T x_i + b) \geq 1\$, the point is **correctly classified and outside the margin**.
- If \$y_i (w^T x_i + b) < 1\$, the point is **inside the margin or misclassified**.

The margin width is:

$$
\text{Margin} = \frac{2}{||w||}
$$

So maximizing the margin is equivalent to minimizing \$||w||^2\$.

---

## 3. Hinge Loss Function

SVM uses **hinge loss** to penalize misclassified or margin-violating points:

$$
L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i (w^T x_i + b)) + \lambda ||w||^2
$$

Where:

- The first term enforces **classification correctness**.
- The second term (\$\lambda ||w||^2\$) is a **regularization penalty** to prevent overfitting.

---

## 4. Gradient Descent for SVM

We minimize hinge loss using **gradient descent**. The gradients are different depending on whether a point is correctly classified or not.

### Case 1: Correctly Classified & Outside Margin

If:

$$
y_i (w^T x_i + b) \geq 1
$$

- Loss contribution = \$0\$ (no hinge penalty)
- Gradient of weights:

$$
\frac{\partial L}{\partial w} = 2 \lambda w
$$

- Gradient of bias:

$$
\frac{\partial L}{\partial b} = 0
$$

This means only the regularization term affects the gradient.

---

### Case 2: Inside Margin or Misclassified

If:

$$
y_i (w^T x_i + b) < 1
$$

- Loss contribution = \$1 - y_i(w^T x_i + b)\$
- Gradient of weights:

$$
\frac{\partial L}{\partial w} = 2 \lambda w - y_i x_i
$$

- Gradient of bias:

$$
\frac{\partial L}{\partial b} = -y_i
$$

This pushes the model to correct the misclassified point.

---

## 5. Why Gradient Looks Like That

- **Term \$2 \lambda w\$**: comes from differentiating the regularization \$||w||^2\$.
- **Term \$-y_i x_i\$**: comes from differentiating the hinge loss \$\max(0, 1 - y_i (w^T x_i + b))\$.
- **Bias gradient \$-y_i\$**: since \$b\$ appears linearly inside the hinge loss term.

This ensures:

- Correctly classified points far from the boundary only shrink weights via regularization.
- Misclassified points directly influence \$w\$ and \$b\$ to shift the boundary.

---

## 6. Final Algorithm Steps

1. Initialize \$w\$, \$b\$ randomly (or zeros).
2. For each epoch:

   - Loop through all training samples \$(x_i, y_i)\$.
   - Compute condition \$y_i (w^T x_i + b)\$.
   - Update \$w\$ and \$b\$ using the correct gradient rule.

3. Repeat until convergence.

---

## 7. Geometric Interpretation

- SVM does not just separate classes—it finds the **maximum margin separator**.
- The support vectors (closest points to boundary) are the only ones that matter for defining the hyperplane.
- The hinge loss ensures misclassified or “too close” points exert pressure to adjust the hyperplane.

---

## 8. Summary

- **Hyperplane**: \$w^T x + b = 0\$
- **Decision Rule**: \$\hat{y} = \text{sign}(w^T x + b)\$
- **Loss**: Hinge loss + Regularization
- **Optimization**: Gradient descent with case-specific updates
- **Support Vectors**: Points on or inside the margin that define the boundary
