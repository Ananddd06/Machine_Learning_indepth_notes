## ✅ Key Strengths of the Linear Regression OOP Code

### 📉 Cost Function:

You use **Mean Squared Error (MSE)**:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$

Correctly implemented in `compute_cost()`.

---

### 📈 Hypothesis Function:

$$
h_\theta(x) = X\theta
$$

Correctly used via matrix multiplication: `X @ self.theta`.

---

### 📊 Batch Gradient Descent (BGD):

Full dataset used in each epoch:

$$
\theta := \theta - \alpha \cdot \frac{1}{m} X^\top (X\theta - y)
$$

✅ Correct.

---

### 🧩 Mini-Batch Gradient Descent:

Subset of samples used in each step. Well-shuffled and iterated in mini-batches.

Each batch update:

$$
\theta := \theta - \alpha \cdot \frac{1}{b} X_b^\top (X_b\theta - y_b)
$$

Where \$b\$ is the batch size, \$X_b\$ is the mini-batch, and \$y_b\$ is the corresponding target vector.

✅ Correct.

---

### 🔁 Stochastic Gradient Descent (SGD):

One sample used at a time:

$$
\theta := \theta - \alpha (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}
$$

✅ Correct, with \$x^{(i)}\$ reshaped and flattened properly.

---

### 📉 Plotting & Prediction Methods:

Good for visualizing loss and making predictions using:

- `plot_cost()`
- `predict()`

---

### 🧠 Summary:

- Modular
- Follows OOP principles
- Mathematically sound
- Extendable (you can add early stopping, regularization, etc.)
