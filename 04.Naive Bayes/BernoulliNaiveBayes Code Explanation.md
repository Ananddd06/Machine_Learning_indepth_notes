# Naive Bayes Implementation and Explanation

This document explains the **Bernoulli Naive Bayes** classifier, along with a full Python implementation, with a detailed breakdown **function by function and line by line**.

---

## **1. What is Naive Bayes?**

Naive Bayes (NB) is a probabilistic classifier based on **Bayes' theorem**, assuming that features are **conditionally independent** given the class.

For a sample $x = [x_1, x_2, ..., x_n]$, the posterior probability for class $y$ is:

$$
P(y|x) = \frac{P(x|y) P(y)}{\sum_{y'} P(x|y') P(y')}
$$

- $P(y)$ = prior probability of class y.
- $P(x|y)$ = likelihood of features given the class.

Naive Bayes is called "Naive" because it assumes **feature independence**.

---

## **2. Bernoulli Naive Bayes**

### **Key Points:**

- Works with **binary features** (0 or 1).
- Uses **Laplace smoothing** to avoid zero probabilities.

### **Python Implementation with Explanations:**

#### **Step 1: Initialization Function (`__init__`)**

```python
class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.phi_y = None          # P(y=1)
        self.phi_j_y1 = None       # P(x_j=1|y=1)
        self.phi_j_y0 = None       # P(x_j=1|y=0)
```

**Explanation:**

- `alpha`: Laplace smoothing factor. Prevents zero probability for unseen features.
- `phi_y`: Prior probability of class 1 (fraction of samples with y=1).
- `phi_j_y1` / `phi_j_y0`: Conditional probabilities for each feature being 1 given class 1 or 0.

---

#### **Step 2: Fitting the Model (`fit`)**

```python
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.phi_y = np.mean(y)  # Prior probability of y=1

        # Conditional probabilities for y=1
        X_y1 = X[y == 1]
        self.phi_j_y1 = (np.sum(X_y1, axis=0) + self.alpha) / (X_y1.shape[0] + 2*self.alpha)

        # Conditional probabilities for y=0
        X_y0 = X[y == 0]
        self.phi_j_y0 = (np.sum(X_y0, axis=0) + self.alpha) / (X_y0.shape[0] + 2*self.alpha)
```

**Explanation line by line:**

1. `n_samples, n_features = X.shape` – Get dataset dimensions.
2. `self.phi_y = np.mean(y)` – Compute **prior probability** of class 1.
3. `X_y1 = X[y == 1]` – Select samples where label is 1.
4. `np.sum(X_y1, axis=0)` – Count how many times each feature is 1 in class 1.
5. `(count + alpha) / (total + 2*alpha)` – Apply **Laplace smoothing** to avoid zero probabilities.
6. Repeat similarly for class 0.

---

#### **Step 3: Likelihood Function (`_likelihood`)**

```python
    def _likelihood(self, x, y_val):
        if y_val == 1:
            phi = self.phi_j_y1
        else:
            phi = self.phi_j_y0

        likelihood = np.prod(np.power(phi, x) * np.power(1 - phi, 1 - x))
        return likelihood
```

**Explanation:**

- Selects the conditional probabilities based on class.
- Computes **Bernoulli likelihood**:

  - For a feature x_j = 1: multiply by phi_j (P(x_j=1|y))
  - For x_j = 0: multiply by (1 - phi_j)

- `np.prod(...)` multiplies probabilities across all features (independence assumption).
- Returns the probability P(x|y).

---

#### **Step 4: Posterior Probabilities (`predict_proba`)**

```python
    def predict_proba(self, X):
        probs = []
        for x in X:
            p_x_y1 = self._likelihood(x, 1)
            p_x_y0 = self._likelihood(x, 0)

            p_y1 = self.phi_y
            p_y0 = 1 - self.phi_y

            numerator1 = p_x_y1 * p_y1
            numerator0 = p_x_y0 * p_y0
            denom = numerator1 + numerator0

            probs.append([numerator0 / denom, numerator1 / denom])

        return np.array(probs)
```

**Explanation line by line:**

1. Loop over each sample `x` in the dataset.
2. Compute **likelihood** for each class using `_likelihood()`.
3. Multiply likelihood by **prior** for class → numerator.
4. Compute denominator = sum of numerators (normalization).
5. Append **normalized posterior probabilities** \[P(y=0|x), P(y=1|x)] to the list.
6. Return as a NumPy array.

---

#### **Step 5: Predict Function (`predict`)**

```python
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
```

**Explanation:**

- Calls `predict_proba` to get posterior probabilities.
- `np.argmax(..., axis=1)` selects the **class with the highest probability** for each sample.
- Returns an array of predicted labels.

---

#### **Step 6: Example Usage**

```python
if __name__ == "__main__":
    X = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ])
    y = np.array([1, 1, 0, 0, 1])

    nb = BernoulliNaiveBayes(alpha=1.0)
    nb.fit(X, y)

    X_test = np.array([[1, 0, 0], [0, 1, 1]])
    print("Predicted probabilities:\n", nb.predict_proba(X_test))
    print("Predicted labels:", nb.predict(X_test))
```

**Explanation:**

- Defines a **toy dataset**.
- Trains Bernoulli Naive Bayes with `alpha=1` (Laplace smoothing).
- Tests on new samples and prints:

  - **Posterior probabilities** for each class.
  - **Predicted labels** (0 or 1).

---

### **Key Takeaways**

1. **Binary features only** – Bernoulli NB.
2. **Laplace smoothing** prevents zero probabilities for unseen features.
3. **Naive assumption**: features are independent given the class.
4. Two-step prediction: calculate **posterior probabilities**, then pick **highest probability class**.

---
