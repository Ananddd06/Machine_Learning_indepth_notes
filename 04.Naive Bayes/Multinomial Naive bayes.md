# naive_bayes_models.py

import numpy as np
import pandas as pd

class NaiveBayesGaussian:
def **init**(self):
self.priors = {}
self.mean_var = {}
self.classes = None

```
def fit(self, df, Y):
    self.classes = np.unique(df[Y])
    self.priors = {c: np.mean(df[Y] == c) for c in self.classes}
    features = list(df.columns[:-1])
    for c in self.classes:
        X_c = df[df[Y]==c][features]
        self.mean_var[c] = {'mean': X_c.mean().values, 'var': X_c.var().values + 1e-6}

def _gaussian_pdf(self, x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean)**2 / (2 * var)))

def predict(self, X):
    predictions = []
    for x in X:
        posteriors = {}
        for c in self.classes:
            likelihood = np.prod(self._gaussian_pdf(x, self.mean_var[c]['mean'], self.mean_var[c]['var']))
            posteriors[c] = self.priors[c] * likelihood
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)
```

class NaiveBayesCategorical:
def **init**(self, alpha=1.0):
self.priors = {}
self.classes = None
self.alpha = alpha

```
def fit(self, df, Y):
    self.classes = np.unique(df[Y])
    self.priors = {c: np.mean(df[Y] == c) for c in self.classes}

def _categorical_likelihood(self, df, feat_name, feat_val, label, Y):
    df_c = df[df[Y] == label]
    return (len(df_c[df_c[feat_name] == feat_val]) + self.alpha) / (len(df_c) + self.alpha * len(df[feat_name].unique()))

def predict(self, df, X, Y):
    features = list(df.columns[:-1])
    predictions = []
    for x in X:
        posteriors = {}
        for c in self.classes:
            likelihood = 1
            for i, feat in enumerate(features):
                likelihood *= self._categorical_likelihood(df, feat, x[i], c, Y)
            posteriors[c] = likelihood * self.priors[c]
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)
```

# Multinomial Naive Bayes Implementation and Explanation

This document explains the **Multinomial Naive Bayes (MNB)** classifier, along with a full Python implementation, with a detailed breakdown **function by function and line by line**.

---

## **1. What is Multinomial Naive Bayes?**

Multinomial Naive Bayes is a probabilistic classifier based on **Bayes' theorem**, commonly used for **count-based features** (e.g., word counts in text).

For a sample \$x = \[x_1, x_2, ..., x_n]\$, the posterior probability for class \$y\$ is:

$$
P(y|x) = \frac{P(x|y) P(y)}{\sum_{y'} P(x|y') P(y')}
$$

- \$P(y)\$ = prior probability of class y.
- \$P(x|y)\$ = likelihood of features given the class.
- Naive assumption: features are conditionally independent given the class.

Laplace smoothing is applied to avoid zero probabilities for unseen features.

---

## **2. Python Implementation with Explanations**

### **Step 1: Initialization (`__init__`)**

```python
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha        # Laplace smoothing parameter. Helps avoid zero probabilities.
        self.priors = {}          # Stores P(Y=c), prior probability of each class.
        self.likelihoods = {}     # Stores P(X_j|Y=c), conditional probability of features given classes.
        self.classes = None       # Unique classes in the dataset.
```

**Explanation:**

- `alpha`: Laplace smoothing factor. Prevents zero probability if a feature never appears in a class.
- `priors`: Stores prior probabilities for each class.
- `likelihoods`: Stores conditional probabilities for all features in each class.
- `classes`: Keeps track of all unique class labels.

---

### **Step 2: Fitting the Model (`fit`)**

```python
    def fit(self, df, Y):
        self.classes = np.unique(df[Y])
        features = list(df.columns[:-1])

        # Calculate prior probabilities
        self.priors = {c: np.mean(df[Y] == c) for c in self.classes}

        # Calculate likelihoods with Laplace smoothing
        self.likelihoods = {c: {} for c in self.classes}
        for c in self.classes:
            df_c = df[df[Y]==c]
            total_count = df_c[features].sum().sum()  # Total count of all features in class c
            V = len(features)  # Number of features
            for feat in features:
                count = df_c[feat].sum()
                self.likelihoods[c][feat] = (count + self.alpha) / (total_count + self.alpha * V)
```

**Explanation line by line:**

1. `self.classes = np.unique(df[Y])` – Extract all unique class labels.
2. `features = list(df.columns[:-1])` – List of feature columns.
3. `self.priors = {...}` – Compute **prior probabilities** for each class.
4. `df_c = df[df[Y]==c]` – Filter dataset for class `c`.
5. `total_count = df_c[features].sum().sum()` – Sum of all feature counts for class `c`.
6. `V = len(features)` – Number of features.
7. `self.likelihoods[c][feat] = ...` – Compute **conditional probabilities with Laplace smoothing** to avoid zero probabilities.

---

### **Step 3: Predict Function (`predict`)**

```python
    def predict(self, X):
        predictions = []
        features = list(X.columns)

        for _, x in X.iterrows():
            posteriors = {}
            for c in self.classes:
                likelihood_product = 1
                for feat in features:
                    likelihood_product *= self.likelihoods[c][feat] ** x[feat]
                posteriors[c] = self.priors[c] * likelihood_product

            predictions.append(max(posteriors, key=posteriors.get))

        return np.array(predictions)
```

**Explanation line by line:**

1. Loop over each sample `x` in the dataset.
2. For each class `c`, initialize `likelihood_product` = 1.
3. Multiply likelihoods for each feature raised to the feature count (`x[feat]`) – **multinomial feature model**.
4. Multiply by prior probability → posterior.
5. Select class with **maximum posterior probability** using `max(posteriors, key=posteriors.get)`.
6. Return an array of predicted labels.

---

### **Step 4: Example Usage**

```python
if __name__ == "__main__":
    data = pd.DataFrame({
        'feat1': [2,1,0,1],
        'feat2': [0,1,2,1],
        'feat3': [1,0,1,0],
        'label': [0,1,0,1]
    })

    X_train = data[['feat1','feat2','feat3']]
    y_train = data['label']

    nb = MultinomialNaiveBayes(alpha=1.0)
    nb.fit(data, 'label')

    X_test = pd.DataFrame({'feat1':[1,0],'feat2':[1,2],'feat3':[0,1]})
    y_pred = nb.predict(X_test)
    print("Predicted labels:", y_pred)
```

**Explanation:**

- Creates a **toy dataset** with feature counts.
- Fits Multinomial Naive Bayes with Laplace smoothing `alpha=1`.
- Predicts labels for new samples.
- Laplace smoothing ensures no zero probability occurs for unseen feature counts.

---

### **Key Takeaways**

1. Multinomial NB works with **count-based features**.
2. **Laplace smoothing** prevents zero probabilities.
3. Predict by computing **posterior probabilities** and selecting the **class with maximum probability**.
4. Ideal for **text classification**, bag-of-words, or categorical count data.
