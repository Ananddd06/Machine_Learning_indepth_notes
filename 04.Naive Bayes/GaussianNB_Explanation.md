# Gaussian and Categorical Naive Bayes Implementation and Explanation

This document explains **Gaussian Naive Bayes** and **Categorical Naive Bayes** with detailed Python code and step-by-step explanations for each function.

---

## **1. Prior Probability Function**

```python
def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i]) / len(df))
    return prior
```

**Explanation:**

- Computes the **prior probability P(Y=y)** for each class.
- `df[Y].unique()` finds all classes.
- Count of samples in class / total samples → probability.

---

## **2. Gaussian Naive Bayes Likelihood Function**

```python
def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2)))
    return p_x_given_y
```

**Explanation:**

- Selects only samples with label `Y=label`.
- Computes **mean and standard deviation** of the feature for that class.
- Applies **Gaussian probability density function** for `feat_val`.
- Returns likelihood P(X=feat_val | Y=label).

---

## **3. Gaussian Naive Bayes Prediction Function**

```python
def naive_bayes_gaussian(df, X, Y):
    features = list(df.columns)[:-1]
    prior = calculate_prior(df, Y)

    Y_pred = []
    labels = sorted(list(df[Y].unique()))

    for x in X:
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        post_prob = [likelihood[j] * prior[j] for j in range(len(labels))]
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)
```

**Explanation:**

1. Extract **feature names** (exclude label column).
2. Compute **priors** using `calculate_prior()`.
3. Loop over **each test sample `x`**.
4. Compute **likelihood** for each class as product of Gaussian PDFs for all features.
5. Multiply by prior → **unnormalized posterior probability**.
6. Pick class with **maximum posterior**.

---

## **4. Example Test for Gaussian NB**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

train, test = train_test_split(data, test_size=0.2, random_state=41)
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values
Y_pred = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")

print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))
```

**Explanation:**

- Split dataset into train/test.
- Predict using Gaussian NB.
- Compute **confusion matrix** and **F1 score**.

---

## **5. Converting Continuous Features to Categorical**

```python
import pandas as pd

# Convert continuous features into 3 bins
data["cat_mean_radius"] = pd.cut(data["mean_radius"], bins=3, labels=[0,1,2])
data["cat_mean_texture"] = pd.cut(data["mean_texture"], bins=3, labels=[0,1,2])
data["cat_mean_smoothness"] = pd.cut(data["mean_smoothness"], bins=3, labels=[0,1,2])

# Drop original continuous columns
data = data.drop(columns=["mean_radius", "mean_texture", "mean_smoothness"])
```

**Explanation:**

- Continuous variables are **binned into categories** 0, 1, 2.
- Makes dataset compatible for **categorical Naive Bayes**.

---

## **6. Categorical Naive Bayes Likelihood Function**

```python
def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y
```

**Explanation:**

- Select samples with label `Y=label`.
- Compute **probability of feature value** in that class: count / total samples in class.
- Returns P(X=feat_val | Y=label).

---

## **7. Categorical Naive Bayes Prediction Function**

```python
def naive_bayes_categorical(df, X, Y):
    features = list(df.columns)[:-1]
    prior = calculate_prior(df, Y)

    Y_pred = []
    labels = sorted(list(df[Y].unique()))

    for x in X:
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        post_prob = [likelihood[j] * prior[j] for j in range(len(labels))]
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)
```

**Explanation:**

- Similar to Gaussian NB, but likelihoods are **categorical counts** instead of Gaussian PDFs.
- Posterior probability = likelihood × prior.
- Predict class with **maximum posterior**.

---

## **8. Example Test for Categorical NB**

```python
train, test = train_test_split(data, test_size=0.2, random_state=41)
X_test = test.iloc[:, :-1].values
Y_test = test.iloc[:, -1].values
Y_pred = naive_bayes_categorical(train, X=X_test, Y="diagnosis")

print(confusion_matrix(Y_test, Y_pred))
print(f1_score(Y_test, Y_pred))
```

**Explanation:**

- Split dataset and predict using **categorical Naive Bayes**.
- Evaluate with **confusion matrix** and **F1 score**.

---

### **Key Takeaways**

1. **Gaussian NB**: For continuous features using Gaussian PDFs.
2. **Categorical NB**: For discretized/categorical features using counts.
3. Both use **prior × likelihood → posterior**, then predict class with **maximum posterior**.
4. Works for multi-class classification.
