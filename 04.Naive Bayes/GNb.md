# naive\_bayes\_models.py

import numpy as np
import pandas as pd

class NaiveBayesGaussian:
def **init**(self):
self.priors = {}
self.mean\_var = {}
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
