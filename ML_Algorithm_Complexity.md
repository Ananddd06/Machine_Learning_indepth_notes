# 📊 Machine Learning Algorithm Complexity

This document summarizes the **time and space complexity** of various machine learning algorithms. Notation used:

- **n**: number of samples
- **d**: number of features
- **k**: number of clusters / classes / neighbors
- **t**: number of trees
- **i**: number of iterations
- **sv**: number of support vectors
- **L**: number of layers
- **T**: sequence length (in RNN/Transformer)

---

## 🧠 1. Supervised Learning

| Algorithm               | Training Time Complexity   | Prediction Time Complexity | Space Complexity |
| ----------------------- | -------------------------- | -------------------------- | ---------------- |
| Linear Regression       | O(nd^2 + d^3), O(nid) (GD) | O(d)                       | O(d)             |
| Ridge Regression        | O(nd^2 + d^3)              | O(d)                       | O(d)             |
| Logistic Regression     | O(nid)                     | O(d)                       | O(d)             |
| SVM (Linear)            | O(nd) to O(n^2d)           | O(d)                       | O(sv \* d)       |
| SVM (Kernel)            | O(n^2 d + n^3)             | O(sv \* d)                 | O(sv \* d)       |
| Naive Bayes             | O(nd)                      | O(d)                       | O(cd)            |
| k-NN                    | O(1)                       | O(nd)                      | O(nd)            |
| Decision Tree           | O(n d log n)               | O(log n)                   | O(n)             |
| Random Forest           | O(t \* n d log n)          | O(t \* log n)              | O(tn)            |
| Gradient Boosting (GBM) | O(t \* n log n)            | O(t \* log n)              | O(tn)            |
| XGBoost/LightGBM        | O(tn log n)                | O(t log n)                 | O(tn)            |
| MLP (Neural Net)        | O(nidL)                    | O(dL)                      | O(#params)       |

---

## 🎲 2. Unsupervised Learning

| Algorithm               | Training Time Complexity | Prediction Time Complexity | Space Complexity |
| ----------------------- | ------------------------ | -------------------------- | ---------------- |
| k-Means                 | O(nkd i)                 | O(kd)                      | O(kd)            |
| K-Medoids               | O(k(n-k)^2) to O(kn^2)   | O(kd)                      | O(n)             |
| Gaussian Mixture Model  | O(nkd i)                 | O(kd)                      | O(kd)            |
| Hierarchical Clustering | O(n^2 log n)             | -                          | O(n^2)           |
| DBSCAN                  | O(n log n)               | -                          | O(n)             |
| PCA (Full SVD)          | O(nd^2 + d^3)            | O(kd)                      | O(kd)            |
| PCA (Truncated SVD)     | O(ndk)                   | O(kd)                      | O(kd)            |
| t-SNE                   | O(n^2)                   | -                          | O(n^2)           |
| UMAP                    | O(n log n)               | -                          | O(n)             |

---

## ⚙️ 3. Ensemble Methods

| Algorithm        | Training Time Complexity       | Prediction Time Complexity | Space Complexity   |
| ---------------- | ------------------------------ | -------------------------- | ------------------ |
| Bagging          | O(t \* n d log n)              | O(t log n)                 | O(tn)              |
| AdaBoost         | O(tnd)                         | O(t)                       | O(tn)              |
| XGBoost/LightGBM | O(tn log n)                    | O(t log n)                 | O(tn)              |
| Stacking         | Sum of base model complexities | Sum of base models         | Sum of base models |

---

## 🔍 4. Dimensionality Reduction

| Algorithm     | Training Time Complexity | Space Complexity |
| ------------- | ------------------------ | ---------------- |
| PCA (SVD)     | O(nd^2 + d^3)            | O(d^2)           |
| Truncated PCA | O(ndk)                   | O(kd)            |
| LDA           | O(nd^2)                  | O(d^2)           |
| t-SNE         | O(n^2)                   | O(n^2)           |
| UMAP          | O(n log n)               | O(n)             |

---

## 🧪 5. Anomaly Detection

| Algorithm         | Training Time   | Inference Time | Space Complexity |
| ----------------- | --------------- | -------------- | ---------------- |
| Isolation Forest  | O(t \* n log n) | O(t log n)     | O(tn)            |
| One-Class SVM     | O(n^3)          | O(sv \* d)     | O(sv \* d)       |
| Elliptic Envelope | O(nd^2)         | O(d^2)         | O(d^2)           |

---

## 🔮 6. Deep Learning (Rough Estimates)

| Architecture | Training Time    | Inference Time | Space Complexity  |
| ------------ | ---------------- | -------------- | ----------------- |
| MLP          | O(nidL)          | O(dL)          | O(#params)        |
| CNN          | O(n d^2 k^2 f^2) | O(f^2 k^2)     | O(filters \* f^2) |
| RNN / LSTM   | O(n T d^2)       | O(T d^2)       | O(d^2)            |
| Transformer  | O(n T^2 d)       | O(T^2 d)       | O(T d^2)          |
