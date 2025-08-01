# 📘 Machine Learning Theory Roadmap (Supervised + Unsupervised)

## 🔵 Supervised Learning

### 1. Linear Regression

- Least Squares Derivation ✅
- Normal Equation
- Gradient Descent (GD, SGD, Mini-batch)
- Convexity & Cost Surface
- Regularization (Ridge & Lasso): L2 vs L1 penalty
- Bias-Variance Tradeoff
- Gauss-Markov Theorem
- Maximum Likelihood Estimation (MLE)

### 2. Logistic Regression

- Log-likelihood derivation
- Gradient ascent & Newton's Method
- Hessian Matrix
- Exponential Family & Canonical Links
- Generalized Linear Models (GLM)
- Decision boundaries
- Regularization (L1/L2)

### 3. Decision Trees

- Entropy, Information Gain
- Gini Impurity
- ID3, CART algorithms
- Tree Pruning (Pre vs Post)
- Overfitting in Trees
- Recursive binary splits
- Cost-complexity pruning

### 4. Random Forests

- Bootstrap Aggregation (Bagging)
- Feature Randomness
- Out-of-bag Error
- Bias-Variance reduction
- Feature Importance
- Limitations of over-correlated trees

### 5. Support Vector Machines (SVM)

- Maximal Margin Classifier
- Soft Margin SVM
- Hinge Loss
- Dual Formulation
- Lagrangian, KKT Conditions
- Kernel Trick
- Common Kernels: RBF, Polynomial
- Convex Optimization (Quadratic Programming)

### 6. Naive Bayes

- Bayes Theorem
- Conditional Independence Assumption
- MAP vs MLE
- Gaussian Naive Bayes
- Multinomial/Bernoulli Variants
- Laplace Smoothing
- Log-space Computation

### 7. K-Nearest Neighbors (KNN)

- Instance-based Learning
- Curse of Dimensionality
- Euclidean vs Minkowski Distance
- Choosing K (Bias-Variance View)
- KD-Trees, Ball Trees
- Weighted KNN

### 8. General ML Theory

- VC Dimension
- Rademacher Complexity
- PAC Learning
- No Free Lunch Theorem
- Cross-validation Theory
- Model Selection Criteria (AIC, BIC)

---

## 🟣 Unsupervised Learning

### 1. K-Means Clustering

- Objective Function
- Lloyd’s Algorithm
- Convergence Properties
- Initialization (KMeans++)
- Distance Metrics
- Curse of Dimensionality
- Hard Assignment

### 2. Hierarchical Clustering

- Agglomerative vs Divisive
- Linkage Criteria (Single, Complete, Average)
- Dendrogram Visualization
- Time Complexity
- Cluster Cut-off Strategies

### 3. Gaussian Mixture Models (GMM)

- Latent Variable Model
- Mixture of Gaussians
- Expectation-Maximization (EM) Algorithm
- Log-likelihood Derivation
- Covariance Structures
- Comparison with K-Means

### 4. Principal Component Analysis (PCA)

- Variance Maximization
- Covariance Matrix
- Eigen Decomposition
- SVD
- Whitening
- Reconstruction Error
- Explained Variance Ratio

### 5. t-SNE

- Pairwise Similarities
- KL Divergence
- Gradient Descent
- Perplexity
- Crowding Problem
- Stochastic Nature

### 6. Autoencoders

- Encoder-Decoder Architecture
- Reconstruction Loss
- Bottleneck Layer
- Denoising and Sparse Autoencoders
- Variational Autoencoders (VAE)
- Reparameterization Trick
- Bayesian Interpretation

### 7. Independent Component Analysis (ICA)

- Blind Source Separation
- Non-Gaussian Assumption
- Kurtosis, Negentropy
- FastICA Algorithm

### 8. Density Estimation

- Histogram and KDE
- Bandwidth Selection
- Parametric Estimation
- Maximum Likelihood Estimation
- Applications in Anomaly Detection

### 9. Manifold Learning

- Isomap
- Locally Linear Embedding (LLE)
- Low-dimensional Manifold Assumption
- Jacobian and Hessian Constraints

### 10. Cross-cutting Concepts

- Curse of Dimensionality
- Non-convex Optimization
- Eigenvalue Problems
- EM Algorithm
- Matrix Factorization (SVD, NMF)

---

# 📘 Machine Learning Theory Roadmap (Supervised + Unsupervised)

This document provides an in-depth theoretical and mathematical breakdown of core Machine Learning models, similar to Stanford CS229 or research-oriented learning paths.

---

## 🔵 Supervised Learning

### 1. Linear Regression

- Least Squares Derivation
- Normal Equation
- Gradient Descent (GD, SGD, Mini-batch)
- Convexity & Cost Surface
- Regularization (Ridge & Lasso): L2 vs L1 penalty
- Bias-Variance Tradeoff
- Gauss-Markov Theorem
- Maximum Likelihood Estimation (MLE)

### 2. Logistic Regression

- Log-likelihood derivation
- Gradient ascent & Newton's Method
- Hessian Matrix
- Exponential Family & Canonical Links
- Generalized Linear Models (GLM)
- Decision boundaries
- Regularization (L1/L2)

### 3. Decision Trees

- Entropy, Information Gain
- Gini Impurity
- ID3, CART algorithms
- Tree Pruning (Pre vs Post)
- Overfitting in Trees
- Recursive binary splits
- Cost-complexity pruning

### 4. Random Forests

- Bootstrap Aggregation (Bagging)
- Feature Randomness
- Out-of-bag Error
- Bias-Variance reduction
- Feature Importance
- Limitations of over-correlated trees

### 5. Support Vector Machines (SVM)

- Maximal Margin Classifier
- Soft Margin SVM
- Hinge Loss
- Dual Formulation
- Lagrangian, KKT Conditions
- Kernel Trick
- Common Kernels: RBF, Polynomial
- Convex Optimization (Quadratic Programming)

### 6. Naive Bayes

- Bayes Theorem
- Conditional Independence Assumption
- MAP vs MLE
- Gaussian Naive Bayes
- Multinomial/Bernoulli Variants
- Laplace Smoothing
- Log-space Computation

### 7. K-Nearest Neighbors (KNN)

- Instance-based Learning
- Curse of Dimensionality
- Euclidean vs Minkowski Distance
- Choosing K (Bias-Variance View)
- KD-Trees, Ball Trees
- Weighted KNN

### 8. General ML Theory

- VC Dimension
- Rademacher Complexity
- PAC Learning
- No Free Lunch Theorem
- Cross-validation Theory
- Model Selection Criteria (AIC, BIC)

---

## 🟣 Unsupervised Learning

### 1. K-Means Clustering

- Objective Function
- Lloyd’s Algorithm
- Convergence Properties
- Initialization (KMeans++)
- Distance Metrics
- Curse of Dimensionality
- Hard Assignment

### 2. Hierarchical Clustering

- Agglomerative vs Divisive
- Linkage Criteria (Single, Complete, Average)
- Dendrogram Visualization
- Time Complexity
- Cluster Cut-off Strategies

### 3. Gaussian Mixture Models (GMM)

- Latent Variable Model
- Mixture of Gaussians
- Expectation-Maximization (EM) Algorithm
- Log-likelihood Derivation
- Covariance Structures
- Comparison with K-Means

### 4. Principal Component Analysis (PCA)

- Variance Maximization
- Covariance Matrix
- Eigen Decomposition
- SVD
- Whitening
- Reconstruction Error
- Explained Variance Ratio

### 5. t-SNE

- Pairwise Similarities
- KL Divergence
- Gradient Descent
- Perplexity
- Crowding Problem
- Stochastic Nature

### 6. Autoencoders

- Encoder-Decoder Architecture
- Reconstruction Loss
- Bottleneck Layer
- Denoising and Sparse Autoencoders
- Variational Autoencoders (VAE)
- Reparameterization Trick
- Bayesian Interpretation

### 7. Independent Component Analysis (ICA)

- Blind Source Separation
- Non-Gaussian Assumption
- Kurtosis, Negentropy
- FastICA Algorithm

### 8. Density Estimation

- Histogram and KDE
- Bandwidth Selection
- Parametric Estimation
- Maximum Likelihood Estimation
- Applications in Anomaly Detection

### 9. Manifold Learning

- Isomap
- Locally Linear Embedding (LLE)
- Low-dimensional Manifold Assumption
- Jacobian and Hessian Constraints

### 10. Cross-cutting Concepts

- Curse of Dimensionality
- Non-convex Optimization
- Eigenvalue Problems
- EM Algorithm
- Matrix Factorization (SVD, NMF)

---

---

## 📚 Recommended Books and When to Use Them

### 🔵 Supervised Learning

- **The Elements of Statistical Learning** (Hastie, Tibshirani, Friedman)  
  → Best for theoretical understanding of linear models, SVMs, boosting, trees.

- **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** (Aurelien Geron)  
  → Great for practical intuition, coding, and understanding how models work in real projects.

- **Understanding Machine Learning: From Theory to Algorithms** (Shalev-Shwartz & Ben-David)  
  → Ideal for foundational ML theory like VC dimension, generalization, algorithms.

- **Pattern Recognition and Machine Learning** (Christopher Bishop)  
  → Covers linear models, kernel methods, Bayesian reasoning, and EM in detail.

- **Convex Optimization** (Boyd & Vandenberghe)  
  → Essential for understanding optimization theory behind SVM, Logistic Regression, etc.

---

### 🟣 Unsupervised Learning

- **Probabilistic Machine Learning** (Kevin Murphy)  
  → Best for EM algorithm, graphical models, latent variable models, density estimation.

- **The Elements of Statistical Learning**  
  → Useful for PCA, unsupervised clustering, and dimensionality reduction.

- **Deep Learning** (Ian Goodfellow, Bengio, Courville)  
  → Excellent for understanding Autoencoders, VAEs, and optimization in neural networks.

- **Pattern Recognition and Machine Learning** (Bishop)  
  → Crucial for GMM, ICA, and Bayesian models.

- **Mathematics for Machine Learning** (Deisenroth et al.)  
  → Helpful for building foundational skills in linear algebra, calculus, and optimization used across both paradigms.

---

> ✨ Tip: Use _Hands-On ML_ alongside any theoretical book to reinforce learning with implementation.
> For deep math, rotate between _Bishop_, _Murphy_, and _CS229 Notes_.

---
