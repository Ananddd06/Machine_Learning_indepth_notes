# Support Vector Machine (SVM)

## Definition

A **Support Vector Machine (SVM)** is a supervised learning algorithm
used for **classification** and **regression** tasks. Its primary goal
is to find an **optimal hyperplane** that best separates data points of
different classes with the **maximum margin**.

## Core Idea & Formula

- For a binary classification problem, SVM tries to find a hyperplane
  that maximizes the **margin** between the closest data points of
  both classes (**support vectors**).

- The hyperplane is defined as:

  $$
     w\^T x + b = 0
  $$

  where:

  - \\( w \\): weight vector (normal to the hyperplane)
  - \\( b \\): bias term
  - \\( x \\): input feature vector

- The decision function for classification is:

  \\( f(x) = ext{sign}(w\^T x + b) \\)

## Assumptions of SVM

- Data is **somewhat separable** (can be linearly separable or made
  separable in higher-dimensional space using kernels).
- Classes are **well-defined**, and the labels are correct.
- The margin (distance between support vectors and hyperplane) should
  be meaningful for classification.

## Types of SVM

1.  **Linear SVM** - Works when data is linearly separable.\
2.  **Non-Linear SVM (Kernel SVM)** - Uses kernel trick (Polynomial,
    RBF, Sigmoid) to handle non-linear data.

## Advantages

- Works well for high-dimensional data.
- Effective when the number of features \> number of samples.
- Robust against overfitting, especially in high-dimensional spaces.

## Disadvantages

- Computationally expensive for large datasets.
- Choice of kernel and hyperparameters is crucial.
- Not efficient on datasets with a lot of noise.

## Real-World Examples

### 1. Email Spam Detection

- **Problem:** Classify emails as spam or not spam.
- **How SVM helps:** Each email is represented as a feature vector.
  SVM finds the hyperplane that separates spam from non-spam emails.

### 2. Handwriting Recognition

- **Problem:** Recognize handwritten digits (0--9).
- **How SVM helps:** Each image is converted into pixels/features. SVM
  classifies images into the correct digit.
- **Example:** Postal code recognition by postal services.

### 3. Face Detection

- **Problem:** Detect if an image contains a face.
- **How SVM helps:** Features are extracted from images (like edges or
  shapes). SVM separates face vs non-face regions.

### 4. Medical Diagnosis

- **Problem:** Predict diseases from patient data (e.g., cancer
  detection).
- **How SVM helps:** Features include patient biomarkers, test
  results, or gene expression levels. SVM classifies patients into
  disease vs healthy.
- **Example:** Breast cancer detection using mammogram images.

### 5. Sentiment Analysis

- **Problem:** Classify text as positive or negative sentiment.
- **How SVM helps:** Text converted into feature vectors using TF-IDF
  or embeddings. SVM finds the decision boundary separating positive
  vs negative reviews.

## Summary

SVM is a powerful supervised learning algorithm that constructs a decision boundary with maximum margin. While linear SVMs are suited for linearly separable data, kernelized SVMs extend this power to handle complex, non-linear datasets effectively.
