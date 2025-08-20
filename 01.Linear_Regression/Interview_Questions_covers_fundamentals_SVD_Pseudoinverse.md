# 📘 Top 50 Interview Questions and Answers on Linear Regression, SVD & Pseudoinverse

---

## 🔹 Section 1: Foundations of Linear Regression

### 1. What is linear regression?

> Linear regression is a method to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

---

### 2. What is the hypothesis function in linear regression?

> $$ h\_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... + \\theta_n x_n $$

---

### 3. What are the assumptions of linear regression?

> Linearity, Independence, Homoscedasticity, No multicollinearity, Normal distribution of residuals, No endogeneity.

---

### 4. How do you interpret the coefficients in linear regression?

> Each coefficient represents the expected change in the dependent variable for a one-unit change in the corresponding independent variable, keeping others constant.

---

### 5. What is the cost function in linear regression?

> The Mean Squared Error (MSE):  
> $$ J(\\theta) = \\frac{1}{2m} \\sum*{i=1}^{m} (h*\\theta(x^{(i)}) - y^{(i)})^2 $$

---

### 6. What is Ordinary Least Squares (OLS)?

> OLS is a method that estimates the parameters by minimizing the sum of squared residuals.

---

### 7. Why is the cost function convex in linear regression?

> Because the second derivative (Hessian) is positive semi-definite. It has only one global minimum.

---

### 8. What is the Normal Equation?

> $$ \\theta = (X^T X)^{-1} X^T y $$

---

### 9. When is the Normal Equation not suitable?

> When \( X^T X \) is non-invertible (singular), e.g., due to redundant features or when \( m < n \).

---

### 10. What are residuals?

> The difference between actual and predicted values:  
> $$ e_i = y_i - \\hat{y}\_i $$

---

## 🔹 Section 2: SVD (Singular Value Decomposition)

### 11. What is SVD?

> A matrix decomposition:  
> $$ X = U \\Sigma V^T $$  
> where:
>
> - \( U \): Left singular vectors
> - \( \\Sigma \): Diagonal matrix of singular values
> - \( V^T \): Right singular vectors

---

### 12. Why is SVD important in linear regression?

> It allows computation of the pseudoinverse even when \( X^T X \) is singular.

---

### 13. What is the computational complexity of SVD?

> Typically \( O(n^2 m) \), faster than matrix inversion for large \( n \).

---

### 14. What properties does the \( U \) matrix have?

> Orthonormal columns: \( U^T U = I \)

---

### 15. What does \( \\Sigma \) contain?

> The singular values (square roots of the eigenvalues of \( X^T X \))

---

### 16. How do you compute the pseudoinverse from SVD?

> $$ X^+ = V \\Sigma^+ U^T $$

---

### 17. What is \( \\Sigma^+ \)?

> The inverse of non-zero singular values in \( \\Sigma \), transposed.

---

### 18. Why is SVD numerically stable?

> It avoids explicit inversion and handles rank deficiency and ill-conditioning.

---

### 19. Can SVD work if the matrix is not full rank?

> Yes — it's one of the main reasons SVD is used in linear regression.

---

### 20. When should you prefer SVD over the Normal Equation?

> - When \( X^T X \) is singular or nearly singular
> - High-dimensional data
> - More numerical stability

---

## 🔹 Section 3: Pseudoinverse (Moore-Penrose)

### 21. What is the Moore-Penrose pseudoinverse?

> A generalization of the matrix inverse that works for non-square or singular matrices.

---

### 22. When do we use the pseudoinverse in linear regression?

> To compute:
> $$ \\theta = X^+ y $$  
> when \( X^T X \) is not invertible.

---

### 23. How do you compute \( X^+ \) using SVD?

> If \( X = U \\Sigma V^T \), then:  
> $$ X^+ = V \\Sigma^+ U^T $$

---

### 24. What happens to small singular values in \( \\Sigma \)?

> They're thresholded to 0 before inversion to prevent numerical instability.

---

### 25. What are the 4 properties of the Moore-Penrose pseudoinverse?

1. \( X X^+ X = X \)
2. \( X^+ X X^+ = X^+ \)
3. \( (X X^+)^T = X X^+ \)
4. \( (X^+ X)^T = X^+ X \)

---

### 26. How does the pseudoinverse help with multicollinearity?

> It handles rank deficiency without requiring feature elimination.

---

### 27. What is the geometric intuition behind pseudoinverse?

> Projects the target vector onto the column space of \( X \) in the least-squares sense.

---

### 28. Is the pseudoinverse unique?

> Yes — it is uniquely defined for any matrix.

---

### 29. Is the pseudoinverse always defined?

> Yes, even for non-square, non-invertible matrices.

---

### 30. What is a real-world use case of pseudoinverse?

> Solving overdetermined or underdetermined systems in ML, image reconstruction, or NLP.

---

## 🔹 Section 4: Edge Cases & Theory Questions

### 31. What does it mean if \( X^T X \) is singular?

> There's multicollinearity or \( m < n \), so columns of \( X \) are linearly dependent.

---

### 32. How do you check if \( X^T X \) is singular?

> Compute determinant or check condition number:  
> $$ \\text{cond}(X^T X) \\gg 1 $$

---

### 33. What's the difference between inverse and pseudoinverse?

> Inverse exists only for square, full-rank matrices; pseudoinverse is more general.

---

### 34. Can you use gradient descent instead of the pseudoinverse?

> Yes — when data is large or doesn't fit in memory, or when you want an iterative solution.

---

### 35. What is multicollinearity and how does it affect regression?

> When two or more features are highly correlated → unstable coefficients.

---

### 36. What are some ways to deal with multicollinearity?

> - Remove features
> - Use Ridge Regression
> - Use SVD + pseudoinverse

---

### 37. How is PCA related to SVD?

> PCA is performed using SVD on the centered data matrix.

---

### 38. What’s the difference between SVD and eigendecomposition?

> SVD works for any matrix; eigendecomposition only for square matrices.

---

### 39. What is a condition number?

> Measures sensitivity of output to input.  
> High condition number → ill-conditioned → unstable inversion.

---

### 40. How does regularization solve singularity issues?

> Adds a penalty (e.g., L2) to make \( X^T X + \\lambda I \) invertible.

---

## 🔹 Section 5: Real-World + Applied Scenarios

### 41. Why is SVD used in recommender systems?

> For latent factor modeling (matrix factorization), dimensionality reduction.

---

### 42. How is the pseudoinverse used in image compression?

> Reduce dimensionality and reconstruct approximation using top singular values.

---

### 43. When is batch gradient descent preferred over closed-form?

> When dataset is too large to fit in memory or when features are sparse.

---

### 44. Is SVD affected by feature scaling?

> Yes — singular values change. Standardizing features is best before applying SVD.

---

### 45. How does pseudoinverse relate to Ridge Regression?

> Ridge can be computed as:  
> $$ \\theta = (X^T X + \\lambda I)^{-1} X^T y $$  
> which avoids singularity too.

---

### 46. What’s a practical issue with inverting \( X^T X \)?

> Computationally expensive (\( O(n^3) \)) and unstable for large or sparse matrices.

---

### 47. Why is numerical stability important in linear models?

> Prevents exploding/vanishing coefficients due to near-singular matrices.

---

### 48. What role does SVD play in NLP?

> Latent Semantic Analysis (LSA) uses SVD for dimensionality reduction.

---

### 49. When would you _not_ use linear regression?

> When data is nonlinear, has strong outliers, or target is categorical.

---

### 50. What metrics do you use to evaluate linear regression?

> MSE, RMSE, R², MAE — depending on business goals and sensitivity to outliers.

---

📌 **Pro Tip for Interviews:**  
Frame your answers with:

1. 💡 Conceptual clarity
2. 📐 Mathematical intuition
3. 🛠️ Edge case awareness
4. 🚀 Real-world application

---
