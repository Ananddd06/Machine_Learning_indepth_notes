# Support Vector Machine (SVM) - Mathematical Formulation

## Hypothesis Function

The decision function (or hypothesis) is defined as:

$$
h_{w,b}(x) = w^T x + b
$$

Where:\
- $w$ → weight vector (normal to the hyperplane)\
- $x$ → input feature vector\
- $b$ → bias term

------------------------------------------------------------------------

## Classification Rule

The classifier assigns a label based on the sign of $h_{w,b}(x)$:

$$
f(x) = \text{sign}(h_{w,b}(x)) = \text{sign}(w^T x + b)
$$

-   If $f(x) = +1$ → belongs to **positive class**\
-   If $f(x) = -1$ → belongs to **negative class**

------------------------------------------------------------------------

## Relationship with Class Labels

For each training data point $(x_i, y_i)$ where $y_i \in \{-1, +1\}$:

$$
y_i (w^T x_i + b) \geq 1
$$

-   If $y_i = +1$: then $w^T x_i + b \geq +1$ → mostly positive region.\
-   If $y_i = -1$: then $w^T x_i + b \leq -1$ → mostly negative region.

This ensures all data points are correctly classified and lie outside
the margin.

------------------------------------------------------------------------

## Hyperplane Formula

The separating hyperplane is defined as:

$$
w^T x + b = 0
$$

This is the decision boundary that separates the two classes.

------------------------------------------------------------------------

## Geometric Margin

The margin is the distance between the hyperplane and the closest data
point.\
For a data point $(x_i, y_i)$, the **geometric margin** is:

$$
\gamma_i = \frac{y_i (w^T x_i + b)}{\|w\|}
$$

-   $y_i (w^T x_i + b)$ measures how confidently the point is
    classified.\
-   Dividing by $\|w\|$ normalizes by the length of the weight vector.

The SVM optimization problem **maximizes the minimum margin** across all
training points.

------------------------------------------------------------------------

## Optimization Problem (Primal Form)

The standard hard-margin SVM optimization problem is:

$$
\min_{w, b} \, \frac{1}{2} \|w\|^2
$$

subject to

$$
y_i (w^T x_i + b) \geq 1 \quad \forall i
$$

------------------------------------------------------------------------

## Summary

-   **Hypothesis:** $h_{w,b}(x) = w^T x + b$\
-   **Classification rule:** $f(x) = sign(w^T x + b)$\
-   **Constraint:** $y_i (w^T x_i + b) \geq 1$\
-   **Hyperplane:** $w^T x + b = 0$\
-   **Geometric Margin:** $\gamma_i = \frac{y_i (w^T x_i + b)}{\|w\|}$\
-   **Optimization Goal:** Maximize margin by minimizing
    $\frac{1}{2}\|w\|^2$
