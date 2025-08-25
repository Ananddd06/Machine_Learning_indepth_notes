
# Naive Bayes with Bernoulli Features (Spam Filtering Example)

When features are **binary** (e.g., presence or absence of words in spam filtering), Naive Bayes is often parameterized as follows:

---

## Parameterization

We define the parameters:

- $\phi_y = P(y = 1)$ = prior probability of class label $y = 1$ (e.g., spam).
- $\phi_{j|y=1} = P(x_j = 1 | y = 1)$ = probability that feature $j$ (word $j$) appears in an email given it is spam.
- $\phi_{j|y=0} = P(x_j = 1 | y = 0)$ = probability that feature $j$ appears in an email given it is not spam.

---

## Likelihood of the Training Data

Given a training set

$$
\mathcal{D} = \{ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(n)}, y^{(n)}) \},
$$

the likelihood of the parameters is:

$$
L(\phi_y, \phi_{j|y=0}, \phi_{j|y=1}) = \prod_{i=1}^n P(x^{(i)}, y^{(i)})
$$

where

$$
P(x^{(i)}, y^{(i)}) = P(y^{(i)}) \, \prod_{j=1}^d P(x_j^{(i)} | y^{(i)}).
$$

---

## Maximum Likelihood Estimates (MLE)

By maximizing the likelihood, we obtain closed-form estimates:

- Class prior:

$$
\phi_y = \frac{\sum_{i=1}^n 1\{ y^{(i)} = 1 \}}{n}
$$

- Conditional probability of feature $j$ given spam:

$$
\phi_{j|y=1} = \frac{\sum_{i=1}^n 1\{ x_j^{(i)} = 1 \wedge y^{(i)} = 1 \}}{\sum_{i=1}^n 1\{ y^{(i)} = 1 \}}
$$

- Conditional probability of feature $j$ given not spam:

$$
\phi_{j|y=0} = \frac{\sum_{i=1}^n 1\{ x_j^{(i)} = 1 \wedge y^{(i)} = 0 \}}{\sum_{i=1}^n 1\{ y^{(i)} = 0 \}}
$$

Here, $1\{ \cdot \}$ is the **indicator function**, and the symbol $\wedge$ means logical **AND**.

---

## Prediction

For a new email $x = (x_1, x_2, \dots, x_d)$, the posterior probability that it is spam is computed as:

$$
P(y = 1 | x) = \frac{ P(x | y = 1) P(y = 1) }{ P(x | y = 1) P(y = 1) + P(x | y = 0) P(y = 0) }
$$

where

$$
P(x | y = 1) = \prod_{j=1}^d P(x_j | y = 1),
$$

and

$$
P(x | y = 0) = \prod_{j=1}^d P(x_j | y = 0).
$$

---

## Interpretation

- $\phi_{j|y=1}$ is the fraction of **spam emails** containing word $j$.
- $\phi_{j|y=0}$ is the fraction of **non-spam emails** containing word $j$.
- $\phi_y$ is simply the fraction of emails that are spam.

The classifier uses these probabilities to decide whether a new email is more likely to be spam or not.

---

## Decision Rule

The decision rule can be written as:

$$
\hat{y} = \underset{y \in \{0,1\}}{\arg\max} \, P(y) \, \prod_{j=1}^d P(x_j | y)
$$

Equivalently, in log form (to avoid underflow):

$$
\hat{y} = \underset{y \in \{0,1\}}{\arg\max} \, \log P(y) + \sum_{j=1}^d \log P(x_j | y).
$$

---

## Summary

- Parameters ($\phi$) are estimated by simple frequency counts.  
- Predictions are based on comparing the posterior probabilities $P(y=1|x)$ vs $P(y=0|x)$.  
- Works well for **text classification tasks** like spam filtering, despite the independence assumption being unrealistic.
