# 🧠 When and How to Convert Non-Convex Problems to Convex in ML

## 🧩 Why Convex Problems are Preferred

- Convex functions have **one global minimum** — no local traps.
- They allow **efficient optimization** using gradient methods.
- Theoretical guarantees: convergence, duality, regularization behavior.

---

## 🔁 Common Strategies to Convert Non-Convex → Convex

### 1. **Convex Relaxation**

Relax a hard constraint into a softer, convex version.

#### Example: Integer programming → Linear programming

Original:

$$
\min_{x \in \{0,1\}^n} \quad c^T x \quad \text{subject to} \quad Ax \leq b
$$

Relaxed:

$$
\min_{x \in [0,1]^n} \quad c^T x \quad \text{subject to} \quad Ax \leq b
$$

---

### 2. **Convex Surrogate Loss**

E.g., Replace 0–1 loss (non-convex) with convex hinge loss:

- Non-convex:

  $$
  L_{0-1}(y, f(x)) = \mathbb{I}[y \neq \text{sign}(f(x))]
  $$

- Convex surrogate (SVM):
  $$
  L_{\text{hinge}}(y, f(x)) = \max(0, 1 - y f(x))
  $$

Or Logistic loss:

$$
L_{\text{logistic}}(y, f(x)) = \log(1 + e^{-y f(x)})
$$

---

### 3. **Regularization to Convexify**

Sometimes adding an L2 term can make the objective strongly convex:

Non-convex:

$$
\min_w \; \sum_{i=1}^n (y_i - w^T x_i)^4
$$

Approximate convex form:

$$
\min_w \; \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \|w\|_2^2
$$

---

### 4. **Convex Envelope (Tightest Under-Approximation)**

If original function is $f(x)$, its **convex envelope** is:

$$
\text{conv}(f)(x) = \sup \{ g(x) \mid g \text{ is convex and } g(x) \leq f(x) \; \forall x \}
$$

Used in global optimization frameworks.

---

### 5. **Change of Variables**

Sometimes a substitution reveals a convex structure.

Example:
Let

$$
f(x) = \frac{(x-1)^2}{x}, \quad x > 0 \quad \text{(non-convex)}
$$

Substitute: $x = e^t$

Then:

$$
f(e^t) = \frac{(e^t - 1)^2}{e^t}
$$

which can be shown convex in $t$.

---

### 6. **Problem Reformulation via Duality**

Use duality to transform the problem into a convex dual.

Primal (possibly non-convex):

$$
\min_x \; f(x) \quad \text{subject to} \quad h(x) = 0
$$

Form Lagrangian:

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)
$$

Dual:

$$
g(\lambda) = \inf_x \mathcal{L}(x, \lambda)
$$

Then solve:

$$
\max_{\lambda} g(\lambda)
$$

---

## ⚠️ Limitations

- Not all non-convex problems are convexifiable.
- Convexification may oversimplify or lose important structure.
- Deep learning models remain inherently non-convex, but we optimize using local convex approximations.

---

## ✅ In Practice

Applied scientists often:

- Use convex surrogates for classification or regression.
- Apply SDP relaxations in graph, control, or NLP.
- Use KKT conditions and duality to interpret constrained problems.
- Build custom loss functions and ensure they are convex or convexifiable.

---

## 🧠 Key Takeaways

- **Convex = Global minimum, fast convergence, stability.**
- **If non-convex**, check if:
  - The objective can be approximated
  - The constraints can be relaxed
  - A convex surrogate can replace the original function

---
