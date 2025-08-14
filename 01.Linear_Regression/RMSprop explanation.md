# 📘 RMSProp Explanation

---

### 1. Why use the moving average of the squared gradients?

RMSProp keeps track of:

$$
s_t = \beta s_{t-1} + (1-\beta) g_t^2
$$

Where:

- $g_t = \nabla_\theta J(\theta)$ is the current gradient
- $s_t$ is a **running average of the squared gradients**
- $\beta$ (e.g., 0.9) is a decay factor

**Reasoning:**

- Some parameters have **steep gradients**, some have **flat gradients**.
- If we use a fixed learning rate, parameters with steep gradients can **overshoot the minimum**, and parameters with small gradients can **update too slowly**.
- By keeping a **moving average of squared gradients**, RMSProp adapts the step size for each parameter individually:

  - Large gradient → divide by large $\sqrt{s_t}$ → smaller step
  - Small gradient → divide by small $\sqrt{s_t}$ → larger step

✅ Intuition: It **normalizes the step size** so updates are more balanced across all parameters.

---

### 2. Why the update formula looks like this?

RMSProp update rule:

$$
\theta = \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
$$

Breaking it down:

1. **Numerator $g_t$** → the direction of the gradient
2. **Denominator $\sqrt{s_t + \epsilon}$** → scales the step by recent gradient magnitudes

   - Prevents overshooting in steep directions
   - Boosts step in flat directions
   - $\epsilon$ is a small number to avoid division by zero (e.g., $10^{-8}$)

3. **Learning rate $\eta$** → global step size scaling

So the **effective step for each parameter** becomes:

$$
\text{effective step} = \frac{\eta g_t}{\sqrt{s_t + \epsilon}}
$$

- Steep gradient → denominator large → small step → stable
- Flat gradient → denominator small → big step → faster learning

---

### 3. Intuition

- Imagine walking downhill on a **bumpy terrain**:

  - Big slopes → walk slowly to avoid falling
  - Flat slopes → walk faster to save time

RMSProp does exactly this automatically for each parameter.
