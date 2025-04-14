# Adam-Clara Optimizer Variants

This repository provides **three variants** of the Adam optimizer extended with **CLARA** (Cumulative Learning Rate Adaptation).

CLARA modifies the learning rate dynamically during training based on the cumulative direction ("path") of parameter updates, enabling **adaptive learning rate control** without manual scheduling.

---

## Optimizer Variants

| Optimizer         | Core Idea                                     | Update Strategy | Stability | Efficiency | Potential Pitfalls                          |
| ----------------- | --------------------------------------------- | --------------- | --------- | ---------- | ------------------------------------------- |
| AdamClaraGlobal   | Global path aggregation across all parameters | Two loops       | Very high | Lower      | Computational overhead (two passes)         |
| AdamClaraLocal    | Immediate local adjustment per parameter      | Single loop     | Medium    | Very high  | Potential LR oscillations without smoothing |
| AdamClaraSmoothed | Local adjustment with clipping and smoothing  | Single loop     | High      | High       | May react slower to large true changes      |

---

## Detailed Explanation

### 1. AdamClaraGlobal

**Concept:**

- Aggregates the cumulative path norms **across all parameters** in a first pass.
- Calculates a **global learning rate scaling factor**.
- Applies a clipped and safe update to all parameters in a second pass.

**Advantages:**

- **Very stable** for all model types and scales.
- Handles outlier parameters gracefully through global averaging.
- **Clipping** ensures no explosion/collapse of learning rates.

**Pitfalls:**

- **Computational cost**: Steps are computed **twice** per optimization step.
- Global behavior may hide per-layer or per-parameter dynamics.

**Best for:**

- Large and deep models (e.g., Transformers, CNNs).
- Tasks that prioritize **stability** (e.g., RL, meta-learning).

---

### 2. AdamClaraLocal

**Concept:**

- Updates each parameter and its path **immediately** after computing the local gradient.
- Adjusts the learning rate **per parameter** based on local path history.
- Scaling factor is clipped to avoid instability.

**Advantages:**

- **Fast**: Only one loop over parameters.
- High **flexibility** in responding to local gradient behavior.

**Pitfalls:**

- Without smoothing, **learning rate oscillations** are possible.
- Potentially unstable for high-dimensional or noisy problems.

**Best for:**

- Small to medium models.
- Fast prototyping where **speed** is more important than ultra-stability.

---

### 3. AdamClaraSmoothed

**Concept:**

- Combines immediate local updates with **clipped scaling** and **smoothed learning rate adjustment** (moving average).
- More aggressive exponentiation for better sensitivity.
- Provides an excellent **trade-off between speed and stability**.

**Advantages:**

- **Robust** to sudden noise or sharp gradient changes.
- **Smooth** learning rate evolution.
- **Clipping** prevents learning rate divergence.

**Pitfalls:**

- Might **respond more slowly** to sharp, true gradient changes.
- Slightly higher internal complexity.

**Best for:**

- General deep learning tasks.
- Pipelines where both **adaptivity** and **robustness** are needed.

---

## Visual Comparison

```
           STABILITY  |  SPEED
AdamClaraGlobal   +++       -
AdamClaraLocal     +       +++
AdamClaraSmoothed  ++       ++
```

---

## Recommendation

| Scenario                                    | Suggested Optimizer |
| ------------------------------------------- | ------------------- |
| Very large models (e.g., Transformers, RL)  | AdamClaraGlobal     |
| Fast experimental prototyping, small models | AdamClaraLocal      |
| Balanced training (stability + speed)       | AdamClaraSmoothed   |

---

## Notable Enhancements in Current Version

- **Clipped Scaling Factors**: Avoids catastrophic learning rate collapse or explosion.
- **Boosted Exponent in Smoothed Version**: Faster adaptation to non-stationary environments.
- **Path Smoothing**: Ensures cumulative steps reflect stable long-term progress.
- **Single and Double Pass Options**: Trade off between computational cost and stability.

---
