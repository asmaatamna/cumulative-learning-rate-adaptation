# Adam-Clara Optimizer Variants

This repository provides **three variants** of the Adam optimizer extended with **CLARA** (Cumulative Learning Rate Adaptation).

CLARA modifies the learning rate dynamically during training based on the cumulative direction ("path") of parameter updates.

---

## Optimizer Variants

| Optimizer             | Core Idea                                   | Update Strategy  | Stability       | Efficiency       | Potential Pitfalls                          |
|-----------------------|--------------------------------------------|------------------|-----------------|------------------|---------------------------------------------|
| AdamClaraGlobal       | Global path aggregation over all params    | Two loops        | High            | Lower            | Double gradient computation per step        |
| AdamClaraLocal        | Immediate local adjustment per parameter   | Single loop      | Lower           | High             | Learning rate oscillations                  |
| AdamClaraSmoothed     | Local adjustment with clipping & smoothing | Single loop      | High (improved) | High             | Might react slower to true large changes    |

---

## Detailed Explanation

### 1. AdamClaraGlobal

**Concept:**  
- Accumulates the path norms of **all parameters** during the first loop.
- Calculates a **single global learning rate scaling factor** based on the average path norm.
- Applies this global scaling factor to all parameters in a **second loop**.

**Advantages:**  
- Very **stable** across all model parameters.
- Smooth learning rate adaptation even with large or imbalanced models.

**Pitfalls:**  
- **Two loops** → Gradients and steps are computed **twice**, which increases computational cost.
- **Global aggregation** can "wash out" the behavior of individual parameters: large changes in one layer can be hidden if other layers are stable.

**Best for:**  
- Large models with diverse parameter scales.
- Tasks where stability is critical (e.g., reinforcement learning, meta-learning).

---

### 2. AdamClaraLocal

**Concept:**  
- **Immediately** updates each parameter and its path.
- **Directly** computes the learning rate adjustment **per parameter** without waiting for all parameters.
- No second pass.

**Advantages:**  
- **Fast** (single loop).
- Behavior is close to classic Adam + immediate CLARA adaptation.

**Pitfalls:**  
- **Learning rate can fluctuate heavily** between parameters and between steps.
- Instability possible, especially for high-dimensional models.
- No smoothing → sensitive to noisy gradients.

**Best for:**  
- Small to medium models.
- Experimental setups where fast iterations are more important than ultra-stable training.

---

### 3. AdamClaraSmoothed

**Concept:**  
- Like AdamClaraLocal: immediate updates after each parameter.
- **Clips** the learning rate scaling factor to a safe range (e.g., 0.5× to 2×).
- **Smooths** the learning rate change over time (moving average).

**Advantages:**  
- **Combines speed and stability**:  
  - Fast single loop.
  - Robust against sudden noisy updates.
  - Smooth learning rate change reduces shock.

**Pitfalls:**  
- May **react slowly** to very sharp, real changes in gradient behavior.
- Slightly higher complexity compared to pure Local.

**Best for:**  
- Most general use cases where you want speed **and** reasonable stability.
- Practical deep learning training pipelines.

---

## Visual Overview

```
           STABILITY  |  SPEED
AdamClaraGlobal   +++       -
AdamClaraLocal     +       +++
AdamClaraSmoothed  ++       ++
```

---

## Recommendation

| Scenario                                    | Suggested Optimizer   |
|---------------------------------------------|------------------------|
| Very large models (e.g., Transformers)      | AdamClaraGlobal        |
| Fast prototyping / small models             | AdamClaraLocal         |
| Balanced training (speed + robustness)      | AdamClaraSmoothed      |

---