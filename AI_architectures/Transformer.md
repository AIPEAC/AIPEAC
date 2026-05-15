## Self-Attention, K,Q,V, and Transformer Decoder
- Apr 5th, 2026
---

# Transformer.md

## 1. Multi-Head Attention Mechanism

### Single-Head Attention
The result reflects the relativity within the input sequence.

$$R = \alpha V = \sigma \left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

**Linear Projections:**
* $Q = X W^Q$
* $K = X W^K$
* $V = X W^V$

**Dimensions:**
* $X \in \mathbb{R}^{n \times d_{model}}$ (Input tensor: word number $\times$ embedding dim)
* $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$
* $W^V \in \mathbb{R}^{d_{model} \times d_v}$

### Multi-Head Attention
Instead of performing a single attention function, we project the queries, keys, and values $h$ times with different, learned linear projections.

$$\text{Head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$
$$\text{MultiHead}(X) = \text{Concat}(\text{Head}_1, \dots, \text{Head}_h)W^O$$

**Typical Parameter Settings:**
* $d_v = d_k = \frac{d_{model}}{h}$
* $W^O \in \mathbb{R}^{hd_k \times d_{model}}$

---

## 2. Symmetry and Role Specialization (Q & K)

### The Symmetry Property
Mathematically, $Q$ and $K$ are symmetric in the dot-product $QK^T$. If you swap $W^Q$ and $W^K$ and transpose the attention matrix, the final result remains identical.
* **Primary Purpose:** The split into $Q$ and $K$ is mainly to **reduce dimensionality** and **computational complexity** while allowing the model to learn complex relationships.

### Breaking Symmetry via Backpropagation (Nabla Intuition)
Despite the mathematical symmetry, the training process forces them into distinct roles ("The Searcher" vs "The Index") via gradient signals:

$$\nabla_{W^Q} L = X^T (\dots) K$$
$$\nabla_{W^K} L = X^T (\dots)^T Q$$

* **$W^Q$ Gradient:** Driven by how well it finds the relevant $K$.
* **$W^K$ Gradient:** Driven by how well it responds to the relevant $Q$.
* **The Anchor ($V$):** Since the output is weighted on $V$, $Q$ and $K$ must evolve as a "Lock and Key" pair to extract the most useful values for the task.

---

## 3. Transformer Decoder Architecture
The Decoder is built for **Autoregressive Generation**, introducing two key modifications.

### A. Masked Multi-Head Attention
* **Mechanism:** A mask is applied to the $QK^T$ scores (setting future positions to $-\infty$) before Softmax.
* **Effect:** Position $i$ can only attend to positions $1, \dots, i$.

### B. Encoder-Decoder Attention (Cross-Attention)
* **Query ($Q$):** Comes from the previous layer of the **Decoder**.
* **Key ($K$) & Value ($V$):** Come from the final output of the **Encoder**.

### C. Feed-Forward Network (FFN)
Applied after attention to process the integrated information.
$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$
It acts as the "knowledge processor," refining the representation of each token independently.
