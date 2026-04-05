## Self-Attention and Transformer Decoder

### 1. The Q-K Duality: Symmetry and Efficiency
While $Q$ and $K$ are conceptually different (Searcher vs. Index), they are mathematically symmetric. If you were to swap the weight matrices $W^Q$ and $W^K$ and transpose the resulting attention matrix, the network's expressive power remains identical.

**Why distinguish them?**
* **Computational Efficiency:** By projecting $X$ into lower-dimensional $d_k$ spaces, we significantly reduce the cost of the dot-product $QK^T$ compared to operating on the full input dimension.
* **Role Specialization:** Although they *could* swap, the training process forces them into a fixed "Lock and Key" relationship to minimize the objective function.

---

### 2. The Gradient Intuition (Nabla Notation)
The backpropagation process "breaks" the initial symmetry. The weight updates are driven by the following gradients:

$$\nabla_{W^Q} L = X^T (\dots) K$$
$$\nabla_{W^K} L = X^T (\dots)^T Q$$

**Key Insight:**
* **$W^Q$** learns by looking at $K$. It adjusts to find the right "tags."
* **$W^K$** learns by looking at $Q$. It adjusts to be "found" by the right queries.
* The **Value ($V$)** acts as the anchor; $Q$ and $K$ must coordinate their symmetry to successfully extract the correct information from $V$.

---

### 3. Decoder Architecture
The Decoder is designed for **Generation**. Unlike the Encoder, it is "Autoregressive," meaning it uses previously generated tokens to predict the next one.

#### A. Masked Multi-Head Attention
* **The Constraint:** In the decoder, a token should not be able to "see" future tokens.
* **The Mechanism:** We apply a **Mask** (setting future values to $-\infty$ before Softmax) to the $QK^T$ matrix. This ensures $z_i$ only depends on $x_{1 \dots i}$.


#### B. Encoder-Decoder Attention (Cross-Attention)
* **The Bridge:** This layer allows the decoder to focus on the input sequence.
* **The Source:** * **$Q$** comes from the Decoder's previous layer (What am I looking for in the source text?).
    * **$K$ and $V$** come from the Encoder's final output (Here is the source information).


#### C. Feed-Forward Network (FFN)
* **The Function:** A position-wise transformation applied to each token independently.
$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$
* It processes the information gathered by the attention layers, acting as the model's "knowledge storage."
