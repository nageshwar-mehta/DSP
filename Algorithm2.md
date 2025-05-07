
# 📡 Algorithm 2: Trilateration

This algorithm estimates the position of a receiver using trilateration by solving an eigenvalue-based formulation. Unlike Algorithm 1, it is designed to **handle degenerate cases** where multiple or no solutions may exist.

---

## 📥 Input

* $s_j$: Sender positions (known anchor locations)
* $d_j$: Measured distances from the receiver to each sender
* $w_{ij}$: Weights for sender pairs (used in the cost function)

---

## 📤 Output

* Estimated receiver position(s) $x \in \mathbb{R}^n$, or no output if the configuration is degenerate

---

## 🧠 Steps

### Step 1: Normalize Weights

Normalize the weight matrix so the total weight sums to 1:

$$
w_{ij} \leftarrow \frac{w_{ij}}{\sum_{i,j} w_{ij}}
$$

---

### Step 2: Translate Sender Positions

Remove the weighted centroid of sender positions:

$$
t = \sum_{i,j} w_{ij} s_i \quad\Rightarrow\quad s_j \leftarrow s_j - t
$$

---

### Step 3: Compute $\mathbf{D}$ and $\mathbf{b}$

Calculate the reduced matrix $\mathbf{D}$ and gradient vector $\mathbf{b}$ from equation (21) using the translated senders and weights.

---

### Step 4: Solve Eigenvalue Problem

Compute the **largest real eigenvalue** $\lambda_{\text{max}}$ of matrix $\mathbf{M}$, defined in equation (24). This matrix is constructed from $\mathbf{D}$, $\mathbf{b}$, and identity/zero blocks.

---

### Step 5–10: Determine Solution Type

#### Case 1: Full Rank

If

$$
\text{rank}(\lambda_{\text{max}} I - D) = n
$$

Then a **unique solution** $y$ exists.

Use equations (33)–(34) to solve for $y$, and choose the sign in equation (34) such that:

$$
\text{sign}(y_1) = -\text{sign}(b_1)
$$

---

#### Case 2: Rank Deficient

If

$$
\text{rank}(\lambda_{\text{max}} I - D) = n - 1
$$

Then **two distinct solutions** $y_1$ and $y_2$ are valid. Solve for both using the same equations.

---

#### Case 3: Ill-Defined

If

$$
\text{rank}(\lambda_{\text{max}} I - D) < n - 1
$$

Then the problem is **ill-defined** (e.g., due to degenerate sensor geometry), and **no solution** is returned.

---

### Step 11: Undo Rotation

Transform the reduced-coordinate solution $y$ back to full space using:

$$
x = Q y
$$

where $Q$ is the orthonormal basis used in the dimensionality reduction step.

---

### Step 12: Undo Translation

Restore the original coordinate frame:

$$
x \leftarrow x + t
$$

---

## ✅ Key Features

* Handles both **unique** and **ambiguous** solutions
* Checks matrix rank to avoid unstable or undefined behavior
* Solves localization using **non-iterative, eigenvalue-based methods**

---

## 📎 References

* Refer to equations (20)–(34) in the paper *"Single-Source Localization as an Eigenvalue Problem"* for mathematical derivations.


