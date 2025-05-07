

### 📍 **Algorithm 1: Simplified Trilateration**




**Input:** Sender positions \( s_j \), distances \( d_j \), weights \( w_{ij} \)  
**Output:** Receiver position \( x \)

1: Normalize weights: \( w_{ij} \leftarrow w_{ij} / \sum_{ij} w_{ij} \)  
2: Translate senders: \( s_j \leftarrow s_j - t \), where \( t = \sum_{ij} w_{ij} s_i \)  
3: Calculate \( A \) and \( g \) in (20).  
4: Find largest real eigenvalue \( \lambda_{\max} \) of \( M_A \) (31).  
5: Find receiver position as \( x = -(\lambda_{\max} I - A)^{-1} g \)  
6: Undo translation: \( x \leftarrow x + t \)



This algorithm estimates the **receiver position** $x$ given known **sender positions** $s_j$, **measured distances** $d_j$, and **weights** $w_{ij}$. The approach reformulates the localization problem as an **eigenvalue problem** for fast, stable, and non-iterative computation.

---

### 🔢 **Inputs**

* $s_j$: Position of sender $j$, for $j = 1, 2, ..., m$
* $d_j$: Distance from the receiver to sender $j$
* $w_{ij}$: Weight assigned to the sender pair $(i, j)$

---

### 🎯 **Output**

* $x$: Estimated receiver position in $\mathbb{R}^n$

---

### 🧠 **Algorithm Steps**

#### 1. **Normalize Weights**

Normalize the weight matrix to ensure the sum of all weights equals 1:

$$
w_{ij} \leftarrow \frac{w_{ij}}{\sum_{i,j} w_{ij}}
$$

#### 2. **Translate Senders**

Translate all sender coordinates to simplify calculations by removing the weighted centroid:

$$
t = \sum_{i,j} w_{ij} s_i
$$

$$
s_j \leftarrow s_j - t
$$

#### 3. **Calculate Matrices $A$ and Vector $g$**

Using the translated sender positions and weights, compute:

* Matrix $A$
* Vector $g$

These are defined based on the gradient of the cost function $\nabla h(x) = (x^T x) x - A x + g$

#### 4. **Compute Maximum Eigenvalue**

Find the **largest real eigenvalue** $\lambda_{\max}$ of the matrix:

$$
M_A = (\text{symmetrized form involving } A, g, \text{ and weight terms})
$$

#### 5. **Solve for Receiver Position**

Using the eigenvalue, compute:

$$
x = -(\lambda_{\max} I - A)^{-1} g
$$

#### 6. **Undo Translation**

Reverse the earlier translation to get the final result in the original coordinate space:

$$
x \leftarrow x + t
$$

---

### 💡 **Key Insight**

This algorithm avoids iterative optimization by reducing the localization problem to an eigenvalue computation, making it:

* **Numerically stable**
* **Non-iterative**
* **Efficient for large-scale or real-time systems**


