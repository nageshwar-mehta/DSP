

### **Algorithm 2: Trilateration**

**Input:** Sender positions $s_j$, distances $d_j$, weights $w_{ij}$
**Output:** Receiver position(s) $\mathbf{x}$

---

1. **Normalize weights:**
   Ensure the weights form a valid convex combination:

   $$
   w_{ij} \leftarrow \frac{w_{ij}}{\sum_{i,j} w_{ij}}
   $$

2. **Translate sender positions:**
   Center the sender coordinates by subtracting the weighted average:

   $$
   \mathbf{t} = \sum_{i,j} w_{ij} \mathbf{s}_i,\quad \mathbf{s}_j \leftarrow \mathbf{s}_j - \mathbf{t}
   $$

3. **Calculate matrices $\mathbf{D}$ and vector $\mathbf{b}$:**
   These represent the reduced quadratic form and gradient in the transformed coordinate system (from equation 21).

4. **Find the largest real eigenvalue $\lambda_{\max}$ of matrix $\mathbf{M}$:**
   Matrix $\mathbf{M}$ (from equation 24) encodes the lifted quartic optimization problem as an eigenvalue problem.

---

### **Case Handling Based on Matrix Rank:**

5. **If** $\text{rank}(\lambda_{\max} \mathbf{I} - \mathbf{D}) = n$:
   There is a **unique solution** for $y$.

6. **Solve for $y$** using equations (33)–(34), and choose the **sign in equation (34)** to ensure:

   $$
   \text{sign}(y_1) = -\text{sign}(b_1)
   $$

   (this resolves ambiguity in square root solutions)

---

7. **Else if** $\text{rank}(\lambda_{\max} \mathbf{I} - \mathbf{D}) = n - 1$:
   The solution is **ambiguous** — there are **two valid solutions**.

8. **Solve for both solutions** $y_1$ and $y_2$ using the same system (equations 33–34).

---

9. **Else:**
   The matrix is too degenerate, and the problem is **ill-defined** (e.g., all sensors lie on a line or plane in 3D).

10. **Return nothing** (i.e., no valid localization possible).

---

11. **Undo the rotation:**
    Map back from reduced coordinate system to full space using:

$$
\mathbf{x} = Q y
$$

12. **Undo the translation:**
    Add back the original translation offset:

$$
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{t}
$$

---

This algorithm is more general than Algorithm 1 and carefully handles **degenerate configurations** (e.g., co-linear or co-planar sensor layouts) by inspecting the rank of the matrix involved in the eigenvalue problem.

