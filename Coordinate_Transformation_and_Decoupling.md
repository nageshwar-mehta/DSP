
---

## 🔄 Coordinate Transformation & Decoupling

This section explains **why and how** we diagonalize the matrix $A$ and apply a **change of variables** to simplify the optimization problem. These steps are crucial to convert the gradient condition into a solvable **eigenvalue problem**.

---

### 📌 Why Diagonalize Matrix $A$?

Matrix $A$ appears in the gradient of the cost function:

```math
∇h(x) = (xᵗx)·x - A·x + g
```

* Since $A$ is **real and symmetric**, it can be **diagonalized**:

  ```math
  A = Q·D·Qᵗ
  ```

  where:

  * $Q$ is an orthogonal matrix (i.e., $QᵗQ = I$),
  * $D$ is a diagonal matrix (i.e., all off-diagonal elements are zero).

This diagonalization simplifies the problem by removing cross-variable interactions in the quadratic term $A·x$.

---

### 🔁 Change of Variables: Rotating the Coordinate System

To exploit the diagonal form of $A$, apply a **change of variables**:

```math
y = Qᵗx
```

Now, in this new coordinate system:

* The cost function becomes $f(y) = h(Qy)$,
* The gradient becomes:

  ```math
  ∇f(y) = (‖y‖²)·y - D·y + b
  ```

  where $b = Qᵗg$.

This transformation **preserves the structure of the problem** but moves it into a **rotated basis** where $A$ is diagonal.

---

### ✂️ What Does "Decoupled" Mean?

A system is **decoupled** when each variable can be analyzed **independently**.

#### ✅ Before Transformation (Coupled System)

* Gradient terms like $A·x$ involve cross-terms:

  ```math
  [a₁₁ a₁₂] [x₁]
  [a₂₁ a₂₂] [x₂]
  → a₁₂·x₂ and a₂₁·x₁ cause variables to interfere
  ```

#### ✅ After Transformation (Decoupled System)

* Diagonal matrix $D$ means:

  ```math
  ∇f(y) = [(‖y‖² - Dₖₖ)·yₖ + bₖ] for each k
  ```

  Now, each coordinate $y_k$ appears only in its own equation. **No cross-terms remain**.

This decoupling:

* Reduces a multivariate nonlinear system to a set of **independent scalar equations**,
* Enables conversion into a **single eigenvalue problem**.

---

### ✅ Summary

| Step                        | Purpose                     | Effect                                         |
| --------------------------- | --------------------------- | ---------------------------------------------- |
| Diagonalize $A$             | Simplify structure          | Get diagonal matrix $D$                        |
| Change variables: $y = Qᵗx$ | Rotate coordinate system    | Convert to frame where variables are separable |
| Decoupling                  | Separate variable influence | Solve each coordinate independently            |

---

This approach is what makes the eigenvalue-based localization method **fast, robust, and algebraically elegant**.
