

# QR Decomposition

## 📘 What is QR Decomposition?

QR decomposition is a **matrix factorization technique** in linear algebra. It decomposes a given matrix **A** into the product of two matrices:

$$
A = Q R
$$

Where:

* **Q** is an **orthogonal matrix** (i.e., $Q^T Q = I$)
* **R** is an **upper triangular matrix**

This technique is widely used in solving linear systems, least squares problems, and eigenvalue algorithms.

---

## 🔍 Why is QR Decomposition Useful?

* **Solving linear systems** $Ax = b$
* **Least squares approximation** for overdetermined systems
* **Eigenvalue computations**
* **Orthogonalization** of vectors (e.g., in Gram-Schmidt process)

---

## 🧪 Example in Python using NumPy

```python
import numpy as np

# Define a matrix A
A = np.array([[12, -51, 4],
              [6, 167, -68],
              [-4, 24, -41]])

# Perform QR decomposition
Q, R = np.linalg.qr(A)

# Display the results
print("Q matrix:")
print(Q)
print("\nR matrix:")
print(R)

# Verify that A = Q @ R
print("\nReconstructed A (Q @ R):")
print(np.dot(Q, R))
```

### ✅ Output Interpretation:

* `Q`: Orthogonal matrix (columns are orthonormal vectors)
* `R`: Upper triangular matrix
* The product `Q @ R` should reconstruct the original matrix `A` (up to small floating-point errors)

---

## 🧠 Geometric Intuition

QR decomposition **orthogonalizes** the column vectors of matrix **A**. You can think of it as decomposing the action of **A** into:

1. A **rotation/reflection** (`Q`)
2. A **scaling/shearing** (`R`)

---

## 📎 Notes

* `np.linalg.qr` performs QR decomposition in NumPy
* For rectangular $A$, you can get **economy size** or **full size** decompositions depending on context
* `Q` will have orthonormal columns; `R` is square upper triangular

---

## 🛠 Advanced

For a vector of all 1s of size $n$, the following creates an orthogonal projection matrix:

```python
one_vec = np.ones(n)
Q_full, _ = np.linalg.qr(np.eye(n) - np.outer(one_vec, one_vec) / n)
```

This is used for **projecting into the subspace orthogonal to the vector of all 1s**, often helpful in signal processing and localization problems.

---

## 📚 References

* [NumPy Documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html)
* [Wikipedia – QR Decomposition](https://en.wikipedia.org/wiki/QR_decomposition)


