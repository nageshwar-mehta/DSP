# Eigenvalue-Based Optimization for Source Localization

This repository contains the formulation and computation of the **matrix M** used in the eigenvalue decomposition for source localization problems. The formulation follows from lifting quartic cost functions in problems such as TDOA/FDOA to a higher-dimensional matrix whose eigenvectors yield the solution.

## 🔍 Mathematical Background

We consider minimizing a quartic cost function of the form:

\[
h(x) = \sum_{i,j} w_{ij} \left( \|x - s_i\|^2 - d_i^2 \right) \|x - s_j\|^2
\]

To find the stationary point, we solve:

\[
\nabla h(x) = 0
\]

After simplifications (homogenization and translation), this becomes:

\[
\nabla h(x) = (x^T x) x - A x + g
\]

Letting:

- \( Q \in \mathbb{R}^{n \times (n-1)} \): Orthonormal basis for the subspace orthogonal to the vector of ones
- \( D = Q^T A Q \)
- \( b = Q^T g \)

We define the lifted vector \( y = Q^T x \), and then construct the **matrix M** for eigenvalue decomposition.

## 🧮 Matrix M Formulation

The matrix \( \mathbf{M} \in \mathbb{R}^{(2d+1) \times (2d+1)} \) is given by:

\[
\mathbf{M} =
\begin{pmatrix}
\mathbf{D} & -\text{diag}(\mathbf{b}) & 0 \\
\mathbf{O} & \mathbf{D} & -\mathbf{b} \\
\mathbf{1}^T & \mathbf{0}^T & 0
\end{pmatrix}
\]

Where:

- \( D \): Reduced matrix from quadratic form
- \( b \): Reduced linear term
- \( d = n - 1 \)
- \( \text{diag}(b) \): Diagonal matrix with elements of vector \( b \)

We solve:

\[
\mathbf{M} \cdot
\begin{pmatrix}
y^2 \\
y \\
1
\end{pmatrix}
= \lambda
\begin{pmatrix}
y^2 \\
y \\
1
\end{pmatrix}
\]

## 🧑‍💻 Python Implementation

```python
import numpy as np

# Number of sensors
n = 3

# Symmetric matrix A (e.g., from a quadratic cost)
A = np.array([
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2]
])

# Vector g
g = np.array([1, -2, 0])

# Create orthonormal basis Q orthogonal to 1^T y = constant
one_vec = np.ones(n)
Q_full, _ = np.linalg.qr(np.eye(n) - np.outer(one_vec, one_vec) / n)
Q = Q_full[:, 1:]  # Drop first column (aligned with ones)

# Compute D and b
D = Q.T @ A @ Q
b = Q.T @ g
d = D.shape[0]

# Construct M
upper = np.hstack([D, -np.diag(b), np.zeros((d, 1))])
middle = np.hstack([np.zeros((d, d)), D, -b.reshape(-1, 1)])
lower = np.hstack([np.ones((1, d)), np.zeros((1, d)), np.zeros((1, 1))])
M = np.vstack([upper, middle, lower])

# Eigenvalue decomposition
eigvals, eigvecs = np.linalg.eig(M)
idx = np.argmax(eigvals.real)
v = eigvecs[:, idx].real

# Extract components
y_squared = v[:d]
y = v[d:-1]
constant = v[-1]

# Normalize homogeneous vector
y = y / constant
