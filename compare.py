import numpy as np
from trilat import trilat, trilat_A

# Generate random test data
np.random.seed(123)
dim, m = 3, 10
s = np.random.randn(dim, m)
x_true = np.random.randn(dim)
# Compute squared distances
d2 = np.sum((s - x_true[:, None])**2, axis=0)

# Pure-Python solutions
x1 = trilat(s, d2).ravel()
x2 = trilat_A(s, d2).ravel()

# Print results
print("True x:      ", x_true)
print("trilat x:    ", x1)
print("trilat_A x:  ", x2)
print(f"||trilat - true||    = {np.linalg.norm(x1 - x_true):.6f}")
print(f"||trilat_A - true||  = {np.linalg.norm(x2 - x_true):.6f}")
print(f"||trilat - trilat_A||= {np.linalg.norm(x1 - x2):.6f}")
