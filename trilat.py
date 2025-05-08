import numpy as np

"""
Pure-Python port of Julia trilat and trilat_A.
"""

def trilat(s, d2, W=None):
    """
    Pure-Python trilateration matching Julia's trilat implementation.
    Returns solution(s) x as shape (d,1).
    """
    s = np.asarray(s)
    d, n = s.shape
    d2 = np.asarray(d2).ravel()
    # Default weights
    if W is None:
        W = np.diag(1.0 / d2)
    W = np.asarray(W, dtype=float)
    # Normalize weights
    W = W / np.sum(W)
    sumW = np.sum(W, axis=1)
    t = s.dot(sumW)
    st = s - t[:, None]
    ws2md2 = W.dot(np.sum(st**2, axis=0) - d2)
    A = -2 * st.dot(W).dot(st.T) - np.sum(ws2md2) * np.eye(d)
    g = -st.dot(ws2md2)
    # Build block matrix Ma
    Ma = np.block([
        [A, np.eye(d), np.zeros((d,1))],
        [np.zeros((d,d)), A, -g[:,None]],
        [-g[None,:], np.zeros((1,d+1))]
    ])
    lambdas = np.linalg.eigvals(Ma).real
    lam = np.max(lambdas)
    x = np.linalg.solve(lam * np.eye(d) - A, -g)
    # Return as column vector
    return (x + t)[:, None]

# Alias for the same method
trilat_A = trilat
