import numpy as np
from qcqp import qcqp


def trilat_linear(s, d2, W=None):
    """
    Linear least-squares trilateration.
    """
    s = np.asarray(s)
    d2 = np.asarray(d2)
    d, n = s.shape
    if W is None:
        W = np.diag(1.0 / d2)
    W = W / np.sum(W)
    A = np.hstack([-2 * s.T, np.ones((n, 1))])
    b = d2 - np.sum(s**2, axis=0)
    Aw = W.dot(A)
    y = np.linalg.solve(A.T.dot(Aw), A.T.dot(W).dot(b))
    return y[:-1]


def trilat_zhou(s, d2):
    """Zhou's closed-form algorithm (not implemented)."""
    raise NotImplementedError("trilat_zhou not yet implemented")


def trilat_zhou_exhaustive(s, d2):
    """Exhaustive Zhou (not implemented)."""
    raise NotImplementedError("trilat_zhou_exhaustive not implemented")


def trilat_beck_srls(s, d2, W=None, tol=1e-5, maxiter=300):
    """Beck SR-LS algorithm (not implemented)."""
    raise NotImplementedError("trilat_beck_srls not implemented")


def trilat_adachi(s, d2, W=None):
    """Adachi eigenvalue QCQP method."""
    # Build QCQP parameters using Beck formulation
    s = np.asarray(s)
    d2 = np.asarray(d2)
    d, m = s.shape
    if W is None:
        W = np.diag(1.0 / d2)
    W = W / np.sum(W)
    # A, a, B, b, beta per Beck et al
    A_mat = np.hstack([-2 * s.T, np.ones((m, 1))])
    a = d2 - np.sum(s**2, axis=0)
    B_mat = np.diag(np.concatenate([np.ones(d), [0]]))
    b_vec = np.zeros(d + 1)
    beta = -0.5
    lambdahat = 0.0
    y = qcqp(A_mat.T.dot(W).dot(A_mat), A_mat.T.dot(W).dot(a),
             B_mat, b_vec, beta, lambdahat)
    return y[:d]
