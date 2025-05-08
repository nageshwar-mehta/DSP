import numpy as np
from scipy.linalg import eig, solve

# S. Adachi and Y. Nakatsukasa, “Eigenvalue-based algorithm and analysis for nonconvex QCQP with one constraint,” Math. Program., vol. 173, no. 1, pp. 79–116, Jan. 2019

def qcqp(A, a, B, b, beta, lambdahat, inftol=1e12):
    """
    Solve minimize x^T A x + 2 a^T x
    subject to x^T B x + 2 b^T x + beta = 0.
    Returns x.
    """
    A = np.asarray(A)
    a = np.asarray(a)
    B = np.asarray(B)
    b = np.asarray(b)
    n = A.shape[0]

    M0 = np.block([
        [beta, b.T, -a.T],
        [b[:,None], B, -A],
        [-a[:,None], -A, np.zeros((n,n))]
    ])
    M1 = np.block([
        [0, np.zeros((1,n)), -b.T],
        [np.zeros((n,1)), np.zeros((n,n)), -B],
        [-b[:,None], -B, np.zeros((n,n))]
    ])
    Mhat = M0 + lambdahat * M1

    xg = -solve(A + lambdahat * B, a + lambdahat * b)
    gamma = xg.T.dot(B).dot(xg) + 2 * b.T.dot(xg) + beta

    gammatol = 1e-16
    if abs(gamma) < gammatol:
        gamma = 0.0

    if gamma > 0:
        eigvals, eigvecs = eig(M1, -Mhat)
        xis = eigvals.real
        keep = np.isfinite(xis) & (xis < inftol)
        xis = xis[keep]
        zs = eigvecs[:, keep]
        xi = np.max(xis)
        z = zs[:, np.argmax(xis)]
        if xi == 0:
            raise ValueError("ξ = 0, consider increasing lambdahat")
        lambd = lambdahat + 1 / xi
    elif gamma < 0:
        eigvals, eigvecs = eig(M1, -Mhat)
        xis = eigvals.real
        keep = np.isfinite(xis) & (xis > -inftol)
        xis = xis[keep]
        zs = eigvecs[:, keep]
        xi = np.min(xis)
        z = zs[:, np.argmin(xis)]
        if xi == 0:
            raise ValueError("ξ = 0, consider increasing lambdahat")
        lambd = lambdahat + 1 / xi
    else:
        lambd = lambdahat
        x = xg

    if gamma != 0:
        theta = z[0]
        y1 = z[1:n+1]
        thetatol = 1e-6
        if abs(theta) > thetatol:
            x = y1 / theta
        else:
            # hard case: singular A + lambda B
            x = -solve(A + lambd * B, a + lambd * b)

    return x
