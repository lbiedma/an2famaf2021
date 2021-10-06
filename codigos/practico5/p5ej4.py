import numpy as np

def cuad_min_svd(A, b):
    m, n = A.shape
    u, s, vt = np.linalg.svd(A)
    b_tilde = u.T @ b
    rango = (np.where(np.abs(s) < 1e-15)[0])[0]
    y = b_tilde[:rango] / s[:rango]
    x_tilde = np.zeros(n)
    x_tilde[:rango] = y

    x = vt.T @ x_tilde
    return x, np.linalg.norm(b - A @ x, 2)
