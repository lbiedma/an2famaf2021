import numpy as np

def egaussp(A, b):
    m, n = A.shape
    U = A.copy()
    y = b.copy()

    for idx in range(min(m - 1, n)):
        if np.max(np.abs(U[idx:, idx])) != 0:
            # Elegimos el pivot
            pivot = idx + np.argmax(np.abs(U[idx:, idx]))
            # Pivoteamos (sin multiplicar por matriz elemental)
            U[[idx, pivot], :] = U[[pivot, idx], :]
            y[[idx, pivot]] = y[[pivot, idx]]
            # Reducir
            v = U[idx + 1:, idx] / U[idx, idx]
            U[idx + 1:, idx] = 0
            U[idx + 1:, idx + 1:] = U[idx + 1:, idx + 1:] - np.outer(v, U[idx, idx + 1:])
            y[idx + 1:] = y[idx + 1:] - v * y[idx]

    return U, y
