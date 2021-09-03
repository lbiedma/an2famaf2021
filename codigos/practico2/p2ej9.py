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

def dlup(A):
    # Suponemos que A es cuadrada
    n = A.shape[0]
    U = A.copy()
    P = np.eye(n)
    detP = 1
    for k in range(n):
        # Busco pivot
        ind_pivoteo = k + np.argmax(np.abs(U[k:, k]))
        if ind_pivoteo != k:
            detP = detP * (-1)
        # Pivotear matriz y P
        U[[k, ind_pivoteo], :] = U[[ind_pivoteo, k], :]
        P[[k, ind_pivoteo], :] = P[[ind_pivoteo, k], :]

        # I = k+1:
        U[k+1:, k] = U[k+1:, k] / U[k, k]
        U[k+1:, k+1:] = U[k+1:, k+1:] - np.outer(U[k+1:, k], U[k, k+1:])

    L = np.tril(U, -1) + np.eye(n)
    U = np.triu(U)

    return L, U, P, detP
