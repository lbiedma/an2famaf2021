from p4ej3 import givens

import numpy as np

def qrgivensp(A):
    m, n = A.shape
    Q = np.eye(m)
    P = np.eye(n)
    R = A.copy()
    # Forma rápida de calcular la norma de todas las columnas de A
    c_norm = np.linalg.norm(A, axis=0) ** 2
    p = min(m - 1, n)

    for j in range(p):
        # Pivoteo por derecha, cambiando columnas en vez de filas
        ind_pivot = j + np.argmax(c_norm[j:])
        if ind_pivot != j:
            R[:, [ind_pivot, j]] = R[:, [j, ind_pivot]]
            P[:, [ind_pivot, j]] = P[:, [j, ind_pivot]]
            c_norm[[ind_pivot, j]] = c_norm[[j, ind_pivot]]
        # Givens común y corriente
        for i in range(j+1, m):
            if R[i, j] != 0:
                #I = [j, i], J = j:
                c, s = givens(R[j, j], R[i, j])
                G = np.array([[c, -s], [s, c]])

                R[[j, i], j:] = G @ R[[j, i], j:]
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T
        # Actualización de c_norm SOLO UNA VEZ DESPUES DE ACTUALIZAR FILAS (error de Luis)
        c_norm[j:] = c_norm[j:] - R[j, j:] * R[j, j:]

    if m <= n and R[m - 1, m - 1] < 0:
        # J = m:
        R[m - 1, m - 1:] = -R[m - 1, m - 1:]
        Q[:, m - 1] = -Q[:, m - 1]

    return Q, R, P

# TEST QRGIVENSP
A = np.random.random((5, 5))
Q, R, P = qrgivensp(A)
print(np.linalg.norm(A @ P - Q @ R))
