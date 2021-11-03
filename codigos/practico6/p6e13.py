import matplotlib.pyplot as plt
import numpy as np

from practico4.p4e7 import sol_cuadmin
from p6e12 import arnoldi
from calor import calor

def sol_gmres(A, b, x_0, err, M, m_A):
    # Paso 0
    x_mas = x_0.copy()
    r = b - A @ x_mas
    sigma = np.linalg.norm(r)

    # Paso 1
    for k in range(M):
        if sigma <= err:
            break
        H, V = arnoldi(A, r, m_A)
        e_1 = np.zeros(H.shape[0])
        e_1[0] = 1.0
        alpha, sigma = sol_cuadmin(H, sigma * e_1)
        x_mas = x_mas + V @ alpha
        r = b - A @ x_mas

    # Paso 2
    return x_mas

# TEST con calor
N = 100
A, b = calor(N)
x_0 = np.random.random((N-2) ** 2)

x_mas = sol_gmres(A, b, x_0, 1e-5, 100, 5)
plt.imshow(x_mas.reshape(N-2, N-2))
plt.show()
