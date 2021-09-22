import sys
sys.path.append(".")
import numpy as np

from practico4.p4ej5 import qrgivensp
from practico1.p1ej3 import soltrsupcol

def sol_cuadmin(A, b):
    m, n = A.shape
    Q, R, P = qrgivensp(A)
    # construyo vector de solucion con ceros
    y = np.zeros(n)
    q = Q.T @ b
    r_i = np.diag(R)
    p = min(m, n)
    busqueda_cero = np.where(r_i == 0)[0]
    if len(busqueda_cero) != 0:
        p = busqueda_cero[0]

    # resolver R[:p, :p] x = q[:p] y depositar en porcion :p de y
    y[:p] = soltrsupcol(R[:p, :p], q[:p])

    return (P @ y, np.linalg.norm(q[p:]))

# TEST para sol_cuadmin
A = np.random.random((5, 5))
b = np.random.random(5)
x, res_2 = sol_cuadmin(A, b)

sol = np.linalg.lstsq(A, b)
x_np = sol[0]
print(x, x_np)
