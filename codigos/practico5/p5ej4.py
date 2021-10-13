import numpy as np
# import sys
# sys.path.append(".")
# from practico4.p4ej7 import sol_cuadmin

def cuad_min_svd(A, b):
    m, n = A.shape
    # Descomponer A
    u, s, vt = np.linalg.svd(A)
    # Multiplicar por U.T para obtener b_tilde
    b_tilde = u.T @ b
    # Encontrar rango de Sigma
    # Suponemos que rango inicial es min(m, n)
    rango = min(m, n)
    # Si llegamos a tener ceros, buscamos el primero
    busqueda_cero = np.where(np.abs(s) < 1e-14)[0]
    if len(busqueda_cero) > 0:
        rango = busqueda_cero[0]
    # Resolver s[:rango] y = b_tilde
    y = np.zeros(n)
    y[:rango] = b_tilde[:rango] / s[:rango]
    # Multiplicar V y para obtener x
    x = vt.T @ y

    return x, np.linalg.norm(A @ x - b, 2)

# TEST
A = np.genfromtxt("https://raw.githubusercontent.com/lbiedma/an2famaf2020/master/datos/A_p5e4.txt")
b = np.genfromtxt("https://raw.githubusercontent.com/lbiedma/an2famaf2020/master/datos/b_p5e4.txt")

x, norma_2 = cuad_min_svd(A, b)
print(x, norma_2)

# x_qr, norma_2_qr = sol_cuadmin(A, b)
# print(x_qr, norma_2_qr)

print(np.linalg.lstsq(A, b)[0])
