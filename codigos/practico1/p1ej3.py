import numpy as np

def soltrinffil(A, b):
    n = len(b)
    x = b.copy()

    # Para detectar el primer elemento distinto de cero podemos
    # hacer un loop en el range(n) y parar cuando encontramos
    # algo distinto, para luego arrancar desde ahí
    for idx in range(n):
        if b[idx] != 0:
            j = idx
            break

    for i in range(j, n):
        x[i] = (b[i] - A[i, :i] @ x[:i])/A[i,i]

    return x

# Para hacer el recorrido del arreglo en reversa (para triangular superior)
# es posible usar reversed
# for idx in reversed(range(n)):
#     if b[idx] != 0:
#         j = idx
#         break

def soltrsupcol(A, b):
    n = len(b)
    x = b.copy()

    for idx in reversed(range(n)):
        if b[idx] != 0:
            j = idx
            break

    for i in reversed(range(j + 1)):
        x[i] = x[i] / A[i, i]
        x[:i] = x[:i] - A[:i, i] * x[i]

    return x
