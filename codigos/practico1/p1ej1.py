import numpy as np
# Definimos nuestra matriz A de forma común
A = np.array([[1., 3, 2], [2, 1, 1], [-1, 0, 1]])
print("A = ")
print(A)

# Definimos los bloques de nuestra matriz A
A_00 = np.array([[1.], [2]])
A_01 = np.array([[3., 2], [1, 1]])
A_10 = np.array([-1.])
A_11 = np.array([0, 1.])

# Generamos la matriz A por bloques
A_b = np.block([[A_00, A_01], [A_10, A_11]])

# Nos fijamos si son iguales
print("A - A_b = ")
print(A - A_b)

# Definimos B y C
B = np.array([[1., 0, 1], [2, 1, 1], [-1, 2, 0]])
C = np.array([[5., 7, 4], [3, 3, 3], [-2, 2, -1]])

# Chequeamos que AB = C
print("AB - C =")
print(A @ B - C)

#### Parte superior izquierda de C
# B[0, 0] es un número y A[0:2, 0] es un array, debemos usar producto común (*)
# A[0:2, 1:] es una matriz y B[1:, 0] es un vector, debemos usar @
C_00 = A[0:2, 0] * B[0, 0] + A[0:2, 1:] @ B[1:, 0]

# Estos deberían ser iguales
print("C_00 = ")
print(C_00, C[0:2, 0])

#### Parte superior derecha de C
# A[0:2, 0] y B[0, 1:] son arrays de una dimensión, pero uno es columna y el otro fila
# Para poder obtener una matriz con estos dos arreglos, se usa np.outer.
# A[0:2, 1:] y B[1:, 1:] son matrices, podemos usar el producto @
C_01 = np.outer(A[0:2, 0], B[0, 1:]) + A[0:2, 1:] @ B[1:, 1:]
# Estos deberían ser iguales
print("C_01 = ")
print(C_01, C[0:2, 1:])
