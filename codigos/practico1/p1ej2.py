# Ejemplo de código para probar la demostración del ejercicio 2
# Importamos Numpy
import numpy as np

# Generamos una matriz aleatoria 4x4
A = np.random.random((4, 4))
# Generamos un vector aleatorio de 4 elementos
x = np.random.random(4)

# Definimos b por la multiplicación matriz-vector común y corriente
b = A @ x

# Obtenemos las filas (m) y columnas (n) de A
m, n = A.shape

# Generamos un vector de ceros para irle agregando la sumatoria
b_b = np.zeros(m)
for idx in range(n):
    # Obtenemos la columna idx de A con el indexado y multiplicamos por el elemento idx de x
    # Sumamos al resultado anterior
    b_b = b_b + x[idx] * A[:, idx]

# Si salió todo bien, la resta de ambos arreglos debería darnos ceros o números muy pequeños
print(f"Resta de los vectores: {b_b - b}")
