import numpy as np
import matplotlib.pyplot as plt

def plotear_matriz_a():
    determinantes = []
    condiciones = []

    for i in range(100):
        eps = 2 ** (-i)
        A = np.array([[1., 1 - eps], [0, 1]])

        det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
        cond = np.linalg.cond(A, 2)
        determinantes.append(det)
        condiciones.append(cond)

    plt.plot(determinantes)
    plt.plot(condiciones)
    plt.legend(["determinantes", "numeros de condicion"])
    plt.show()

def plotear_matriz_b():
    determinantes = []
    condiciones = []

    for i in range(100):
        eps = 2 ** (-i)
        A = np.array([[1 / eps, 0], [0, eps]])

        det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
        cond = np.linalg.cond(A, 2)
        determinantes.append(det)
        condiciones.append(cond)

    plt.semilogy(determinantes)
    plt.semilogy(condiciones)
    plt.legend(["determinantes", "numeros de condicion"])
    plt.show()

plotear_matriz_b()
