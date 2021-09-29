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

def ejercicio_c(eps):
    A = np.array([[1., 1 - eps], [0, 1]])
    B = np.array([[1 / eps, 0], [0, eps]])

    r = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(r)
    y = np.sin(r)

    trans_A = A @ np.vstack([x, y])
    trans_B = B @ np.vstack([x, y])

    plt.plot(x, y, label="bola unidad")
    plt.plot(trans_A[0, :], trans_A[1, :], label="transformacion por A")
    plt.plot(trans_B[0, :], trans_B[1, :], label="transformacion por B")
    plt.axis("equal")
    plt.legend()
    plt.show()

#plotear_matriz_a()
#plotear_matriz_b()

ejercicio_c(1e-5)
