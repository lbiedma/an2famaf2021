import sys
sys.path.append("..")

from p4ej7 import sol_cuadmin
import numpy as np
import matplotlib.pyplot as plt

def predictor(N, tau):
    datos = np.loadtxt("/home/lbiedma/Documents/clases/numerico2/2021/datos/dryer2.dat")
    t = datos[:, 0]
    u = datos[:, 1]
    y = datos[:, 4]
    tiempo_final = len(t)

    A = np.zeros((N - tau, tau))
    b = y[tau:N]

    for ind in range(tau, N - tau + 1):
        A[ind - tau, :] = u[ind - tau:ind]

    h = sol_cuadmin(A, b)[0]

    entradas = np.zeros((tiempo_final - N, tau))

    for ind in range(tiempo_final - N):
        entradas[ind, :] = u[ind + N - tau:ind + N]

    predicciones = entradas @ h

    plt.plot(t, u, '.', t, y, t[N:], predicciones)
    plt.show()

predictor(500, 101)
