import matplotlib.pyplot as plt
import numpy as np

def level(niveles):
    # Una matriz aleatoria multiplicada por su conjugada
    # va a ser simétrica definida positiva
    B = np.random.random((2, 2))
    A = B @ B.T

    # Ponemos la cantidad de puntos que vamos a usar para
    # nuestra grilla
    puntos = 100

    # Armamos grillas unidimensionales
    x = np.linspace(-2, 2, puntos)
    y = np.linspace(-2, 2, puntos)

    # Creamos una grilla bidimensional, generando dos
    # matrices que nos dan el elemento de cada par ordenado
    xx, yy = np.meshgrid(x, y)
    # Armamos nuestra grilla de evaluación de x.T A x
    zz = np.zeros((puntos, puntos))

    # Recorremos la grilla y evaluamos la función en cada punto
    for idx in range(puntos):
        for jdx in range(puntos):
            v = np.array([xx[idx, jdx], yy[idx, jdx]])
            zz[idx, jdx] = v.T @ A @ v

    # Graficamos nuestras curvas de nivel
    plt.contour(xx, yy, zz, niveles)
    plt.show()
