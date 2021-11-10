import matplotlib.pyplot as plt
import numpy as np
from p7ej3 import sol_newton

def func_ej4(x, valfun=True, derfun=False):
    f, df = None, None
    x1 = x[0]
    x2 = x[1]
    if valfun:
        f = np.array([
            x1**2 - x1 - x2,
            x1**2 / 16 + x2**2 - 1,
        ])

    if derfun:
        df = np.array([
            [2 * x1 - 1, -1.],
            [x1 / 8, 2 * x2],
        ])

    return f, df

# Graficar
# Parábola
x1 = np.linspace(-4, 4, 100)
x2 = x1 ** 2 - x1
plt.plot(x1, x2)

# Elipse
theta = np.linspace(0, 2 * np.pi, 100)
y1 = 4 * np.cos(theta)
y2 = np.sin(theta)
plt.plot(y1, y2)

# Aplicamos Newton con 2 puntos cerca de nuestras estimaciones de raíces
inicial_1 = np.array([-1., 1])
inicial_2 = np.array([2., 1])
raiz_1 = sol_newton(func_ej4, inicial_1, 1e-6, 100)
raiz_2 = sol_newton(func_ej4, inicial_2, 1e-6, 100)
print(raiz_1, raiz_2)
plt.plot(raiz_1[0], raiz_1[1], '*r')
plt.plot(raiz_2[0], raiz_2[1], '*r')

plt.show()
