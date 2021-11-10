import numpy as np

from p7ej1 import fun_tres

def sol_newton(func, x_0, eps, m):
    # Paso 0
    x = x_0.copy()
    r, M = func(x, valfun=True, derfun=True)
    sigma = np.linalg.norm(r)

    # Paso 1
    for k in range(m):
        if sigma <= eps:
            break

        d = np.linalg.solve(M, -r)
        x = x + d
        r, M = func(x, valfun=True, derfun=True)
        sigma = np.linalg.norm(r)

    # Paso 2
    return x

# TEST
x_0 = np.random.random(3)
x_sol = sol_newton(fun_tres, x_0, 1e-5, 100)
print(x_sol, fun_tres(x_sol)[0])
