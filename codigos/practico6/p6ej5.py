import numpy as np

def sol_gastinel(A, b, x_0, eps, m):
    n = A.shape[0]
    r = b - A @ x_0
    d = np.sign(r)
    x_mas = x_0

    for iter in range(m):
        alpha = np.dot(r, d) / np.dot(d, A @ d)
        x_mas = x_mas + alpha * d

        if np.linalg.norm(r, 2) < eps:
            print("Llegamos a la solución")
            break

        r = b - A @ x_mas
        d = np.sign(r)

    return x_mas

# TEST
A = np.random.random((5, 5))
A = A @ A.T + np.eye(5)
b = np.random.random(5)
x_0 = np.random.random(5)
eps = 1e-5
m = 1000

x_star = np.linalg.solve(A, b)
x_gas = sol_gastinel(A, b, x_0, eps, m)
assert(np.linalg.norm(x_star - x_gas) < eps)
