import numpy as np

def sol_resmin(A, b, x_0, eps=1e-6, m=100):
    # paso 0
    x_mas = x_0.copy()
    r = b - A @ x_mas
    sigma = np.linalg.norm(r)

    # paso 1
    for k in range(m):
        if sigma <= eps:
            break

        v = A @ r
        t = np.dot(r, v) / np.dot(v, v)

        if t <= 0:
            break

        x_mas = x_mas + t * r
        r = r - t * v
        sigma = np.linalg.norm(r)

    # paso 2
    return x_mas

# TEST
N = 5
A = np.random.random((N, N))
A = A @ A.T + np.eye(N)

x_0 = np.random.random(N)
b = np.random.random(N)
x_sol = np.linalg.solve(A, b)
x_res = sol_resmin(A, b, x_0)
assert(np.linalg.norm(x_sol - x_res) <= 1e-5)
