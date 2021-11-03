import numpy as np

def arnoldi(A, v, m):
    n = A.shape[0]
    if m > n:
        print("Elegir un m menor o igual a la dimensión")
        return None

    # Paso 0
    H = np.zeros((m + 1, m))
    V = np.zeros((n, m))

    if np.linalg.norm(v) == 0:
        return H, V

    # Paso 1
    for j in range(m):
        V[:, j] = v / np.linalg.norm(v)
        v = A @ V[:, j]

        for i in range(j + 1):
            H[i, j] = np.dot(v, V[:, i])
            v = v - H[i, j] * V[:, i]

        if np.linalg.norm(v) == 0:
            break

        H[j + 1, j] = np.linalg.norm(v)

    # Paso 2
    return H[:j+2, :j+1], V[:, :j+1]

# TEST
N = 5
A = np.random.random((N, N))
v = np.random.random(N)

H, V = arnoldi(A, v, N-1)
# for i in range(N-1):
#     for j in range(N-1):
#         print(np.dot(V[:, i], V[:, j]) - int(i == j) < 1e-5)

# print(V.T @ V)
assert(np.allclose(V.T @ V - np.eye(N-1), 0))

assert(np.allclose(np.tril(H, -2), 0))
