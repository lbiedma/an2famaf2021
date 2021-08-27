import numpy as np

def cholesky(A):
    n = A.shape[0]
    G = np.zeros(A.shape)
    for i in range(n): # I = :i y J = i:
        G[i, i:] = A[i, i:] - G[:i, i].T @ G[:i, i:]
        G[i, i:] = G[i, i:] / np.sqrt(G[i, i])

    return G

A = np.random.random((4, 4))
A = A @ A.T
print(A)

G = cholesky(A)
print(G)

print(G.T @ G)
