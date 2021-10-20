import numpy as np

from p5ej13 import fhess, givens

def autqr(A, eps=1e-10, m=500):
    n = A.shape[0]
    Q, H = fhess(A)
    rotaciones = np.zeros((n, 2))

    for k in range(m):
        for j in range(n - 1):
            # I = [j, j + 1], J = j:
            c, s = givens(H[j, j], H[j + 1, j])
            rotaciones[j, :] = np.array([c, s])
            rot = np.array([[c, -s], [s, c]])
            H[[j, j + 1], j:] = rot @ H[[j, j + 1], j:]

        for l in range(n - 1):
            # I = [l, l + 1]
            c, s = rotaciones[l, :]
            rot = np.array([[c, -s], [s, c]])
            H[:, [l, l + 1]] = H[:, [l, l + 1]] @ rot.T
            Q[:, [l, l + 1]] = Q[:, [l, l + 1]] @ rot.T

        if np.linalg.norm(H - np.diag(np.diag(H)), 'fro') < eps:
            print("Llegamos a la tolerancia!")
            break

    return H, Q

# TEST AUTQR
A = np.random.random((3, 3))
H, Q = autqr(A)
print(np.linalg.eigvals(A))
print(np.diag(H))

print(np.linalg.norm(Q @ H @ Q.T - A))
