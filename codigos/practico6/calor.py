import numpy as np
from scipy.sparse import csr_matrix

def trans_aux(k, m):
    i = k // m
    j = k % m
    return i, j


def calor(n):
    aes = []
    ies = []
    jes = []
    b = np.zeros((n-2)**2)
    h = 1 / (n-1)

    for k in range((n-2)**2):
        i, j = trans_aux(k, n-2)
        if i == 0:
            b[k] = 100
        # Elemento (k,k) -> (i,j) x (i,j) = 4
        aes.append(4)
        ies.append(k)
        jes.append(k)

        if i != 0:
            aes.append(-1)
            ies.append(k)
            jes.append(k - (n-2))

        if i != n-3:
            aes.append(-1)
            ies.append(k)
            jes.append(k + (n-2))

        if j != 0:
            aes.append(-1)
            ies.append(k)
            jes.append(k-1)

        if j != n-3:
            aes.append(-1)
            ies.append(k)
            jes.append(k+1)

    aes = np.array(aes) / h**2
    b = np.array(b) / h**2

    A = csr_matrix((aes, (ies, jes)), shape=((n-2)**2, (n-2)**2))

    return A, b
