import numpy as np
import sys
sys.path.append("..")

from practico4.p4ej5 import qrgivensp

### DESCOMPOSICION JACOBI DEL TEORICO
def descom_esp(A):

    if A[0, 1] != 0:
        tau = (A[1,1]-A[0,0])/(2*A[0,1])

        if tau >= 0:
            t = -1/(tau + np.sqrt(tau**2+1))
        else:
            t = 1/(-tau + np.sqrt(tau**2+1))
        c = 1/(np.sqrt(1+t**2))
        s = t*c
    else:
        c = 1
        s = 0
    return c, s

#A = np.random.random((2,2))
#A_sim = 0.5*(A + A.T)
#c, s = descom_esp(A_sim)
#Q = np.array([[c, -s], [s, c]])
#print(Q.T@A_sim@Q)

### EJERCICIO 7
def off(A):
    A_aux = A - np.diag(np.diag(A))
    off_A = (np.linalg.norm(A_aux, "fro"))
    return off_A


def autjacobi(A, err=1e-15, m=500):
    A = 0.5*(A + A.T)
    n = A.shape[0]
    Q = np.eye(n)
    B = A.copy()
    off_A = off(B)
    for k in range(m):

        if off_A >= err:
            B_aux = B - np.diag(np.diag(B))
            i,j = np.unravel_index(np.argmax(np.abs(B_aux)), B.shape)
            c, s = descom_esp(np.array([[B[i, i], B[i, j]], [B[j, i], B[j, j]]]))
            J = np.array([[c, -s], [s, c]])
            B[[i,j], :] = J.T @ B[[i,j], :]
            B[:, [i,j]] = B[:, [i,j]]@J
            Q[:, [i,j]] = Q[:, [i,j]]@J
            off_A = off(B)
        else:
            print("iteraciones", k)
            return B, Q

    return B, Q

### EJERCICIO 8
def dvsingulares(A):
    C = A.T @ A
    D, W = autjacobi(C)
    Q, R, P = qrgivensp(A @ W)

    return Q, R, W @ P

### TEST
A = np.random.random((4,5))
print('A=')
print(A)

print('SVD de numpy=')
U1,S1,vt=np.linalg.svd(A)
k=len(S1)
print(U1*S1@vt[:k,:])
print('Valores singulares=')
print(S1)

U, S, V = dvsingulares(A)
print('Descomposicion SVD con dvsingulares=')
print(U@S@V.T)
print('Valores singulares=')
print(np.diag(S))

print(np.allclose(U@S@V.T,(U1*S1)@vt[:k,:]))
