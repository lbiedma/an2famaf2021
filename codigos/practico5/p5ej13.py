import numpy as np

def house(x):
  '''
  u, rho = house(x)
  Calcula u y rho tal que Q = I - rho u u^T
  cumple Qx = \|x\|_2 e^1
  '''
  n = len(x)
  rho = 0
  u = x.copy()
  u[0] = 1.

  if n == 1:
    sigma = 0
  else:
    sigma = np.sum(x[1:]**2)

  if sigma>0 or x[0]<0:
    mu = np.sqrt(x[0]**2 + sigma)
    if x[0]<=0:
      gamma = x[0] - mu
    else:
      gamma = -sigma/(x[0] + mu)

    rho = 2*gamma**2/(gamma**2 + sigma)
    u = u/gamma
    u[0] = 1

  return u, rho

def givens(x1,x2):
    '''
    c, s = givens(x1, x2)
    Calcula el coseno y seno para la rotación de Givens
    que hace (x1,x2) -> (y,0).
    '''
    c = 1.
    s = 0.
    ax1 = abs(x1)
    ax2 = abs(x2)
    if ax1 + ax2 > 0:
        if ax2 > ax1:
            tau = -x1/x2
            s = -np.sign(x2)/np.sqrt(1 + tau**2)
            c = tau*s
        else:
            tau = -x2/x1
            c = np.sign(x1)/np.sqrt(1 + tau**2)
            s = tau*c
    return c, s

def fhess(A, p=0):
    m, n = A.shape
    if m != n:
        print("La matriz no es cuadrada")
        return None

    Q = np.eye(m)
    H = A.copy()

    if p == 0: # Hago Householder
        for j in range(n - 2):
            # I = j+1:, J = j:
            u, rho = house(H[j + 1:, j])
            w = rho * u
            H[j + 1:, j:] = H[j + 1:, j:] - np.outer(w, u.T @ H[j + 1:, j:])
            H[:, j + 1:] = H[:, j + 1:] - H[:, j + 1:] @ np.outer(w, u.T)
            Q[:, j + 1:] = Q[:, j + 1:] - Q[:, j + 1:] @ np.outer(w, u.T)
    elif p == 1: # Hago Givens
        for j in range(n - 2):
            for i in range(j + 2, n):
                c, s = givens(H[j + 1, j], H[i, j])
                rot = np.array([[c, -s], [s, c]])
                H[[j + 1, i], j:] = rot @ H[[j + 1, i], j:]
                H[:, [j + 1, i]] = H[:, [j + 1, i]] @ rot.T
                Q[:, [j + 1, i]] = Q[:, [j + 1, i]] @ rot.T
    else:
        print("Elegir un p que sea 0 o 1")
        return None

    return Q, H

# TEST HOUSEHOLDER
# A = np.random.random((5, 5))
# Q, H = fhess(A)
# print(np.linalg.norm(Q @ H @ Q.T - A))

# TEST GIVENS
# A = np.random.random((5, 5))
# Q, H = fhess(A, p=1)
# print(np.linalg.norm(Q @ H @ Q.T - A))
