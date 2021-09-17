import numpy as np

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

def qrgivens(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    p = min(m - 1, n)

    for j in range(p):
        for i in range(j+1, m):
            if R[i, j] != 0:
                #I = [j, i], J = j:
                c, s = givens(R[j, j], R[i, j])
                G = np.array([[c, -s], [s, c]])

                R[[j, i], j:] = G @ R[[j, i], j:]
                Q[:, [j, i]] = Q[:, [j, i]] @ G.T

    if m <= n and R[m - 1, m - 1] < 0:
        # J = m:
        R[m - 1, m - 1:] = -R[m - 1, m - 1:]
        Q[:, m - 1] = -Q[:, m - 1]

    return Q, R

def qrhholder(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    p = min(m, n)

    for j in range(p):
        # I = j:, J = j:
        u, rho = house(R[j:, j])
        w = rho * u
        R[j:, j:] = R[j:, j:] - np.outer(w, u.T @ R[j:, j:])
        Q[:, j:] = Q[:, j:] - Q[:, j:] @ np.outer(w, u)

    return Q, R

def qrgschmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        # I = :j
        R[:j, j] = Q[:, :j].T @ A[:, j]
        q = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(q, 2)
        Q[:, j] = q / R[j, j]

    return Q, R

# TEST DE QRGIVENS
# A = np.random.random((5, 5))
# Q, R = qrgivens(A)
# print(np.linalg.norm(A - Q @ R))

# TEST DE QRHHOLDER
# A = np.random.random((5, 5))
# Q, R = qrhholder(A)
# print(np.linalg.norm(A - Q @ R))

# TEST DE QRGSCHMIDT
# A = np.random.random((5, 5))
# Q, R = qrgschmidt(A)
# print(np.linalg.norm(A - Q @ R))
