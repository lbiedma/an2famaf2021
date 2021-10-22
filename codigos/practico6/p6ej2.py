import numpy as np


def sol_Jacobi(A, b, x_0, err, M):
    n = A.shape[0]
    for i in range(n):
        if A[i, i] == 0:
            print("hay al menos un cero en la diagonal")
            return None
    D = np.diag(np.diag(A))
    
    U = np.triu(A)-D
    
    L = np.tril(A)-D
    
    D_inv = np.zeros((n,n))
    
    for j in range(n):
        D_inv[j,j] = 1/D[j,j]
    
    b = D_inv @ b 
    A = D_inv @ (L + U)
    for k in range(M):
        x_mas = b - A @ x_0
        norm = np.linalg.norm(x_mas - x_0, np.inf)
        if norm <= err:
            print("llegamos a la solucion", k)
            break
        x_0 = x_mas 
    
    return x_mas 

#test Jacobi
A = np.random.random((3,3))
A = np.eye(3) + A
b = np.random.random(3)
x_0 = np.zeros(3)
#x_sol = sol_jacobi(A, b, x_0, 1e-5, 50)

#print(A@x_sol-b)

def sol_gseidel(A, b, x_0, err, M):
    n = A.shape[0]
    for l in range(n):
        if A[l, l] == 0:
            print("hay al menos un cero en la diagonal")
            return None
    D = np.diag(np.diag(A))
    U = np.triu(A)-D
    L = np.tril(A)-D
    D_inv = np.zeros((n,n))
    for j in range(n):
        D_inv[j,j] = 1/D[j,j]
    
    b = D_inv @ b 
    A = D_inv @ (L + U)
    
    
    x_mas = x_0
    for k in range(M):
        x_tilde = x_mas.copy()
        for i in range(n):
            x_mas[i] = b[i] - A[i, :] @ x_mas
            
        norm = np.linalg.norm(x_mas - x_tilde, np.inf)
        if norm <= err:
            print("llegamos a la solucion", k)
            break 
    
    return x_mas 


#Test Gseidel
A = np.random.random((3,3))
A = np.eye(3) + A
b = np.random.random(3)
x_0 = np.zeros(3)

x_sol = sol_gseidel(A, b, x_0, 1e-5, 50)
print(A@x_sol-b)

