import sys
sys.path.append("..")

import numpy as np
from practico1.p1ej3 import soltrsupcol
from p2ej9 import egaussp

def sol_egauss(A, b):
    U, y = egaussp(A, b)
    x = soltrsupcol(U, y)

    return x

A = np.array([[2., 10, 8, 8, 6], [1, 4, -2, 4, -1], [0, 2, 3, 2, 1], [3, 8, 3, 10, 9], [1, 4, 1, 2, 1]])
b = np.array([52., 14, 12, 51, 15])

print(sol_egauss(A, b))
