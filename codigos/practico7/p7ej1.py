import numpy as np

def fun_tres(x, valfun=True, derfun=False):
    f = None
    df = None
    x1, x2, x3 = x
    exp_x1 = np.exp(-x1)

    if valfun:
        # calcular el valor de la funcion
        f = np.array([
            x1 + x1 * exp_x1 + 3 * x2 + np.sin(x3) - 3,
            np.dot(x, x) - 1,
            x1 ** 2 + 2 * x2 * exp_x1 + 2* x3 - 2
        ])

    if derfun:
        # calcular el valor de la derivada,
        # calculando cada gradiente
        df1 = np.array([
            1 + exp_x1 - x1 * exp_x1,
            3,
            np.cos(x3)
        ])
        df2 = 2 * x
        df3 = np.array([
            2 * x1 - 2 * x2 * exp_x1,
            2 * exp_x1,
            2,
        ])
        # Apilar los gradientes como filas
        df = np.vstack([df1, df2, df3])

    return f, df

# f, df = fun_tres(np.ones(3), True, True)
# print(f)
# print(df)
# print("--------------------------------")

# f, df = fun_tres(np.ones(3), True, False)
# print(f)
# print(df)
# print("--------------------------------")

# f, df = fun_tres(np.ones(3), False, True)
# print(f)
# print(df)

# print("--------------------------------")
# f, df = fun_tres(np.ones(3), False, False)
# print(f)
# print(df)
