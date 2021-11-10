import numpy as np

def fun_mlocal(x, valfun=True, gradfun=False, hesfun=False):
    n = len(x)
    f, df, d2f = None, None, None

    if valfun:
        f = (1 - x[-1]) ** 3 * np.sum(x[:-1]**2) + x[-1]**2

    if gradfun:
        df = np.zeros(n)
        for i in range(n-1):
            df[i] = 2 * (1 - x[-1])**3 * x[i]

        df[-1] = -3 * (1 - x[-1])**2 * np.sum(x[:-1]**2) + 2 * x[-1]

    if hesfun:
        d2f = np.zeros((n, n))

        for i in range(n-1):
            d2f[i, i] = 2 * (1 - x[-1])**3
            d2f[n-1, i] = -6 * (1 - x[-1])**2 * x[i]
            d2f[i, n-1] = d2f[n-1, i]

        d2f[n-1, n-1] = 6 * (1 - x[-1]) * np.sum(x[:-1]**2) + 2


    return f, df, d2f
