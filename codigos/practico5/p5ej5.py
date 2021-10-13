import matplotlib.pyplot as plt
import numpy as np

def im_aprox_svd(img, tol):
    m, n = img.shape
    # Descomponer la imagen
    u, s, vt = np.linalg.svd(img)
    for k in range(1, min(m, n)):
        print(f"paso {k}")
        img_aprox = u[:, :k] * s[:k] @ vt[:k, :]
        if np.linalg.norm(img - img_aprox, np.inf) < tol:
            print(f"Llegamos a la tolerancia en el paso {k}")
            break

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original")
    ax[1].imshow(img_aprox, cmap="gray")
    ax[1].set_title(f"Aproximada {k} valores singulares")
    plt.show()

# TEST
img = np.loadtxt("https://raw.githubusercontent.com/lbiedma/an2famaf2020/master/datos/p5e5.txt")
im_aprox_svd(img, 10000)
