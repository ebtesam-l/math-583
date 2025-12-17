import numpy as np
import matplotlib.pyplot as plt
from math import comb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- Bernstein + Bezier surface ----------------
def bernstein(i, n, t):
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_surface_point(u, v, P):
    n = P.shape[0] - 1
    m = P.shape[1] - 1
    pt = np.zeros(3)
    for i in range(n + 1):
        Bu = bernstein(i, n, u)
        for j in range(m + 1):
            Bv = bernstein(j, m, v)
            pt += P[i, j] * Bu * Bv
    return pt

# ---------------- Plotting ----------------
def plot_bezier_surface(P, nu=30, nv=30):
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)

    X = np.zeros((nu, nv))
    Y = np.zeros((nu, nv))
    Z = np.zeros((nu, nv))

    for a, u in enumerate(us):
        for b, v in enumerate(vs):
            X[a, b], Y[a, b], Z[a, b] = bezier_surface_point(u, v, P)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 🔵 Bezier surface (semi-transparent)
    ax.plot_surface(
        X, Y, Z,
        color="cornflowerblue",
        alpha=0.6,
        edgecolor="none"
    )

    # 🔴 Control points
    ax.scatter(
        P[:, :, 0],
        P[:, :, 1],
        P[:, :, 2],
        color="red",
        s=60,
        label="Control points"
    )

    # ⚫ Control net (grid lines)
    for i in range(P.shape[0]):
        ax.plot(
            P[i, :, 0],
            P[i, :, 1],
            P[i, :, 2],
            color="black",
            linewidth=2
        )

    for j in range(P.shape[1]):
        ax.plot(
            P[:, j, 0],
            P[:, j, 1],
            P[:, j, 2],
            color="black",
            linewidth=2
        )

    ax.set_title("Bezier Surface with Control Net")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()

# ---------------- Example ----------------
def main():
    # 2×2 Bezier surface (bilinear patch)
    P = np.array([
        [[0, 0, 0], [0, 1, 1]],
        [[1, 0, 1], [1, 1, 0]],
    ], dtype=float)

    plot_bezier_surface(P, nu=40, nv=40)

if __name__ == "__main__":
    main()
