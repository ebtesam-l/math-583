import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --------------------------------------------------
# Cox–de Boor B-spline basis
# --------------------------------------------------
def bspline_basis(i, p, t, knots):
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0

    denom1 = knots[i+p] - knots[i]
    denom2 = knots[i+p+1] - knots[i+1]

    term1 = 0.0
    term2 = 0.0

    if denom1 != 0:
        term1 = (t - knots[i]) / denom1 * bspline_basis(i, p-1, t, knots)
    if denom2 != 0:
        term2 = (knots[i+p+1] - t) / denom2 * bspline_basis(i+1, p-1, t, knots)

    return term1 + term2


# --------------------------------------------------
# B-spline surface point
# --------------------------------------------------
def bspline_surface_point(u, v, P, p, q, U, V):
    n = P.shape[0] - 1
    m = P.shape[1] - 1

    S = np.zeros(3)
    for i in range(n + 1):
        Nu = bspline_basis(i, p, u, U)
        for j in range(m + 1):
            Nv = bspline_basis(j, q, v, V)
            S += P[i, j] * Nu * Nv

    return S


# --------------------------------------------------
# Plotting function (same colors as before)
# --------------------------------------------------
def plot_bspline_surface(P, p, q, U, V, nu=30, nv=30):
    us = np.linspace(U[p], U[-p-1], nu)
    vs = np.linspace(V[q], V[-q-1], nv)

    X = np.zeros((nu, nv))
    Y = np.zeros((nu, nv))
    Z = np.zeros((nu, nv))

    for a, u in enumerate(us):
        for b, v in enumerate(vs):
            X[a, b], Y[a, b], Z[a, b] = bspline_surface_point(u, v, P, p, q, U, V)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 🔵 B-spline surface
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

    # ⚫ Control net (grid)
    for i in range(P.shape[0]):
        ax.plot(P[i, :, 0], P[i, :, 1], P[i, :, 2],
                color="black", linewidth=2)
    for j in range(P.shape[1]):
        ax.plot(P[:, j, 0], P[:, j, 1], P[:, j, 2],
                color="black", linewidth=2)

    ax.set_title("B-spline Surface with Control Net")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.show()


# --------------------------------------------------
# Example
# --------------------------------------------------
def main():
    # 3x3 control net
    
    P = np.array([
        [[0,0,0], [0,1,1], [0,2,0]],
        [[1,0,1], [1,1,2], [1,2,1]],
        [[2,0,0], [2,1,1], [2,2,0]],
    ], dtype=float)

    p = q = 2  # quadratic surface

    # Open-uniform knot vectors
    U = [0,0,0,1,1,1]
    V = [0,0,0,1,1,1]

    plot_bspline_surface(P, p, q, U, V, nu=40, nv=40)

if __name__ == "__main__":
    main()
