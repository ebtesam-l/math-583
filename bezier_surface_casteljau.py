import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def lerp(A, B, t):
    return (1 - t) * A + t * B

def de_casteljau_curve_point(points, t):
    """
    points: (k, dim) control points for a Bezier curve
    returns: point on curve at t
    """
    pts = np.array(points, dtype=float)
    while len(pts) > 1:
        pts = np.array([lerp(pts[i], pts[i+1], t) for i in range(len(pts)-1)])
    return pts[0]

def de_casteljau_surface_point(P, u, v):
    """
    P: control net shape (n+1, m+1, 3)
    returns S(u,v) using De Casteljau surface construction
    """
    P = np.array(P, dtype=float)
    n1, m1, dim = P.shape  # n1=n+1, m1=m+1

    # 1) collapse in u for each column j -> R_j(u)
    R = []
    for j in range(m1):
        curve_pts = P[:, j, :]          # points P_{0j}..P_{nj}
        Rj = de_casteljau_curve_point(curve_pts, u)
        R.append(Rj)
    R = np.array(R)  # shape (m+1, 3)

    # 2) collapse in v on the resulting points -> S(u,v)
    return de_casteljau_curve_point(R, v)

def plot_bezier_surface_casteljau(P, nu=25, nv=25):
    us = np.linspace(0, 1, nu)
    vs = np.linspace(0, 1, nv)

    X = np.zeros((nu, nv))
    Y = np.zeros((nu, nv))
    Z = np.zeros((nu, nv))

    for a, u in enumerate(us):
        for b, v in enumerate(vs):
            x, y, z = de_casteljau_surface_point(P, u, v)
            X[a, b], Y[a, b], Z[a, b] = x, y, z

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Surface (blue)
    ax.plot_surface(X, Y, Z, color="cornflowerblue", alpha=0.6, edgecolor="none")

    # Control net (black lines) + control points (red)
    P = np.array(P, dtype=float)
    ax.scatter(P[:, :, 0], P[:, :, 1], P[:, :, 2], color="red", s=60)

    for i in range(P.shape[0]):
        ax.plot(P[i, :, 0], P[i, :, 1], P[i, :, 2], color="black", linewidth=2)
    for j in range(P.shape[1]):
        ax.plot(P[:, j, 0], P[:, j, 1], P[:, j, 2], color="black", linewidth=2)

    ax.set_title("Bezier Surface via De Casteljau Construction")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()

# ---- Example control net (2x2 => bilinear patch) ----
P_example = np.array([
    [[0, 0, 0], [0, 1, 1]],
    [[1, 0, 1], [1, 1, 0]],
], dtype=float)

plot_bezier_surface_casteljau(P_example, nu=40, nv=40)
