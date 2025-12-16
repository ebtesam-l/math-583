import numpy as np
import matplotlib.pyplot as plt
import math


# -------------------------
# 1) Graph of a function: y = f(x)
# -------------------------
def f_graph(x):
    return x**2  # example: parabola


# -------------------------
# 2) Parametric curve: (x(t), y(t))
# -------------------------
def parametric_curve(t):
    x = t**2 - 2*t
    y = t + 1
    return x, y

def tangent_slope_parametric(t):
    dx_dt = 2*t - 2
    dy_dt = 1
    if np.isclose(dx_dt, 0.0):
        return np.inf
    return dy_dt / dx_dt


# -------------------------
# 3) Polar curve: r = f(theta), then x=r cosθ, y=r sinθ
# -------------------------
def r_polar(theta):
    return 1.0  # example: circle r=1

def polar_to_xy(theta):
    r = r_polar(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# -------------------------
# 4) Implicit curve: F(x,y)=0
# Example circle: x^2 + y^2 - 1 = 0
# -------------------------
def F_implicit(x, y):
    return x**2 + y**2 - 1.0


# -------------------------
# Plot helpers
# -------------------------
def plot_graph(ax):
    x = np.linspace(-2, 2, 400)
    y = f_graph(x)
    ax.plot(x, y)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Graph: y = f(x)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

def plot_parametric(ax):
    t = np.linspace(-2, 3, 400)
    x, y = parametric_curve(t)
    ax.plot(x, y)

    # mark a couple sample points + tangent info
    for t0 in [0, 1]:
        x0, y0 = parametric_curve(np.array([t0]))
        x0, y0 = float(x0), float(y0)
        m = tangent_slope_parametric(t0)
        ax.scatter([x0], [y0], s=50)
        if np.isfinite(m):
            # draw small tangent segment
            dx = 0.6
            x_line = np.array([x0 - dx, x0 + dx])
            y_line = y0 + m * (x_line - x0)
            ax.plot(x_line, y_line)
            ax.text(x0, y0, f"  t={t0}, m={m:.2f}")
        else:
            # vertical tangent
            y_line = np.array([y0 - 0.8, y0 + 0.8])
            ax.plot([x0, x0], y_line)
            ax.text(x0, y0, f"  t={t0}, vertical")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Parametric: (x(t), y(t))")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

def plot_polar(ax):
    theta = np.linspace(0, 2*np.pi, 600)
    x, y = polar_to_xy(theta)
    ax.plot(x, y)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Polar: r = f(θ)  →  (x,y)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

def plot_implicit(ax):
    # grid for contour
    xs = np.linspace(-1.5, 1.5, 400)
    ys = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = F_implicit(X, Y)

    # draw F(x,y)=0
    ax.contour(X, Y, Z, levels=[0.0])

    # optional: show sign regions lightly (inside/outside)
    ax.contourf(X, Y, Z, levels=[-10, 0, 10], alpha=0.15)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_title("Implicit: F(x,y)=0 (contour)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    plot_graph(axes[0, 0])
    plot_parametric(axes[0, 1])
    plot_polar(axes[1, 0])
    plot_implicit(axes[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
