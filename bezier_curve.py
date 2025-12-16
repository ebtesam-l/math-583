import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Basic linear interpolation
# -------------------------
def lerp(A, B, t):
    return (1 - t) * A + t * B


# -------------------------
# De Casteljau with levels
# -------------------------
def de_casteljau_levels(control_points, t):
    """
    Returns all interpolation levels.
    levels[0] = original control points
    levels[-1][0] = final Bezier point
    """
    levels = [np.array(control_points, dtype=float)]
    pts = levels[0]

    while len(pts) > 1:
        pts = np.array([lerp(pts[i], pts[i + 1], t)
                        for i in range(len(pts) - 1)])
        levels.append(pts)

    return levels


# -------------------------
# Bezier curve sampling
# -------------------------
def bezier_curve(control_points, num=200):
    ts = np.linspace(0, 1, num)
    curve = np.array([
        de_casteljau_levels(control_points, t)[-1][0]
        for t in ts
    ])
    return curve


# -------------------------
# Visualization with levels
# -------------------------
def plot_bezier_with_levels(control_points, t):
    levels = de_casteljau_levels(control_points, t)
    cp = np.array(control_points, dtype=float)
    curve = bezier_curve(control_points)

    plt.figure(figsize=(6, 6))

    # Control polygon
    plt.plot(cp[:, 0], cp[:, 1], "o--", label="Control polygon")

    # Bezier curve
    plt.plot(curve[:, 0], curve[:, 1], label="Bezier curve")

    # Plot each level
    for i, lvl in enumerate(levels):
        plt.plot(lvl[:, 0], lvl[:, 1], "o-", label=f"Level {i}")

        # Print points for each level
        print(f"Level {i}:")
        for p in lvl:
            print(f"  {p}")

    # Final Bezier point
    final_point = levels[-1][0]
    plt.scatter(final_point[0], final_point[1], s=100, label="Bezier point")

    plt.title(f"De Casteljau levels at t = {t}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    control_points = [
        [0, 0],   # P0
        [1, 2],   # P1
        [3, 2],   # P2
        [4, 0],   # P3
    ]

    t = 0.5  # choose any value in [0, 1]
    plot_bezier_with_levels(control_points, t)


if __name__ == "__main__":
    main()
