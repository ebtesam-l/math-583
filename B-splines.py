import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Cox–de Boor B-spline basis N_i^p(t)
# ------------------------------------------------------------
def bspline_basis(i, p, t, knots):
    """
    N_i^p(t) with knot vector 'knots'
    i: basis index
    p: degree
    t: scalar parameter
    """
    if p == 0:
        # Include the very last endpoint so the curve reaches the last point
        if (knots[i] <= t < knots[i + 1]) or (np.isclose(t, knots[-1]) and np.isclose(knots[i + 1], knots[-1])):
            return 1.0
        return 0.0

    denom1 = knots[i + p] - knots[i]
    denom2 = knots[i + p + 1] - knots[i + 1]

    term1 = 0.0
    term2 = 0.0

    if denom1 != 0:
        term1 = (t - knots[i]) / denom1 * bspline_basis(i, p - 1, t, knots)
    if denom2 != 0:
        term2 = (knots[i + p + 1] - t) / denom2 * bspline_basis(i + 1, p - 1, t, knots)

    return term1 + term2


# ------------------------------------------------------------
# B-spline curve point C(t) = sum_i P_i N_i^p(t)
# ------------------------------------------------------------
def bspline_point(t, control_points, degree, knots):
    cps = np.asarray(control_points, dtype=float)
    C = np.zeros(cps.shape[1], dtype=float)
    for i in range(len(cps)):
        C += cps[i] * bspline_basis(i, degree, t, knots)
    return C


# ------------------------------------------------------------
# Open-uniform (clamped) knot vector (correct clamping: p+1 repeats)
# length = m + p + 1
# ------------------------------------------------------------
def make_open_uniform_knots(m, p):
    """
    m: number of control points
    p: degree
    returns knot vector of length (m + p + 1)
    """
    if m < p + 1:
        raise ValueError("Need at least p+1 control points for degree p.")

    # number of interior knots (excluding the repeated ends)
    num_interior = m - p - 1  # can be 0

    if num_interior > 0:
        interior = np.linspace(0, 1, num_interior + 2)[1:-1]  # exclude 0 and 1
        knots = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])
    else:
        knots = np.concatenate([np.zeros(p + 1), np.ones(p + 1)])

    return knots


# ------------------------------------------------------------
# Plot basis functions and curve, and show weights at one t0
# ------------------------------------------------------------
def main():
    # 2D control points
    control_points = np.array([
        [0, 0],
        [1, 2],
        [3, 3],
        [4, 0],
        [6, 2],
        [7, 0],
    ], dtype=float)

    p = 3  # cubic
    m = len(control_points)

    knots = make_open_uniform_knots(m, p)

    # valid parameter range for clamped spline
    t_min = knots[p]
    t_max = knots[-p-1]

    ts = np.linspace(t_min, t_max, 500)

    # --- Plot basis functions ---
    plt.figure(figsize=(8, 4.5))
    for i in range(m):
        vals = [bspline_basis(i, p, t, knots) for t in ts]
        plt.plot(ts, vals, label=f"N{i}^{p}")
    plt.title("B-spline basis functions (open-uniform, clamped)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8)
    plt.show()

    # --- Plot curve ---
    curve = np.array([bspline_point(t, control_points, p, knots) for t in ts])

    plt.figure(figsize=(7.5, 6))
    plt.plot(control_points[:, 0], control_points[:, 1], "o--", label="control polygon")
    plt.plot(curve[:, 0], curve[:, 1], label="B-spline curve")
    plt.title("Cubic B-spline curve (open-uniform, clamped)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # --- Show the connection at one parameter value ---
    t0 = 0.35 * (t_max - t_min) + t_min  # pick a t inside the valid range
    w = np.array([bspline_basis(i, p, t0, knots) for i in range(m)], dtype=float)
    C = (w[:, None] * control_points).sum(axis=0)

    print("Knots:", np.round(knots, 3))
    print("t range:", (t_min, t_max))
    print("t0:", t0)
    print("weights:", np.round(w, 6))
    print("weights sum:", w.sum())
    print("C(t0) from weights:", C)
    print("C(t0) direct:", bspline_point(t0, control_points, p, knots))

if __name__ == "__main__":
    main()
