import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Original polynomial (degree 2 example from note)
# P(t) = 3t^2 + 5t - 3
# -------------------------------------------------
def P(t):
    return 3*t*t + 5*t - 3


# -------------------------------------------------
# 2) Blossom of the polynomial
# p(u1, u2) such that p(t, t) = P(t)
# -------------------------------------------------
def blossom(u1, u2):
    return 3*u1*u2 + 2.5*(u1 + u2) - 3


# -------------------------------------------------
# 3) Compute Bezier control points using blossom
# For degree 2 Bezier:
# P0 = p(0,0), P1 = p(0,1), P2 = p(1,1)
# -------------------------------------------------
P0 = blossom(0, 0)
P1 = blossom(0, 1)
P2 = blossom(1, 1)

control_points = np.array([P0, P1, P2])

print("Bezier control points (from Blossom):")
print("P0 =", P0)
print("P1 =", P1)
print("P2 =", P2)


# -------------------------------------------------
# 4) Bernstein basis (degree 2) to build Bezier curve
# -------------------------------------------------
def B0(t): return (1 - t)**2
def B1(t): return 2*t*(1 - t)
def B2(t): return t**2

def bezier(t):
    return P0*B0(t) + P1*B1(t) + P2*B2(t)


# -------------------------------------------------
# 5) Verify blossom diagonal property
# p(t,t) == P(t)
# -------------------------------------------------
t_test = 0.4
print("\nVerification:")
print("P(t)      =", P(t_test))
print("p(t, t)   =", blossom(t_test, t_test))


# -------------------------------------------------
# 6) Plot polynomial and Bezier curve (they match)
# -------------------------------------------------
ts = np.linspace(0, 1, 200)

plt.figure(figsize=(6, 4))
plt.plot(ts, [P(t) for t in ts], label="Original P(t)")
plt.plot(ts, [bezier(t) for t in ts], "--", label="Bezier from Blossom")
plt.scatter([0, 0.5, 1], control_points, zorder=5, label="Control points")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Blossom → Bezier (Same Curve)")
plt.show()
