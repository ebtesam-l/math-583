import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Original cubic polynomial
# P(t) = t^3 + 2t^2 - t + 1
# -------------------------------------------------
def P(t):
    return t**3 + 2*t**2 - t + 1


# -------------------------------------------------
# 2) Cubic blossom p(u1, u2, u3)
# Constructed so that p(t,t,t) = P(t)
# -------------------------------------------------
def blossom(u1, u2, u3):
    return (
        u1*u2*u3                       # t^3 term
        + (2/3)*(u1*u2 + u1*u3 + u2*u3) # 2t^2 term
        - (1/3)*(u1 + u2 + u3)          # -t term
        + 1                             # constant
    )


# -------------------------------------------------
# 3) Bezier control points from blossom
# -------------------------------------------------
P0 = blossom(0, 0, 0)
P1 = blossom(0, 0, 1)
P2 = blossom(0, 1, 1)
P3 = blossom(1, 1, 1)

control_points = np.array([P0, P1, P2, P3])

print("Cubic Bezier control points (from Blossom):")
print("P0 =", P0)
print("P1 =", P1)
print("P2 =", P2)
print("P3 =", P3)


# -------------------------------------------------
# 4) Bernstein basis (degree 3)
# -------------------------------------------------
def B0(t): return (1 - t)**3
def B1(t): return 3*t*(1 - t)**2
def B2(t): return 3*t**2*(1 - t)
def B3(t): return t**3

def bezier(t):
    return (
        P0*B0(t) +
        P1*B1(t) +
        P2*B2(t) +
        P3*B3(t)
    )


# -------------------------------------------------
# 5) Verify diagonal property
# -------------------------------------------------
t_test = 0.35
print("\nVerification:")
print("P(t)        =", P(t_test))
print("p(t,t,t)    =", blossom(t_test, t_test, t_test))


# -------------------------------------------------
# 6) Plot comparison
# -------------------------------------------------
ts = np.linspace(0, 1, 300)

plt.figure(figsize=(6, 4))
plt.plot(ts, [P(t) for t in ts], label="Original P(t)")
plt.plot(ts, [bezier(t) for t in ts], "--", label="Bezier via Blossom")
plt.scatter([0, 1/3, 2/3, 1], control_points, zorder=5, label="Control points")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Cubic Blossom → Cubic Bezier (Same Curve)")
plt.show()
