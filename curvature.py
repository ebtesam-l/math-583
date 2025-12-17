import numpy as np

# ============================================================
# Helpers
# ============================================================
EPS = 1e-12

def norm(x):
    return np.linalg.norm(x, axis=-1)

def safe_div(a, b):
    return a / (b + EPS)

def cross(a, b):
    return np.cross(a, b)

def dot(a, b):
    return np.sum(a * b, axis=-1)

def det3(a, b, c):
    # determinant of 3 vectors (scalar triple product)
    return dot(a, cross(b, c))

# ============================================================
# 1) CURVE CURVATURE (2D parametric): k(t)
# k(t) = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
# ============================================================
def curvature_2d_parametric(x, y, t):
    """
    x, y, t: 1D arrays of same length (t should be increasing)
    returns k(t) array
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    t = np.asarray(t, float)

    dx  = np.gradient(x, t)
    dy  = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    num = np.abs(dx * ddy - dy * ddx)
    den = (dx*dx + dy*dy)**1.5
    return safe_div(num, den)

# ============================================================
# 2) SPACE CURVE CURVATURE (3D): k(t)
# k(t) = ||r' x r''|| / ||r'||^3
# ============================================================
def curvature_3d_parametric(r, t):
    """
    r: (N,3) array, t: (N,) array
    returns k(t) array
    """
    r = np.asarray(r, float)
    t = np.asarray(t, float)

    r1 = np.gradient(r, t, axis=0)
    r2 = np.gradient(r1, t, axis=0)

    num = norm(cross(r1, r2))
    den = norm(r1)**3
    return safe_div(num, den)

# ============================================================
# 3) TORSION (3D): tau(t)
# tau(t) = det(r', r'', r''') / ||r' x r''||^2
# ============================================================
def torsion_3d_parametric(r, t):
    """
    r: (N,3) array, t: (N,) array
    returns tau(t) array
    """
    r = np.asarray(r, float)
    t = np.asarray(t, float)

    r1 = np.gradient(r, t, axis=0)
    r2 = np.gradient(r1, t, axis=0)
    r3 = np.gradient(r2, t, axis=0)

    num = det3(r1, r2, r3)
    den = norm(cross(r1, r2))**2
    return safe_div(num, den)

# ============================================================
# 4) BEZIER ENDPOINT CURVATURE from control points (planar)
# For degree n Bezier with P0,P1,P2 in 2D:
# k(0) = 2*(n-1)/n * Area(P0,P1,P2) / ||P1-P0||^3
# where Area is signed area of triangle (P0,P1,P2) (2D)
# ============================================================
def triangle_area2(P0, P1, P2):
    # signed double-area / 2; we use absolute area
    x0,y0 = P0
    x1,y1 = P1
    x2,y2 = P2
    return 0.5 * ((x1-x0)*(y2-y0) - (y1-y0)*(x2-x0))

def bezier_endpoint_curvature_2d(P0, P1, P2, n, endpoint="start"):
    """
    endpoint="start": uses (P0,P1,P2) for k(0)
    endpoint="end":   uses (Pn,Pn-1,Pn-2) for k(1) -> pass those as P0,P1,P2 appropriately
    returns scalar curvature
    """
    P0 = np.asarray(P0, float)
    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)

    area = np.abs(triangle_area2(P0, P1, P2))
    denom = np.linalg.norm(P1 - P0)**3
    return (2*(n-1)/n) * safe_div(area, denom)

# ============================================================
# 5) SURFACE CURVATURE (parametric surface): Gaussian K and Mean H
# Using first & second fundamental forms:
# E = Xu·Xu, F = Xu·Xv, G = Xv·Xv
# L = n·Xuu, M = n·Xuv, N = n·Xvv
# K = (L N - M^2)/(E G - F^2)
# H = (E N + G L - 2 F M)/(2(E G - F^2))
# ============================================================
def surface_curvatures_parametric(X, u, v):
    """
    X: function (u,v) -> (3,) numpy array
    u, v: scalars
    Returns: (K, H) at (u,v) using finite differences.

    Note: This is a numerical approximation (good for studying/visualizing).
    """
    h = 1e-4  # finite difference step

    def Xu(u,v):  return (X(u+h,v) - X(u-h,v)) / (2*h)
    def Xv(u,v):  return (X(u,v+h) - X(u,v-h)) / (2*h)
    def Xuu(u,v): return (X(u+h,v) - 2*X(u,v) + X(u-h,v)) / (h*h)
    def Xvv(u,v): return (X(u,v+h) - 2*X(u,v) + X(u,v-h)) / (h*h)
    def Xuv(u,v): return (X(u+h,v+h) - X(u+h,v-h) - X(u-h,v+h) + X(u-h,v-h)) / (4*h*h)

    xu = Xu(u,v); xv = Xv(u,v)
    xuu = Xuu(u,v); xvv = Xvv(u,v); xuv = Xuv(u,v)

    # normal
    nvec = cross(xu, xv)
    nlen = np.linalg.norm(nvec)
    if nlen < EPS:
        return np.nan, np.nan
    nunit = nvec / nlen

    E = dot(xu, xu)
    F = dot(xu, xv)
    G = dot(xv, xv)

    L = dot(nunit, xuu)
    M = dot(nunit, xuv)
    N = dot(nunit, xvv)

    denom = (E*G - F*F)
    K = safe_div((L*N - M*M), denom)
    H = safe_div((E*N + G*L - 2*F*M), (2*denom))
    return float(K), float(H)

# ============================================================
# 6) DISCRETE GAUSSIAN CURVATURE (angle deficit)
# K(v) ≈ 2π - sum(angles around v)
# (Optionally divide by area around vertex; here we return angle deficit only.)
# ============================================================
def angle_at_vertex(a, b, c):
    """
    angle at b in triangle (a,b,c) in 3D
    """
    ba = a - b
    bc = c - b
    cosang = safe_div(dot(ba, bc), (np.linalg.norm(ba)*np.linalg.norm(bc)))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))

def discrete_gaussian_angle_deficit(vertices, faces):
    """
    vertices: (V,3)
    faces: (F,3) int indices
    returns: deficits (V,) where deficit[v] = 2π - sum angles around v
    """
    V = len(vertices)
    deficit = np.full(V, 2*np.pi, dtype=float)

    for (i,j,k) in faces:
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        deficit[i] -= angle_at_vertex(vj, vi, vk)
        deficit[j] -= angle_at_vertex(vi, vj, vk)
        deficit[k] -= angle_at_vertex(vi, vk, vj)

    return deficit

# ============================================================
# 7) DISCRETE MEAN CURVATURE (cotangent Laplacian)
# Hn(v_i) ≈ (1 / (2 A_i)) * sum_j (cot α_ij + cot β_ij) (v_i - v_j)
# This returns:
#   H_mag: |H|  (scalar)
#   Hn: mean curvature normal vector approximation
# ============================================================
def cotangent(u, v):
    # cot(angle) between u and v: cot = dot(u,v)/||u×v||
    return safe_div(dot(u, v), np.linalg.norm(np.cross(u, v)))

def vertex_areas_barycentric(vertices, faces):
    """
    Simple per-vertex area: 1/3 of each incident triangle area (barycentric).
    """
    V = len(vertices)
    A = np.zeros(V, dtype=float)
    for (i,j,k) in faces:
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        area = 0.5*np.linalg.norm(np.cross(vj-vi, vk-vi))
        A[i] += area/3
        A[j] += area/3
        A[k] += area/3
    return A

def discrete_mean_curvature(vertices, faces):
    """
    vertices: (V,3)
    faces: (F,3)
    returns: (H_mag (V,), Hn (V,3))
    """
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)

    Vn = len(vertices)
    W = {}  # edge weights sum cotα+cotβ

    # accumulate cot weights per undirected edge
    for (i,j,k) in faces:
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # angles opposite edges:
        # edge (i,j) opposite k => cot at k
        cot_k = cotangent(vi - vk, vj - vk)
        # edge (j,k) opposite i
        cot_i = cotangent(vj - vi, vk - vi)
        # edge (k,i) opposite j
        cot_j = cotangent(vk - vj, vi - vj)

        def add_edge(a, b, w):
            key = (a,b) if a < b else (b,a)
            W[key] = W.get(key, 0.0) + float(w)

        add_edge(i, j, cot_k)
        add_edge(j, k, cot_i)
        add_edge(k, i, cot_j)

    A = vertex_areas_barycentric(vertices, faces)  # per-vertex area
    Hn = np.zeros((Vn, 3), dtype=float)

    # Build Hn using weights
    for (a,b), w in W.items():
        va = vertices[a]
        vb = vertices[b]
        Hn[a] += w * (va - vb)
        Hn[b] += w * (vb - va)

    # Normalize: (1 / (2 A_i))
    for i in range(Vn):
        Hn[i] = safe_div(Hn[i], (2*A[i]))

    H_mag = 0.5 * norm(Hn)  # common convention: mean curvature magnitude = 0.5||Hn||
    return H_mag, Hn

# ============================================================
# Quick mini-demos (optional): uncomment to test
# ============================================================
if __name__ == "__main__":
    # ---- Demo 2D circle curvature: should be ~1/R ----
    R = 2.0
    t = np.linspace(0, 2*np.pi, 400)
    x = R*np.cos(t)
    y = R*np.sin(t)
    k = curvature_2d_parametric(x, y, t)
    print("2D circle curvature ~", np.mean(k), " expected ", 1/R)

    # ---- Demo 3D helix curvature/torsion (constant) ----
    t = np.linspace(0, 10, 500)
    r = np.stack([np.cos(t), np.sin(t), 0.5*t], axis=1)
    k3 = curvature_3d_parametric(r, t)
    tau = torsion_3d_parametric(r, t)
    print("3D helix mean k:", np.mean(k3), " mean tau:", np.mean(tau))

    # ---- Demo surface (graph z = x^2 + y^2) via parametric X(u,v) = (u,v,u^2+v^2) ----
    def X(u,v):

        return np.array([u, v, u*u + v*v], float)
    K, H = surface_curvatures_parametric(X, 0.2, 0.1)
    print("Surface at (0.2,0.1): K=", K, " H=", H)
