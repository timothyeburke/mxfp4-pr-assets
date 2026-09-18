#!/usr/bin/env python3
"""
Verify MXAttention's Universal Optimal Scaling (UOS) Qmax.

The paper (arxiv 2607.24377) derives a distribution-independent optimal scaling
boundary Qmax by minimizing a global quantization-error objective:
  E(x) = integral_0^x (v - Pi(v))^2 dv        (cumulative projection error)
  D(x) = E(x) / x^3                            (dimensionless relative error)
  J(q) = integral_{q/2}^{q} D(x) / (x ln2) dx  (log-uniform density over one octave)
where Pi is round-to-nearest onto the element grid (with saturation at the max
level). The paper reports Qmax = 7.25 for E2M1 (MXFP4).

This script reproduces that objective and checks:
  1. E2M1  -> does it give Qmax = 7.25?   (validates the method)
  2. E4M3  -> what is the optimal Qmax?    (the 464 question: is 464.0f right?)
"""
import numpy as np

def build_grid_e2m1():
    # nonnegative finite E2M1 values, saturation at 6.
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def build_grid_e4m3():
    # nonnegative finite E4M3 (OCP, bias 7, 4 exp + 3 mantissa; e=15,m=7 is NaN).
    vals = [0.0]
    # subnormals: 2^-6 * (m/8), m=1..7
    for m in range(1, 8):
        vals.append(2.0**-6 * (m / 8.0))
    # normals: 2^(e-7) * (1 + m/8)
    for e in range(1, 15):
        for m in range(0, 8):
            vals.append(2.0**(e - 7) * (1.0 + m / 8.0))
    e = 15
    for m in range(0, 7):  # m=7 is NaN, excluded
        vals.append(2.0**(e - 7) * (1.0 + m / 8.0))
    g = np.array(sorted(set(vals)))
    return g

def make_projector(grid, sat):
    """Return a fast round-to-nearest projector (sorted grid) with saturation."""
    mid = 0.5 * (grid[:-1] + grid[1:])
    def project(x):
        i = np.clip(np.searchsorted(mid, x), 0, len(grid) - 1)
        return np.minimum(grid[i], sat)
    return project

def D_of_x(x, project, n=20000):
    # D(x) = E(x)/x^3, E(x) = integral_0^x (v - Pi(v))^2 dv (cumulative).
    xmax = float(np.max(x))
    v = np.linspace(0.0, xmax, n)
    err2 = (v - project(v)) ** 2
    E = np.cumsum(err2) * (xmax / (n - 1))
    E_at = np.interp(x, v, E)
    return E_at / (x ** 3 + 1e-30)

def J_of_q(q, project, n=2000):
    x = np.linspace(q / 2.0, q, n)
    D = D_of_x(x, project)
    return float(np.trapezoid(D / (x * np.log(2.0)), x))

def find_optimal_q(grid, sat, lo, hi):
    project = make_projector(grid, sat)
    qs = np.linspace(lo, hi, 300)
    Js = np.array([J_of_q(q, project) for q in qs])
    i = int(np.argmin(Js))
    lo2, hi2 = qs[max(0, i - 2)], qs[min(len(qs) - 1, i + 2)]
    qs2 = np.linspace(lo2, hi2, 150)
    Js2 = np.array([J_of_q(q, project) for q in qs2])
    i2 = int(np.argmin(Js2))
    return float(qs2[i2])


if __name__ == "__main__":
    g4 = build_grid_e2m1()
    print(f"E2M1 grid: {g4.tolist()}  (max={g4[-1]})")
    q4 = find_optimal_q(g4, g4[-1], 4.0, 12.0)
    print(f"E2M1  optimal Qmax = {q4:.4f}   (paper: 7.25)")

    g8 = build_grid_e4m3()
    print(f"\nE4M3 grid: {len(g8)} values, max={g8[-1]}, min_nonzero={g8[1]:.3e}")
    q8 = find_optimal_q(g8, g8[-1], 300.0, 700.0)
    print(f"E4M3  optimal Qmax = {q8:.4f}   (kept in code: 464.0f)")
