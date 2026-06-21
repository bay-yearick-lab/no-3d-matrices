#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.1",
#   "scipy>=1.14",
# ]
# ///
"""
Solve the 3D Poisson equation -nabla^2 u = g on the unit cube with
homogeneous Dirichlet boundaries by two routes to the *same* seven-point
discrete system:

  (A) assemble the monolithic sparse Laplacian (size N^3 x N^3) via Kronecker
      sums and solve it with a sparse direct factorization (SuperLU);
  (B) never assemble it: solve by fast diagonalization, i.e. a discrete sine
      transform along each direction, a pointwise division by the summed
      eigenvalues, and an inverse transform.

Both routes solve the identical discrete system, so their errors against the
manufactured solution u = sin(pi x) sin(pi y) sin(pi z) agree to roundoff.
Only the operator storage (including direct-solver fill-in) and the
wall-clock time differ. The assembled route is run only up to the size where
a direct factorization is still practical; the matrix-free route continues
far past it.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import dstn, idstn

OUTDIR = Path(__file__).resolve().parents[1] / "paper" / "figures"

NAVY = "#1f4e8a"
RED = "#a11e24"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def grid_1d(n: int) -> tuple[np.ndarray, float]:
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) * h
    return x, h


def second_difference_1d(n: int, h: float) -> sp.csr_matrix:
    off = np.ones(n - 1)
    main = -2.0 * np.ones(n)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr") / h**2


def manufactured(n: int):
    x, h = grid_1d(n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    g = 3.0 * np.pi**2 * u_exact  # -lap u = 3 pi^2 u
    return u_exact, g, h


def solve_assembled(n: int, h: float, g: np.ndarray):
    """Route (A): build -L explicitly and factor it with SuperLU."""
    eye = sp.identity(n, format="csr")
    d2 = second_difference_1d(n, h)
    lap = (
        sp.kron(sp.kron(eye, eye), d2)
        + sp.kron(sp.kron(eye, d2), eye)
        + sp.kron(sp.kron(d2, eye), eye)
    )
    A = (-lap).tocsc()
    a_bytes = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
    t0 = time.perf_counter()
    lu = spla.splu(A)
    u = lu.solve(g.ravel())
    dt = time.perf_counter() - t0
    fill_nnz = lu.L.nnz + lu.U.nnz
    fill_bytes = fill_nnz * 12  # ~8B value + 4B index per stored entry
    return u.reshape(g.shape), dt, A.nnz, a_bytes, fill_nnz, fill_bytes


def solve_fast_diagonalization(n: int, h: float, g: np.ndarray, repeats: int = 5):
    """Route (B): matrix-free fast diagonalization via sine transforms."""
    m = np.arange(1, n + 1)
    lam = -(2.0 / h**2) * (1.0 - np.cos(np.pi * m / (n + 1)))  # 1D eigenvalues
    denom = -(lam[:, None, None] + lam[None, :, None] + lam[None, None, :])
    op_bytes = lam.nbytes * 3  # only the 1D eigenvalues are stored

    best, u = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ghat = dstn(g, type=1)
        uhat = ghat / denom
        u = idstn(uhat, type=1)
        best = min(best, time.perf_counter() - t0)
    return u, best, op_bytes


def make_figure(rows: list[dict]) -> None:
    """Two log-log panels: solve time and operator storage versus unknowns."""
    asm = [r for r in rows if r["t_assem"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.loglog([r["dof"] for r in asm], [r["t_assem"] for r in asm],
              "o-", color=NAVY, lw=1.6, ms=5, label="assembled direct solve")
    ax.loglog([r["dof"] for r in rows], [r["t_fast"] for r in rows],
              "s-", color=RED, lw=1.6, ms=5, label="matrix-free fast diagonalization")
    ax.set_xlabel("unknowns $N_x N_y N_z$", fontsize=10)
    ax.set_ylabel("solve time (s)", fontsize=10)
    ax.set_title("(a) wall-clock time for one 3D solve", fontsize=11)

    ax = axes[1]
    ax.loglog([r["dof"] for r in asm], [r["fill_bytes"] / 1e6 for r in asm],
              "o-", color=NAVY, lw=1.6, ms=5, label="assembled factor (fill-in)")
    ax.loglog([r["dof"] for r in rows], [r["op_bytes"] / 1e6 for r in rows],
              "s-", color=RED, lw=1.6, ms=5, label="matrix-free 1D operators")
    ax.set_xlabel("unknowns $N_x N_y N_z$", fontsize=10)
    ax.set_ylabel("operator storage (MB)", fontsize=10)
    ax.set_title("(b) operator storage", fontsize=11)

    # One legend for both panels: the two routes share colours and markers
    # across (a) and (b), so a single key centred above the figures suffices.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2,
               frameon=False, fontsize=9)

    fig.savefig(OUTDIR / "poisson3d_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    assembled_max = 48           # largest N for which we attempt a direct factorization
    sizes = [16, 32, 48, 64, 128, 256]

    cols = (
        f"{'N':>4} {'unknowns':>10} | "
        f"{'A_nnz':>9} {'LUfill_nnz':>11} {'fill_MB':>8} {'t_assem_s':>10} {'err_A':>9} | "
        f"{'op_KB':>7} {'t_fast_s':>9} {'err_B':>9} | {'mem_x':>9} {'time_x':>8} {'agree':>9}"
    )
    print(cols, flush=True)
    print("-" * len(cols), flush=True)

    rows: list[dict] = []
    for n in sizes:
        u_exact, g, h = manufactured(n)
        u_b, t_b, op_b = solve_fast_diagonalization(n, h, g)
        err_b = float(np.max(np.abs(u_b - u_exact)))
        row = {"n": n, "dof": n**3, "t_fast": t_b, "op_bytes": op_b, "err_b": err_b,
               "t_assem": None, "fill_bytes": None}

        if n <= assembled_max:
            u_a, t_a, a_nnz, a_bytes, fill_nnz, fill_bytes = solve_assembled(n, h, g)
            err_a = float(np.max(np.abs(u_a - u_exact)))
            agree = float(np.max(np.abs(u_a - u_b)))
            row.update(t_assem=t_a, fill_bytes=fill_bytes)
            print(
                f"{n:>4} {n**3:>10} | "
                f"{a_nnz:>9} {fill_nnz:>11} {fill_bytes/1e6:>8.1f} {t_a:>10.4f} {err_a:>9.2e} | "
                f"{op_b/1e3:>7.2f} {t_b:>9.5f} {err_b:>9.2e} | "
                f"{fill_bytes/op_b:>9.0f} {t_a/t_b:>8.0f} {agree:>9.1e}",
                flush=True,
            )
            assert agree < 1e-9, f"routes disagree at N={n}: {agree:.2e}"
        else:
            print(
                f"{n:>4} {n**3:>10} | "
                f"{'--':>9} {'--':>11} {'--':>8} {'infeasible':>10} {'--':>9} | "
                f"{op_b/1e3:>7.2f} {t_b:>9.5f} {err_b:>9.2e} | "
                f"{'--':>9} {'--':>8} {'--':>9}",
                flush=True,
            )
        rows.append(row)

    make_figure(rows)
    print(f"Wrote figure to {OUTDIR / 'poisson3d_benchmark.pdf'}", flush=True)


if __name__ == "__main__":
    main()
