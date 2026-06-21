#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.1",
#   "scipy>=1.14",
# ]
# ///

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # render to file only; no display / interactive backend needed
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


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


BLUE = "#1f4e8a"
RED = "#a11e24"


def open_uniform_knots(n: int, degree: int) -> np.ndarray:
    interior_count = n - degree - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
        return np.concatenate(
            [np.zeros(degree + 1), interior, np.ones(degree + 1)]
        )
    return np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])


def greville_points(knots: np.ndarray, degree: int, n: int) -> np.ndarray:
    if degree == 0:
        return knots[:n]
    return np.array(
        [np.sum(knots[i + 1 : i + degree + 1]) / degree for i in range(n)]
    )


def bspline_basis_matrix(
    knots: np.ndarray, degree: int, points: np.ndarray, deriv: int = 0
) -> np.ndarray:
    n = len(knots) - degree - 1
    mat = np.zeros((len(points), n))
    for i in range(n):
        coeffs = np.zeros(n)
        coeffs[i] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        if deriv:
            spline = spline.derivative(deriv)
        values = spline(points)
        values = np.where(np.isnan(values), 0.0, values)
        mat[:, i] = values
    return mat


def compact_first_derivative_mats(n: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((n, n))
    r = np.zeros((n, n))

    a[0, 0] = 1.0
    a[-1, -1] = 1.0
    r[0, 0:3] = np.array([-1.5, 2.0, -0.5])
    r[-1, -3:] = np.array([0.5, -2.0, 1.5])

    for i in range(1, n - 1):
        a[i, i - 1] = 0.25
        a[i, i] = 1.0
        a[i, i + 1] = 0.25
        r[i, i - 1] = -3.0 / 4.0
        r[i, i + 1] = 3.0 / 4.0
    return a, r


def explicit_centered_mats(n: int) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.zeros((n, n))
    d2 = np.zeros((n, n))

    d1[0, 0:3] = np.array([-1.5, 2.0, -0.5])
    d1[-1, -3:] = np.array([0.5, -2.0, 1.5])
    d2[0, 0:4] = np.array([2.0, -5.0, 4.0, -1.0])
    d2[-1, -4:] = np.array([-1.0, 4.0, -5.0, 2.0])

    for i in range(1, n - 1):
        d1[i, i - 1] = -0.5
        d1[i, i + 1] = 0.5
        d2[i, i - 1] = 1.0
        d2[i, i] = -2.0
        d2[i, i + 1] = 1.0
    return d1, d2


def galerkin_spline_mats(n: int, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots = open_uniform_knots(n, degree)
    xi, wi = leggauss(12)
    spans = [(a, b) for a, b in zip(knots[:-1], knots[1:]) if b > a]
    points = []
    weights = []
    for a, b in spans:
        points.append(0.5 * (b - a) * xi + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * wi)
    points = np.concatenate(points)
    weights = np.concatenate(weights)

    bmat = bspline_basis_matrix(knots, degree, points, deriv=0)
    dbmat = bspline_basis_matrix(knots, degree, points, deriv=1)
    wdiag = np.diag(weights)

    mass = bmat.T @ wdiag @ bmat
    transport = bmat.T @ wdiag @ dbmat
    stiffness = dbmat.T @ wdiag @ dbmat
    return mass, transport, stiffness


def collocation_spline_mats(
    n: int, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots = open_uniform_knots(n, degree)
    points = greville_points(knots, degree, n)
    bmat = bspline_basis_matrix(knots, degree, points, deriv=0)
    dbmat = bspline_basis_matrix(knots, degree, points, deriv=1)
    ddbmat = bspline_basis_matrix(knots, degree, points, deriv=2)
    return bmat, dbmat, ddbmat


def implicit_line_mats(
    n: int, degree: int, mu: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, d2 = explicit_centered_mats(n)
    a_fd = np.eye(n) - mu * d2

    a_compact, r_compact = compact_first_derivative_mats(n)
    a_compact2 = a_compact - mu * (
        np.eye(n, k=-1) - 2.0 * np.eye(n) + np.eye(n, k=1)
    )
    a_compact2[0, :] = 0.0
    a_compact2[-1, :] = 0.0
    a_compact2[0, 0] = 1.0
    a_compact2[-1, -1] = 1.0

    mass, _, stiffness = galerkin_spline_mats(n, degree)
    a_galerkin = mass + mu * stiffness
    return a_fd, a_compact2, a_galerkin


def nz_coords(mat: np.ndarray, tol: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(np.abs(mat) > tol)
    return cols, rows


def setup_axis(ax: plt.Axes, ncols: int, nrows: int, xlabel: str, ylabel: str) -> None:
    ticks_x = np.arange(0, ncols, 5)
    ticks_y = np.arange(0, nrows, 5)
    ax.set_xlim(-0.8, ncols - 0.2)
    ax.set_ylim(nrows - 0.2, -0.8)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)
    ax.set_aspect("equal")


def plot_single(ax: plt.Axes, mat: np.ndarray, title: str, color: str, xlabel: str, ylabel: str) -> None:
    cols, rows = nz_coords(mat)
    ax.scatter(cols, rows, s=11, c=color, marker="s", linewidths=0)
    setup_axis(ax, mat.shape[1], mat.shape[0], xlabel, ylabel)
    ax.set_title(title, fontsize=11)


def plot_overlay(
    ax: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    title: str,
    label_first: str,
    label_second: str,
    xlabel: str,
    ylabel: str,
) -> None:
    cols_a, rows_a = nz_coords(first)
    cols_b, rows_b = nz_coords(second)
    ax.scatter(cols_a, rows_a, s=11, c=BLUE, marker="s", linewidths=0, label=label_first)
    ax.scatter(cols_b, rows_b, s=7, c=RED, marker="s", linewidths=0, label=label_second)
    setup_axis(ax, first.shape[1], first.shape[0], xlabel, ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, frameon=False, handletextpad=0.4)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTDIR / name, bbox_inches="tight")
    plt.close(fig)


def make_fd_figure(n: int = 33) -> None:
    d1, d2 = explicit_centered_mats(n)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    plot_single(
        axes[0],
        d1,
        r"(a) Centered first-derivative map $D_x$",
        BLUE,
        "input node index",
        "output node index",
    )
    plot_single(
        axes[1],
        d2,
        r"(b) Centered second-derivative map $D_{xx}$",
        RED,
        "input node index",
        "output node index",
    )
    save(fig, "fd_sparsity.pdf")


def make_compact_figure(n: int = 33) -> None:
    a, r = compact_first_derivative_mats(n)
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.2), constrained_layout=True)
    plot_overlay(
        ax,
        a,
        r,
        r"(a) Banded compact maps $A$ (blue) and $R$ (red)",
        r"$A$",
        r"$R$",
        "input node index",
        "output node index",
    )
    save(fig, "compact_sparsity.pdf")


def make_galerkin_figure(n: int = 33, degree: int = 3) -> None:
    mass, transport, stiffness = galerkin_spline_mats(n, degree)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    plot_overlay(
        axes[0],
        mass,
        transport,
        r"(a) Banded Galerkin maps $M$ (blue) and $G$ (red)",
        r"$M$",
        r"$G$",
        "basis index",
        "basis index",
    )
    plot_overlay(
        axes[1],
        mass,
        stiffness,
        r"(b) Banded Galerkin maps $M$ (blue) and $K$ (red)",
        r"$M$",
        r"$K$",
        "basis index",
        "basis index",
    )
    save(fig, "galerkin_sparsity.pdf")


def make_bspline_figure(n: int = 33, degree: int = 3) -> None:
    bmat, dbmat, ddbmat = collocation_spline_mats(n, degree)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    plot_overlay(
        axes[0],
        bmat,
        dbmat,
        r"(a) Banded nodal maps $B$ (blue) and $B'$ (red)",
        r"$B$",
        r"$B'$",
        "basis index",
        "node index",
    )
    plot_overlay(
        axes[1],
        bmat,
        ddbmat,
        r"(b) Banded nodal maps $B$ (blue) and $B''$ (red)",
        r"$B$",
        r"$B''$",
        "basis index",
        "node index",
    )
    save(fig, "bspline_sparsity.pdf")


def make_implicit_figure(n: int = 33, degree: int = 3) -> None:
    a_fd, a_compact, a_galerkin = implicit_line_mats(n, degree)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    plot_single(
        axes[0],
        a_fd,
        r"(a) Backward Euler factor $I-\mu D_{xx}$",
        BLUE,
        "input node index",
        "output node index",
    )
    plot_single(
        axes[1],
        a_compact,
        r"(b) Compact line factor $A-\mu R$",
        RED,
        "input node index",
        "output node index",
    )
    plot_single(
        axes[2],
        a_galerkin,
        r"(c) Galerkin line factor $M+\mu K$",
        BLUE,
        "basis index",
        "basis index",
    )
    save(fig, "implicit_sparsity.pdf")


def main() -> None:
    make_fd_figure()
    make_compact_figure()
    make_galerkin_figure()
    make_bspline_figure()
    make_implicit_figure()
    print(f"Wrote figures to {OUTDIR}")


if __name__ == "__main__":
    main()
