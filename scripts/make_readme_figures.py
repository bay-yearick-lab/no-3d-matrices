"""Build README assets for no-3d-matrices from the paper's Table 2 data
and the repo's own 1D operator constructors (scripts/make_sparsity_figures.py)."""
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(sys.argv[1])          # source repo root
OUT = Path(sys.argv[2])          # assets dir
STYLE = str(Path(__file__).resolve().parent / "lab.mplstyle")

INK, TEAL, AMBER, SLATE, CORAL, GRAY = "#1B2A41", "#0F9D8A", "#F2A541", "#3D5A80", "#E07A5F", "#8D99AE"

# ---- Paper Table 2 (tab:poisson3d), verbatim -------------------------------
N = np.array([16, 32, 48, 64, 128, 256])
DOF = N**3
FACTOR_MB = np.array([14.8, 385, 2799])            # assembled LU fill-in, N=16,32,48
T_ASM = np.array([0.08, 3.45, 55.4])
OP_KB = np.array([0.38, 0.77, 1.15, 1.54, 3.07, 6.14])  # matrix-free 1D eigenvalues
T_MF = np.array([0.0001, 0.0006, 0.0022, 0.0044, 0.073, 1.98])
ERR = np.array([2.8e-3, 7.5e-4, 3.4e-4, 2.0e-4, 4.9e-5, 1.2e-5])


def fmt_dof(n):
    return f"{n}³"


# ---------------------------------------------------------------------------
def hero():
    plt.style.use(STYLE)
    W, H = 1600, 560
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    fig.patch.set_facecolor("white")

    # right: 3D grid with highlighted pencils
    ax = fig.add_axes([0.60, -0.04, 0.42, 1.08], projection="3d")
    ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-58)
    n = 7
    g = np.arange(n)
    # faint lattice points
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=2.2, c="#C9D1DC", depthshade=False, linewidths=0)
    # cube edges
    for a in (0, n - 1):
        for b in (0, n - 1):
            ax.plot([0, n - 1], [a, a], [b, b], color="#AEB7C4", lw=0.7)
            ax.plot([a, a], [0, n - 1], [b, b], color="#AEB7C4", lw=0.7)
            ax.plot([a, a], [b, b], [0, n - 1], color="#AEB7C4", lw=0.7)
    # pencils: x-lines (teal), y-lines (amber), z-lines (slate)
    for j, k in [(1, 5), (3, 5), (5, 5), (1, 3)]:
        ax.plot([0, n - 1], [j, j], [k, k], color=TEAL, lw=3.0, solid_capstyle="round")
        ax.scatter(g, np.full(n, j), np.full(n, k), s=10, c=TEAL, depthshade=False)
    for i, k in [(6, 1), (6, 3)]:
        ax.plot([i, i], [0, n - 1], [k, k], color=AMBER, lw=3.0, solid_capstyle="round")
        ax.scatter(np.full(n, i), g, np.full(n, k), s=10, c=AMBER, depthshade=False)
    for i, j in [(2, 0), (4, 0)]:
        ax.plot([i, i], [j, j], [0, n - 1], color=SLATE, lw=3.0, solid_capstyle="round")
        ax.scatter(np.full(n, i), np.full(n, j), g, s=10, c=SLATE, depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    lim = (-0.3, n - 0.7)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)

    # left: text
    tx = fig.add_axes([0, 0, 1, 1]); tx.axis("off"); tx.patch.set_alpha(0)
    tx.set_xlim(0, 1); tx.set_ylim(0, 1)
    tx.add_patch(plt.Rectangle((0.035, 0.18), 0.006, 0.64, color=TEAL, transform=tx.transAxes))
    tx.text(0.06, 0.76, "No 3D Matrices", fontsize=26, weight="bold", color=INK, va="center")
    tx.text(0.06, 0.645, "Matrix-free Cartesian PDE solvers as batched 1D line kernels",
            fontsize=10.5, color=SLATE, va="center")
    tx.text(0.06, 0.50, "Finite difference, compact, Galerkin / isogeometric,", fontsize=11.5, color=INK, va="center")
    tx.text(0.06, 0.43, "ADI and fast Poisson solvers, written as independent", fontsize=11.5, color=INK, va="center")
    tx.text(0.06, 0.36, "batched 1D banded sweeps along x, y and z", fontsize=11.5, color=INK, va="center")
    tx.text(0.06, 0.22, "O(N) work  ·  O(Nx+Ny+Nz) operator storage  ·  no assembled 3D operator",
            fontsize=8.2, color=GRAY, va="center")

    tx.text(0.62, 0.07, "x-sweep", color=TEAL, fontsize=8.5, weight="bold")
    tx.text(0.72, 0.07, "y-sweep", color=AMBER, fontsize=8.5, weight="bold")
    tx.text(0.82, 0.07, "z-sweep", color=SLATE, fontsize=8.5, weight="bold")
    tx.text(0.62, 0.025, "each sweep = one batched 1D banded kernel", color=GRAY, fontsize=7.5)

    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(OUT / "hero.png", dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
def memory_card():
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(N))
    w = 0.38
    asm_bytes = FACTOR_MB * 1e6
    mf_bytes = OP_KB * 1e3
    ax.bar(x[:3] - w / 2, asm_bytes, w, color=AMBER, label="assembled sparse factor (LU fill-in)")
    ax.bar(np.where(x < 3, x + w / 2, x), mf_bytes, w, color=TEAL,
           label="matrix-free 1D operators")
    ax.text(4, 2e5, "assembled factor: out of memory\npast 48³ on a workstation", ha="center", va="center",
            fontsize=9.5, color=AMBER, weight="bold")
    ratios = asm_bytes / mf_bytes[:3]
    labels = ["~39,000×", "~500,000×", "~2,400,000×"]
    for i, lab in enumerate(labels):
        ax.text(x[i], asm_bytes[i] * 2.2, lab, ha="center", fontsize=10, weight="bold", color=INK)
    for i in range(6):
        ax.text(x[i] + (w / 2 if i < 3 else 0), mf_bytes[i] * 1.6, f"{OP_KB[i]:g} KB",
                ha="center", fontsize=8.5, color=TEAL)
    ax.set_yscale("log")
    ax.set_ylim(1e2, 3e10)
    ax.set_xticks(x, [f"{n}³\n{d:,}" for n, d in zip(N, DOF)], fontsize=9.5)
    ax.set_xlabel("grid  /  unknowns")
    ax.set_ylabel("operator storage (bytes)")
    ax.set_yticks([1e3, 1e6, 1e9], ["1 KB", "1 MB", "1 GB"])
    ax.set_title(">2,000,000× less operator storage at 48³", loc="left")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.07, "3D Poisson, 7-point stencil. Data: paper Table 2 (scripts/poisson3d_benchmark.py).",
             fontsize=8, color=GRAY)
    fig.savefig(OUT / "memory_footprint.png")
    plt.close(fig)
    return ratios


def time_card():
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.loglog(DOF[:3], T_ASM, "o-", color=AMBER, lw=2.2, ms=7, label="assembled sparse direct (SuperLU)")
    ax.loglog(DOF, T_MF, "s-", color=TEAL, lw=2.2, ms=7, label="matrix-free fast diagonalization")
    ax.annotate("factor exhausts\nworkstation memory", xy=(DOF[2], T_ASM[2]), xytext=(DOF[2] * 2.2, T_ASM[2] * 1.1),
                fontsize=9, color=AMBER, va="center",
                arrowprops=dict(arrowstyle="-", color=AMBER, lw=1))
    ax.annotate("~25,000× faster\nat 48³", xy=(DOF[2], np.sqrt(T_ASM[2] * T_MF[2])), xytext=(DOF[2] * 0.30, 0.03),
                fontsize=9.5, weight="bold", color=INK, ha="center")
    ax.vlines(DOF[2], T_MF[2], T_ASM[2], color=GRAY, lw=1, ls=":")
    ax.annotate("256³ ≈ 16.8M unknowns\n1.98 s", xy=(DOF[-1], T_MF[-1]), xytext=(DOF[-1] * 0.06, 8),
                fontsize=9.5, weight="bold", color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=1))
    ax.set_xlabel("unknowns  N = Nx·Ny·Nz")
    ax.set_ylabel("time to solution (s)")
    ax.set_ylim(5e-5, 300)
    ax.set_title("16.8M-unknown 3D Poisson solve in 1.98 s", loc="left")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.text(0.01, -0.05, "Same discrete system both ways; solutions agree to ~1e-14. SciPy, single workstation. Data: paper Table 2.",
             fontsize=8, color=GRAY)
    fig.savefig(OUT / "time_to_solution.png")
    plt.close(fig)


def scaling_card():
    plt.style.use(STYLE)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(wspace=0.32))
    field = 8.0 * DOF  # one double-precision field, reference
    ax.loglog(DOF[:3], FACTOR_MB * 1e6, "o-", color=AMBER, lw=2.2, ms=7, label="assembled factor (measured)")
    ax.loglog(DOF, field, "--", color=GRAY, lw=1.6, label="one solution field, 8·N bytes (reference)")
    ax.loglog(DOF, OP_KB * 1e3, "s-", color=TEAL, lw=2.2, ms=7, label="matrix-free 1D operators (measured)")
    ax.text(DOF[3], OP_KB[3] * 1e3 * 3.2, "O(Nx+Ny+Nz)", color=TEAL, fontsize=10, weight="bold")
    ax.text(DOF[3], field[3] * 3.0, "O(N)", color=GRAY, fontsize=10, weight="bold")
    ax.set_yticks([1e3, 1e6, 1e9], ["1 KB", "1 MB", "1 GB"])
    ax.set_xlabel("unknowns N")
    ax.set_ylabel("storage (bytes)")
    ax.set_title("Operators shrink below the data", loc="left", fontsize=13)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(1e2, 1e12)

    h = 1.0 / (N + 1)
    ax2.loglog(N, ERR, "s-", color=TEAL, lw=2.2, ms=7, label="max error (both routes, identical)")
    ref = ERR[0] * (h / h[0]) ** 2
    ax2.loglog(N, ref, "--", color=GRAY, lw=1.4, label="O(h²) reference")
    ax2.set_xticks(N, [str(n) for n in N])
    ax2.minorticks_off()
    ax2.set_xlabel("points per direction")
    ax2.set_ylabel(r"$\|u_h-u\|_\infty$")
    ax2.set_title("No accuracy cost: 2nd order to 256³", loc="left", fontsize=13)
    ax2.legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Storage and accuracy scaling", x=0.07, ha="left", fontsize=15, weight="bold", y=1.04)
    fig.text(0.01, -0.04, "Measured points: paper Table 2. Field-storage line and O(h²) line are analytic references, not measurements.",
             fontsize=8, color=GRAY)
    fig.savefig(OUT / "scaling.png")
    plt.close(fig)


def sparsity_panel():
    spec = importlib.util.spec_from_file_location("msf", SRC / "scripts" / "make_sparsity_figures.py")
    msf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msf)
    mpl.rcdefaults()
    plt.style.use(STYLE)
    n, p = 33, 3
    _, d2 = msf.explicit_centered_mats(n)
    a, r = msf.compact_first_derivative_mats(n)
    mass, _, stiff = msf.galerkin_spline_mats(n, p)
    bmat, _, ddb = msf.collocation_spline_mats(n, p)
    _, _, a_gal = msf.implicit_line_mats(n, p)
    panels = [
        ("Finite difference", [(d2, r"$D_{xx}$", TEAL, 14)]),
        ("Compact Padé", [(r, r"$R$", AMBER, 14), (a, r"$A$", TEAL, 7)]),
        ("Spline Galerkin (p = 3)", [(stiff, r"$K$", AMBER, 14), (mass, r"$M$", TEAL, 7)]),
        ("B-spline collocation", [(ddb, r"$B''$", AMBER, 14), (bmat, r"$B$", TEAL, 7)]),
        ("ADI / implicit line", [(a_gal, r"$M+\mu K$", SLATE, 14)]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9))
    for ax, (title, mats) in zip(axes, panels):
        for m, lab, c, s in mats:
            rows, cols = np.where(np.abs(m) > 1e-12)
            ax.scatter(cols, rows, s=s, c=c, marker="s", linewidths=0, label=lab)
        ax.set_xlim(-0.8, n - 0.2); ax.set_ylim(n - 0.2, -0.8)
        ax.set_aspect("equal")
        ax.set_xticks([0, 16, 32]); ax.set_yticks([0, 16, 32])
        ax.tick_params(labelsize=9)
        ax.grid(False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color("#E6E9EF")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower left", fontsize=10, handletextpad=0.2, markerscale=1.4)
    fig.suptitle("Every 1D factor is banded (N = 33)",
                 x=0.125, ha="left", fontsize=14, weight="bold", y=0.93)
    fig.savefig(OUT / "sparsity_patterns.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hero()
    print("ratios", memory_card())
    time_card()
    scaling_card()
    sparsity_panel()
