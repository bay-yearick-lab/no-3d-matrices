# No 3D Matrices

Source and reproducibility code for the paper

**No 3D Matrices: A Unified Tensor-Product View of Matrix-Free Cartesian PDE Solvers**
Yong Yi Bay and Kathleen A. Yearick

> Preprint: [arXiv:2606.25148](https://arxiv.org/abs/2606.25148)

Every Cartesian tensor-product PDE discretization — finite differences, compact
(Padé) schemes, Galerkin, B-spline / isogeometric, collocation, ADI time
stepping, and fast Poisson/Helmholtz solvers — factors along the coordinate axes
into one-dimensional banded line operations. The three-dimensional operator is
never assembled, factored, or stored. The paper derives that reduction with
Kronecker-product algebra, marks the boundary of where it is exact, and shows how
production codes turn it into hardware-efficient kernels.

## Cite

```bibtex
@article{bay2026no3d,
  title         = {No 3D Matrices: A Unified Tensor-Product View of Matrix-Free Cartesian PDE Solvers},
  author        = {Bay, Yong Yi and Yearick, Kathleen A.},
  year          = {2026},
  eprint        = {2606.25148},
  archivePrefix = {arXiv},
  primaryClass  = {math.NA}
}
```

## Build the paper

```bash
./build_paper.sh
```

Requires a TeX distribution with `latexmk`, `pdflatex`, and `bibtex` (`bibtex8`
is used when available). The output is `paper/main.pdf`.

## Reproduce the figures and the experiment

The scripts declare their own dependencies and run with
[uv](https://docs.astral.sh/uv/):

```bash
uv run scripts/make_sparsity_figures.py   # operator sparsity figures
uv run scripts/poisson3d_benchmark.py     # 3D Poisson benchmark (Table 2 + Figure 13)
```

Both write their PDFs into `paper/figures/`.

## Layout

```
paper/           LaTeX source, bibliography, and figures (self-contained arXiv submission)
scripts/         Python scripts that regenerate the figures and the benchmark
build_paper.sh   Reproducible build with latexmk
```

## License

The source code (`scripts/`, `build_paper.sh`) is released under the MIT License
(`LICENSE`). The paper text and figures (`paper/`) are licensed under CC BY 4.0
(`paper/LICENSE`).
