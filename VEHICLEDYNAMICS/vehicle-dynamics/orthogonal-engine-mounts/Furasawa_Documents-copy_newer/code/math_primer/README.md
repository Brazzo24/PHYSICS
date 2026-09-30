# Math primer

Seven short notebooks that build the mathematics behind the orthogonal engine
mount work from scratch, on systems small enough to check by hand. Read them in
order; each one is 10–20 minutes and ends where the next one starts.

The spine of the series is a deliberate choice: notebooks 1–3 use a two-mass
spring chain, notebook 4 switches to a **rigid bar on two springs**, and
notebooks 5–7 go to three dimensions. The bar is not an arbitrary example — its
2×2 mass and stiffness matrices have exactly the same structure as the paper's
roll–yaw pair, so the silencing condition you derive for it *is* Furusawa's
Eq. (16), just with different letters.

| # | notebook | what it settles |
|---|---|---|
| 1 | `01_matrices_from_newton.ipynb` | where $[M]$ and $[K]$ come from; what an off-diagonal entry means; the assembly rule $k\,bb^\mathsf{T}$; congruence transforms |
| 2 | `02_eigenvalues_and_modes.ipynb` | why the eigenproblem is $([K]-\omega^2[M])u=0$; what a mode shape is; arbitrary scale; what breaks when two frequencies coincide |
| 3 | `03_orthogonality.ipynb` | why "orthogonal" means $M$-orthogonal and not perpendicular; the three-line proof; decoupling into independent oscillators; what modal mass is |
| 4 | `04_participation_and_silent_modes.ipynb` | **the pivotal one** — modal participation, a mode that is genuinely invisible, and the inverse problem the paper actually solves |
| 5 | `05_rigid_body_and_inertia.ipynb` | the inertia tensor from point masses; principal axes as eigenvectors; tensor rotation; where $I_{zx}$ comes from (paper Eq. 20) |
| 6 | `06_mounts_and_stiffness.ipynb` | one mount → a $6\times6$ $[K]$; skew matrices; the elastic centre; why narrow lateral spacing collapses roll stiffness |
| 7 | `07_the_papers_2x2.ipynb` | all of it assembled into Eqs. (16), (22), (25), (27), (29), symbolically, with a symbol map to the paper and the code |

## Running them

```bash
pip install numpy scipy matplotlib sympy jupyter
jupyter lab 01_matrices_from_newton.ipynb
```

Notebooks 5–7 use SymPy for the symbolic steps; 6 and 7 also import
`orthomount` from the parent directory to cross-check the hand-built matrices
against the library.

`primer.py` holds only the shared plot style and the toy spring-mass drawings —
no physics. `build_primers.py` regenerates all seven notebooks from source; edit
that rather than the `.ipynb` files if you want to change them.

## The one-line version

> A mode responds only to $\{u^r\}^\mathsf{T}\{F\}$. Notebook 4 shows you can
> drive that to exactly zero by tuning the ratio of inertial coupling to elastic
> coupling. Notebook 7 shows that is what Furusawa did.

---

**Then:** `../orthogonal_engine_mounts.ipynb` — the paper itself, applied.
