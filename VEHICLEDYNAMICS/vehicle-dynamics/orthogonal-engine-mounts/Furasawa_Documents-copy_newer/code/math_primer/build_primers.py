"""Generates the seven math_primer notebooks."""
import nbformat as nbf

NBS = {}
_cur = None


def start(name):
    global _cur
    _cur = []
    NBS[name] = _cur


def md(s):
    _cur.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    _cur.append(nbf.v4.new_code_cell(s.strip("\n")))


HEAD = '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import primer
from primer import SERIES, INK, INK2, GRID, tidy
primer.use_style()
'''

# ###########################################################################
# 01
# ###########################################################################
start("01_matrices_from_newton")

md(r"""
# 1 · Where the matrices come from

**Math primer for the orthogonal engine mount work — notebook 1 of 7.**

Before any of the paper's cleverness, there is a boring but load-bearing step:
turning "two masses joined by springs" into $[M]\{\ddot q\} + [K]\{q\} = \{f\}$.
If the matrix assembly feels like magic, everything built on it will too.

This notebook does it entirely by hand, on a system small enough to check on
paper, and then shows the *assembly rule* that scales to the six-degree-of-freedom
engine.

**You should come out knowing:** what each entry of $[K]$ physically means, why
off-diagonal entries are called *coupling*, and how a stiffness matrix is built
by summing one contribution per spring.
""")

code(HEAD)

md(r"""
## 1.1 · The system

Wall — spring $k_1$ — mass $m_1$ — spring $k_2$ — mass $m_2$.

Let $q_1, q_2$ be the displacements of the two masses from their rest positions.
""")

code(r'''
fig, ax = plt.subplots(figsize=(5.4, 1.5))
primer.chain_2dof(ax, u=(0, 0))
ax.text(0.5, 0.30, "$k_1$", ha="center", color=INK2, fontsize=10)
ax.text(1.6, 0.30, "$k_2$", ha="center", color=INK2, fontsize=10)
ax.text(1.0, -0.42, "$q_1$", ha="center", color=SERIES[0], fontsize=10)
ax.text(2.2, -0.42, "$q_2$", ha="center", color=SERIES[1], fontsize=10)
plt.tight_layout(); plt.show()
''')

md(r"""
## 1.2 · Newton, one mass at a time

Spring $k_2$ is stretched by $q_2 - q_1$. It therefore pulls $m_1$ forward with
$k_2(q_2-q_1)$ and pulls $m_2$ backward with the same magnitude. Spring $k_1$ is
stretched by $q_1$ and pulls $m_1$ back with $k_1 q_1$.

$$m_1\ddot q_1 = -k_1 q_1 + k_2(q_2-q_1) + f_1$$
$$m_2\ddot q_2 = -k_2(q_2-q_1) + f_2$$

Move everything that is not an inertia or force term to the left:

$$m_1\ddot q_1 + (k_1+k_2)q_1 - k_2 q_2 = f_1$$
$$m_2\ddot q_2 - k_2 q_1 + k_2 q_2 = f_2$$

Which is exactly

$$\begin{bmatrix}m_1&0\\0&m_2\end{bmatrix}
\begin{Bmatrix}\ddot q_1\\ \ddot q_2\end{Bmatrix} +
\begin{bmatrix}k_1+k_2&-k_2\\-k_2&k_2\end{bmatrix}
\begin{Bmatrix}q_1\\ q_2\end{Bmatrix} =
\begin{Bmatrix}f_1\\ f_2\end{Bmatrix}$$

Nothing has happened except bookkeeping. **That is the whole content of "writing
it in matrix form".**
""")

code(r'''
m1, m2 = 2.0, 1.0
k1, k2 = 800.0, 400.0

M = np.array([[m1, 0.0],
              [0.0, m2]])
K = np.array([[k1 + k2, -k2],
              [-k2,      k2]])
print("M =\n", M)
print("\nK =\n", K)
''')

md(r"""
## 1.3 · What the entries mean

Read column $j$ of $[K]$ as: *hold every coordinate at zero except $q_j = 1$, and
report the forces you must apply at each coordinate to hold it there.*

Push $m_1$ to $q_1 = 1$ while pinning $m_2$ at zero: you must fight both springs,
so you need $k_1+k_2$ at coordinate 1 — and you must also *hold* $m_2$ back with
$-k_2$, because spring $k_2$ is now pushing it. That is column 1.

Two consequences worth internalising:

* $K_{ij}$ is the force at $i$ caused by a unit displacement at $j$. Because the
  springs are conservative, $K_{ij}=K_{ji}$ — **the stiffness matrix is
  symmetric**, always.
* $K_{12}\ne 0$ says coordinate 1 and coordinate 2 are **coupled**: you cannot
  move one without a force appearing at the other. Coupling is not a property of
  the physics alone — it is a property of the physics *and the coordinates you
  chose*. Notebook 4 leans on that hard.
""")

code(r'''
# Column j of K, obtained the way the definition says: unit displacement, hold the rest.
for j in range(2):
    q = np.zeros(2); q[j] = 1.0
    print(f"unit displacement at coordinate {j+1}  ->  forces needed: {K @ q}")

print("\nsymmetric?", np.allclose(K, K.T))
''')

md(r"""
## 1.4 · The assembly rule (this is the part that scales)

Building $[K]$ by writing Newton's law for every mass works for two masses and
becomes hopeless for six degrees of freedom with four mounts. The scalable
version: **each spring contributes its own matrix, and you add them up.**

A spring of rate $k$ connecting coordinates $i$ and $j$ stores energy
$\tfrac12 k (q_j - q_i)^2$. Write $q_j - q_i = \{b\}^\mathsf{T}\{q\}$ with
$\{b\}$ a vector of $-1$ and $+1$ in the right slots. Then the energy is
$\tfrac12 k (\{b\}^\mathsf{T}\{q\})^2$ and the contribution to $[K]$ is

$$[K]_{\text{spring}} = k\,\{b\}\{b\}^\mathsf{T}$$

An outer product — a rank-1 matrix. This is the *entire* idea behind assembling
the engine's $6\times6$ stiffness matrix later: each mount contributes
$k\,\{b\}\{b\}^\mathsf{T}$ for each of its three directions, where $\{b\}$ says
how that mount's deflection depends on the six rigid-body coordinates.
""")

code(r'''
def spring_K(k, b, n=2):
    """Stiffness contribution of one spring, k * b b^T."""
    b = np.asarray(b, dtype=float).reshape(n, 1)
    return k * (b @ b.T)

# spring 1 connects ground to coordinate 1  ->  deflection = q1        -> b = [1, 0]
# spring 2 connects coordinate 1 to 2       ->  deflection = q2 - q1   -> b = [-1, 1]
K_assembled = spring_K(k1, [1, 0]) + spring_K(k2, [-1, 1])

print("assembled:\n", K_assembled)
print("\nsame as the hand-derived K?", np.allclose(K, K_assembled))
''')

md(r"""
### Why the outer product is always symmetric and positive semi-definite

$\{b\}\{b\}^\mathsf{T}$ is symmetric by construction, and
$\{q\}^\mathsf{T}\{b\}\{b\}^\mathsf{T}\{q\} = (\{b\}^\mathsf{T}\{q\})^2 \ge 0$.
So a sum of springs can never produce a negative stiffness, and $[K]$ is singular
exactly when there is a rigid-body motion no spring resists. Useful sanity check
on a real model: if `eigvalsh(K)` has a near-zero entry, some direction is
unrestrained.
""")

code(r'''
print("eigenvalues of K:", np.linalg.eigvalsh(K))

# drop the ground spring: now the pair can drift together, unrestrained
K_free = spring_K(k2, [-1, 1])
print("without the ground spring:", np.linalg.eigvalsh(K_free))
print("  -> a zero: the rigid-body drift mode", np.linalg.eigh(K_free)[1][:, 0])
''')

md(r"""
## 1.5 · Changing coordinates

Coordinates are a choice. Suppose instead of $(q_1, q_2)$ you use
$(q_1,\ \Delta)$ where $\Delta = q_2 - q_1$ is the spring-2 stretch. The old
coordinates in terms of the new are $\{q\} = [T]\{p\}$ with

$$[T]=\begin{bmatrix}1&0\\1&1\end{bmatrix}$$

Substituting into the equations and pre-multiplying by $[T]^\mathsf{T}$ (which
keeps the energy interpretation intact) gives

$$[T]^\mathsf{T}[M][T]\{\ddot p\} + [T]^\mathsf{T}[K][T]\{p\} = [T]^\mathsf{T}\{f\}$$

This **congruence transform**, $A \mapsto T^\mathsf{T} A T$, is the single most
important operation in the rest of the primer. Notice what happened: $[K]$ became
diagonal — the elastic coupling vanished — but $[M]$ picked up off-diagonal terms.
The coupling did not disappear; it moved from the stiffness matrix into the mass
matrix.
""")

code(r'''
T = np.array([[1.0, 0.0],
              [1.0, 1.0]])
Mp = T.T @ M @ T
Kp = T.T @ K @ T
print("M' =\n", Mp)
print("\nK' =\n", Kp)
print("\nelastic coupling K'_12 =", Kp[0, 1], "   inertial coupling M'_12 =", Mp[0, 1])
''')

md(r"""
That observation is worth sitting with, because the paper's entire trick lives
here. **Coupling is not intrinsic — it is a property of your coordinate choice**,
and there are two kinds of it (through $[M]$ and through $[K]$) that can be traded
against each other. Furusawa does not remove coupling. He arranges for the two
kinds to stand in a specific ratio.

Notebook 4 makes that concrete; notebook 7 is the paper's own version of it.

---

**Next:** [2 · Eigenvalues and mode shapes](02_eigenvalues_and_modes.ipynb)
""")

# ###########################################################################
# 02
# ###########################################################################
start("02_eigenvalues_and_modes")

md(r"""
# 2 · Eigenvalues and mode shapes

**Math primer — notebook 2 of 7.**

We have $[M]\{\ddot q\}+[K]\{q\}=\{0\}$. Now: what does the system do when you let
go of it?

**You should come out knowing:** what a mode shape *is* physically, why the
eigenproblem is $\left([K]-\omega^2[M]\right)\{u\}=0$ rather than the plain
$[A]\{u\}=\lambda\{u\}$, that mode shapes have arbitrary scale, and what goes
wrong when two natural frequencies coincide.
""")

code(HEAD)

md(r"""
## 2.1 · Guess a solution

Try the guess that every coordinate oscillates at the *same* frequency and in
*fixed proportion* — that is, the shape does not change, only its amplitude:

$$\{q(t)\} = \{u\}\cos\omega t$$

Then $\{\ddot q\} = -\omega^2\{u\}\cos\omega t$, and substituting:

$$\left(-\omega^2[M]+[K]\right)\{u\}\cos\omega t = \{0\}$$

which must hold at every instant, so

$$\boxed{\left([K]-\omega^2[M]\right)\{u\}=\{0\}}$$

This is the paper's Eq. (3). It is a **generalised** eigenproblem: the $[M]$ sits
where the identity matrix sits in the textbook $[A]\{u\}=\lambda\{u\}$. That is
not cosmetic — it is why the orthogonality in notebook 3 is with respect to $[M]$
rather than the plain dot product.

A non-zero $\{u\}$ exists only if the matrix is singular:

$$\det\left([K]-\omega^2[M]\right)=0$$

which is a polynomial in $\omega^2$ of degree $n$. Its $n$ roots are the natural
frequencies.
""")

code(r'''
m1, m2 = 2.0, 1.0
k1, k2 = 800.0, 400.0
M = np.diag([m1, m2])
K = np.array([[k1 + k2, -k2], [-k2, k2]])

# --- the determinant route, done explicitly -------------------------------
# det(K - s M) = (k1+k2 - s m1)(k2 - s m2) - k2^2  = a s^2 + b s + c,  s = w^2
a = m1 * m2
b = -((k1 + k2) * m2 + k2 * m1)
c = (k1 + k2) * k2 - k2 ** 2
roots = np.roots([a, b, c])
print("roots of the characteristic polynomial, s = w^2:", np.sort(roots))
print("natural frequencies [rad/s]:", np.sqrt(np.sort(roots)))
print("natural frequencies [Hz]   :", np.sqrt(np.sort(roots)) / (2 * np.pi))
''')

code(r'''
# --- and the way you will actually do it ----------------------------------
w2, V = eigh(K, M)          # solves K u = w^2 M u
omega = np.sqrt(w2)
print("eigh gives w^2 =", w2)
print("agrees with the polynomial?", np.allclose(np.sort(w2), np.sort(roots)))
print("\nmode shapes (columns):\n", V)
''')

md(r"""
## 2.2 · Reading a mode shape

`eigh` returns the modes mass-normalised ($\{u\}^\mathsf{T}[M]\{u\}=1$), which is
convenient but hides the physical picture. Rescale so the largest component is
$+1$ and the shapes become readable:
""")

code(r'''
Vs = V / V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])]
for r in range(2):
    print(f"mode {r+1}:  f = {omega[r]/2/np.pi:6.2f} Hz   shape = {Vs[:, r]}")

fig, axes = plt.subplots(1, 2, figsize=(8.6, 1.9))
for r, ax in enumerate(axes):
    primer.strobe(ax, Vs[:, r], amp=0.30,
                  title=f"mode {r+1} — {omega[r]/2/np.pi:.2f} Hz")
plt.tight_layout(); plt.show()
''')

md(r"""
The faded dots are the masses at several phases of one cycle; the solid dots are
the extreme. Mode 1: both masses move the same way, the light one further. Mode
2: they move in opposite directions.

That is all a mode shape is — **the fixed ratio in which the coordinates move
when the system is left to vibrate at one of its natural frequencies.**

### Scale is arbitrary
""")

code(r'''
u = V[:, 0]
for s in (1.0, -3.7, 0.02):
    resid = (K - w2[0] * M) @ (s * u)
    print(f"scaling by {s:6.2f}: residual norm = {np.linalg.norm(resid):.2e}")
print("\nAny multiple of a mode shape is still a mode shape. Only the RATIO is physical.")
print("Which is why every convention (unit largest entry, unit norm, mass-normalised)")
print("is equally valid -- and why you must always say which one you used.")
''')

md(r"""
## 2.3 · Watching the modes appear

A useful intuition: sweep a parameter and watch the frequencies and the shapes
move. Here we soften the coupling spring $k_2$ toward zero.
""")

code(r'''
k2s = np.linspace(20, 2000, 300)
fs = np.zeros((k2s.size, 2))
ratio = np.zeros((k2s.size, 2))
for i, kk in enumerate(k2s):
    Ki = np.array([[k1 + kk, -kk], [-kk, kk]])
    wi, Vi = eigh(Ki, M)
    fs[i] = np.sqrt(wi) / (2 * np.pi)
    ratio[i] = Vi[1] / Vi[0]          # q2/q1 for each mode

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
axes[0].plot(k2s, fs[:, 0], color=SERIES[0], label="mode 1")
axes[0].plot(k2s, fs[:, 1], color=SERIES[1], label="mode 2")
tidy(axes[0], "2.3 · natural frequencies as $k_2$ varies", "$k_2$ [N/m]", "frequency [Hz]")
axes[0].legend()

axes[1].plot(k2s, ratio[:, 0], color=SERIES[0], label="mode 1")
axes[1].plot(k2s, ratio[:, 1], color=SERIES[1], label="mode 2")
axes[1].axhline(0, color=INK2, lw=0.9, ls="--")
axes[1].set_ylim(-6, 6)
tidy(axes[1], "mode shape ratio $q_2/q_1$", "$k_2$ [N/m]", "$q_2/q_1$")
axes[1].legend()
plt.tight_layout(); plt.show()
''')

md(r"""
Mode 1 always has $q_2/q_1 > 0$ (in phase), mode 2 always $< 0$ (out of phase),
and the two frequency curves never touch. That last point matters.

## 2.4 · When two frequencies coincide

If $\omega_1 = \omega_2$, the matrix $[K]-\omega^2[M]$ drops rank by two, and
**any** vector in a two-dimensional subspace is a valid mode shape. The mode
shapes are no longer determined by the system — only the subspace is.

This is not a curiosity. The paper spends Eqs. (17)–(19) proving that its design
does *not* land in this case, because if it did, "make mode 2 orthogonal to the
excitation" would be a statement about a shape that is not uniquely defined.
""")

code(r'''
# force a degenerate system: two identical uncoupled oscillators
Md = np.diag([1.0, 1.0])
Kd = np.diag([500.0, 500.0])
wd, Vd = eigh(Kd, Md)
print("frequencies [Hz]:", np.sqrt(wd) / 2 / np.pi)
print("shapes returned by eigh:\n", Vd)

# but an arbitrary rotation of those two vectors works just as well
th = 0.7
Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Valt = Vd @ Rot
print("\na rotated pair, equally valid:\n", Valt)
for r in range(2):
    print(f"  residual for rotated mode {r+1}: "
          f"{np.linalg.norm((Kd - wd[r]*Md) @ Valt[:, r]):.2e}")
print("\nThe subspace is determined; the individual shapes are not.")
''')

md(r"""
**Practical rule:** whenever you write code that depends on a particular mode
shape — as the paper's design does — check that the corresponding eigenvalue is
well separated from its neighbours. In `orthomount` that check is the assertion
that $K_{\phi\phi}/K_{\psi\psi} \ne I_x/I_z$, which is exactly the paper's
Eq. (19) argument.

---

**Next:** [3 · Orthogonality and modal coordinates](03_orthogonality.ipynb)
""")

# ###########################################################################
# 03
# ###########################################################################
start("03_orthogonality")

md(r"""
# 3 · Orthogonality — and why it is not the dot product

**Math primer — notebook 3 of 7.**

The word "orthogonal" in the paper's title does *not* mean the mode shapes are
perpendicular in the ordinary sense. They usually are not. They are orthogonal
*with respect to the mass matrix*, and that distinction is the reason the whole
method works.

**You should come out knowing:** what $\{u^s\}^\mathsf{T}[M]\{u^r\}=0$ means and
where it comes from, how it turns coupled equations into independent
single-degree-of-freedom oscillators, and what modal mass actually is.
""")

code(HEAD)

code(r'''
m1, m2 = 2.0, 1.0
k1, k2 = 800.0, 400.0
M = np.diag([m1, m2])
K = np.array([[k1 + k2, -k2], [-k2, k2]])
w2, V = eigh(K, M)
V = V / V[np.argmax(np.abs(V), axis=0), np.arange(2)]   # largest entry = 1
omega = np.sqrt(w2)
print("shapes:\n", V)
print("\nplain dot product u1 . u2 =", V[:, 0] @ V[:, 1], "   <- NOT zero")
print("M-weighted   u1^T M u2   =", V[:, 0] @ M @ V[:, 1], "   <- zero")
print("K-weighted   u1^T K u2   =", V[:, 0] @ K @ V[:, 1], "   <- zero")
''')

md(r"""
## 3.1 · Where it comes from

Three lines. Take two modes $r\ne s$:

$$[K]\{u^r\}=\omega_r^2[M]\{u^r\},\qquad [K]\{u^s\}=\omega_s^2[M]\{u^s\}$$

Pre-multiply the first by $\{u^s\}^\mathsf{T}$ and the second by
$\{u^r\}^\mathsf{T}$:

$$\{u^s\}^\mathsf{T}[K]\{u^r\}=\omega_r^2\{u^s\}^\mathsf{T}[M]\{u^r\}$$
$$\{u^r\}^\mathsf{T}[K]\{u^s\}=\omega_s^2\{u^r\}^\mathsf{T}[M]\{u^s\}$$

Because $[M]$ and $[K]$ are symmetric, the left-hand sides are equal and the
$[M]$-products on the right are equal. Subtract:

$$0=\left(\omega_r^2-\omega_s^2\right)\{u^s\}^\mathsf{T}[M]\{u^r\}$$

If the frequencies differ, the $[M]$-product must vanish. And then the
$[K]$-product vanishes too. That is the paper's Eqs. (4) and (5).

Note the two things the proof needed: **symmetry** of $[M]$ and $[K]$ (which
notebook 1 showed is automatic for springs), and **distinct frequencies** (which
notebook 2 showed can fail).
""")

code(r'''
# The full orthogonality tables.
print("V^T M V =\n", V.T @ M @ V)
print("\nV^T K V =\n", V.T @ K @ V)
print("\nOff-diagonals are zero to machine precision; the diagonals are the")
print("MODAL MASS m_r and MODAL STIFFNESS k_r, and k_r / m_r = w_r^2:")
mr = np.diag(V.T @ M @ V)
kr = np.diag(V.T @ K @ V)
print("  m_r =", mr)
print("  k_r =", kr)
print("  k_r/m_r      =", kr / mr)
print("  w_r^2        =", w2)
''')

md(r"""
## 3.2 · Geometric picture

Ordinary orthogonality means perpendicular. $[M]$-orthogonality means
perpendicular *after* the space has been stretched by the mass distribution. Draw
both: the unit circle becomes an ellipse whose axes are set by $[M]$, and the two
mode shapes are conjugate diameters of that ellipse.
""")

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

th = np.linspace(0, 2*np.pi, 400)
circle = np.vstack([np.cos(th), np.sin(th)])

for ax, A, name in ((axes[0], np.eye(2), "plain dot product"),
                    (axes[1], M, "M-weighted inner product")):
    # {x : x^T A x = 1}
    L = np.linalg.cholesky(np.linalg.inv(A))
    ell = L @ circle
    ax.plot(ell[0], ell[1], color=GRID, lw=1.4)
    for r in range(2):
        u = V[:, r] / np.sqrt(V[:, r] @ A @ V[:, r])
        ax.annotate("", xy=(u[0], u[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=SERIES[r], lw=2.4))
        ax.text(u[0]*1.16, u[1]*1.16, f"$u^{r+1}$", color=SERIES[r], fontsize=11,
                ha="center", va="center")
    ang = np.degrees(np.arccos(
        (V[:, 0] @ A @ V[:, 1]) /
        np.sqrt((V[:, 0] @ A @ V[:, 0]) * (V[:, 1] @ A @ V[:, 1]))))
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color=GRID, lw=0.7); ax.axvline(0, color=GRID, lw=0.7)
    tidy(ax, f"3.2 · {name}\nangle between the modes: {ang:.1f}°", "$q_1$", "$q_2$")
plt.tight_layout(); plt.show()
''')

md(r"""
Same two vectors, two different notions of "the angle between them". In the
$[M]$-weighted geometry they are exactly 90° apart. **That** is the orthogonality
the paper is talking about.

## 3.3 · The payoff: decoupling

Collect the modes as columns of $[\Phi]$ and change coordinates with
$\{q\}=[\Phi]\{\eta\}$. Apply the congruence transform from notebook 1:

$$[\Phi]^\mathsf{T}[M][\Phi]\{\ddot\eta\}+[\Phi]^\mathsf{T}[K][\Phi]\{\eta\}
=[\Phi]^\mathsf{T}\{f\}$$

Both transformed matrices are **diagonal** — that is exactly what orthogonality
says. So the coupled system has become $n$ independent scalar equations:

$$m_r\ddot\eta_r + k_r\eta_r = \{u^r\}^\mathsf{T}\{f\},\qquad r=1\ldots n$$

Each is a single mass on a single spring. This is what "modal analysis" buys you,
and it is the machinery behind the paper's Eq. (6).
""")

code(r'''
Phi = V
print("Phi^T M Phi =\n", Phi.T @ M @ Phi)
print("\nPhi^T K Phi =\n", Phi.T @ K @ Phi)
print("\nOff-diagonal magnitudes: "
      f"{np.max(np.abs(Phi.T @ K @ Phi - np.diag(np.diag(Phi.T @ K @ Phi)))):.2e}")

# integrate both forms and compare
def simulate(Mx, Kx, q0, T=0.6, n=3000):
    dt = T / n
    q = np.array(q0, float); v = np.zeros_like(q)
    Minv = np.linalg.inv(Mx)
    out = np.zeros((n, len(q)))
    for i in range(n):
        v += dt * (Minv @ (-Kx @ q))
        q += dt * v
        out[i] = q
    return np.linspace(0, T, n), out

q0 = np.array([0.01, 0.0])
t, q_direct = simulate(M, K, q0)
eta0 = np.linalg.solve(Phi, q0)
_, eta = simulate(Phi.T @ M @ Phi, Phi.T @ K @ Phi, eta0)
q_modal = (Phi @ eta.T).T

fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(t*1000, q_direct[:, 0]*1000, color=SERIES[0], label="$q_1$ direct")
ax.plot(t*1000, q_modal[:, 0]*1000, color=SERIES[3], ls="--", lw=1.6,
        label="$q_1$ via modal coordinates")
tidy(ax, "3.3 · the same motion, computed two ways", "time [ms]", "displacement [mm]")
ax.legend()
plt.tight_layout(); plt.show()
print("max difference:", np.max(np.abs(q_direct - q_modal)))
''')

md(r"""
## 3.4 · Modal mass is a bookkeeping quantity, not a physical one

Because mode shapes have arbitrary scale, so does modal mass — scale the shape by
$c$ and $m_r$ scales by $c^2$. What is *not* arbitrary is any physical answer,
because the scale cancels: in $\{u^r\}\{u^r\}^\mathsf{T}\{F\}/m_r$ the numerator
scales by $c^2$ too.
""")

code(r'''
for c in (1.0, 5.0, 0.1):
    u = c * V[:, 0]
    mr_ = u @ M @ u
    F = np.array([1.0, 0.0])
    contribution = u * (u @ F) / mr_
    print(f"scale c = {c:4.1f}:  modal mass = {mr_:9.4f}   "
          f"contribution to response = {contribution}")
print("\nModal mass changes; the physical contribution does not.")
''')

md(r"""
### Repeated frequencies, again

If two frequencies coincide, `eigh` still hands you an $[M]$-orthogonal pair — it
picks one arbitrarily out of the valid subspace. Fine numerically, dangerous if
your design *means* something by "mode 2".

---

**Next:** [4 · Participation, and how to silence a mode](04_participation_and_silent_modes.ipynb)
""")

# ###########################################################################
# 04
# ###########################################################################
start("04_participation_and_silent_modes")

md(r"""
# 4 · Participation, and how to silence a mode

**Math primer — notebook 4 of 7.**

This is the pivotal notebook. Everything so far has been standard vibration
theory; here is where the paper's idea appears, in a two-degree-of-freedom system
you can draw on a napkin.

**You should come out knowing:** what the modal participation factor is
geometrically, why a mode with zero participation is genuinely invisible, and —
the paper's actual problem — how to *move a mode shape* so that a **fixed**
excitation stops seeing it.
""")

code(HEAD)

md(r"""
## 4.1 · Forcing, in modal coordinates

From notebook 3, mode $r$ obeys $m_r\ddot\eta_r+k_r\eta_r=\{u^r\}^\mathsf{T}\{f\}$.
For harmonic forcing $\{f\}=\{F\}e^{i\omega t}$ the steady state is

$$\eta_r=\frac{\{u^r\}^\mathsf{T}\{F\}}{m_r\left(\omega_r^2-\omega^2\right)},
\qquad\text{so}\qquad
\{Q\}=\sum_r \frac{\{u^r\}\left(\{u^r\}^\mathsf{T}\{F\}\right)}{m_r\left(\omega_r^2-\omega^2\right)}$$

the paper's Eq. (6). The scalar

$$\Gamma_r=\{u^r\}^\mathsf{T}\{F\}$$

is the **modal participation factor**. Geometrically it is the projection of the
force vector onto the mode shape — how much of the push points along the way that
mode wants to move.

If $\Gamma_r=0$, mode $r$ receives no work from the force. Not "a little at
resonance". None, at any frequency.
""")

md(r"""
## 4.2 · A system with both kinds of coupling

The two-mass chain from notebooks 1–3 is too simple for what comes next: it has
elastic coupling but no inertial coupling. So switch to a system that has both —
and which happens to be an exact structural analogue of the paper's problem.

**A rigid bar on two springs.** Mass $m$, moment of inertia $I_G$ about its own
centre of gravity, springs $k_f$ and $k_r$ at distances $a$ and $b$ either side of
the CG. Take as coordinates the vertical displacement $z$ of a **reference point
$P$** offset by $e$ from the CG, and the rotation $\theta$.

Because $P$ is not the CG, accelerating $P$ vertically also produces angular
acceleration — that is **inertial coupling**. Because the springs are not
symmetric about $P$, a vertical deflection also produces a moment — that is
**elastic coupling**. Both at once, exactly like $I_{zx}$ and $K_{\phi\psi}$ in
the paper.
""")

code(r'''
def bar_system(m=12.0, I_G=0.55, k_f=9.0e3, k_r=6.0e3, a=0.32, b=0.28, e=0.0):
    """Rigid bar on two springs, coordinates (z at point P, theta).

    a, b are the spring positions measured from the CG (a forward, b rearward).
    e is the offset of the reference point P from the CG (positive forward).
    """
    # z_P = z_G + e*theta, so z_G = z_P - e*theta and the displacement at a
    # station x (measured from the CG) is  z_P + (x - e)*theta.
    # Kinetic energy (1/2) m (z_P - e th')^2 + (1/2) I_G th'^2 gives M12 = -m e.
    af, ar = a - e, -b - e            # spring stations relative to P
    M = np.array([[m,       -m * e],
                  [-m * e,   I_G + m * e**2]])
    K = np.array([[k_f + k_r,            k_f*af + k_r*ar],
                  [k_f*af + k_r*ar,      k_f*af**2 + k_r*ar**2]])
    return M, K

M, K = bar_system(e=0.10)
print("M =\n", M, "\n\nK =\n", K)
print("\ninertial coupling M12 =", M[0, 1], "   elastic coupling K12 =", K[0, 1])
''')

code(r'''
fig, ax = plt.subplots(figsize=(6.6, 2.1))
m, I_G, k_f, k_r, a, b, e = 12.0, 0.55, 9.0e3, 6.0e3, 0.32, 0.28, 0.10
ax.plot([-0.55, 0.55], [0, 0], color=SERIES[0], lw=8, solid_capstyle="round", zorder=3)
primer.spring(ax, 0, 0, y=0)   # noop keeps import used
for xs, kk, col in ((a, "$k_f$", SERIES[2]), (-b, "$k_r$", SERIES[3])):
    ax.plot([xs, xs], [-0.34, -0.05], color=col, lw=2.5, zorder=2)
    ax.plot([xs-0.07, xs+0.07], [-0.38, -0.38], color=INK, lw=2.5)
    ax.text(xs, -0.52, kk, ha="center", color=col, fontsize=10)
ax.plot([0], [0], "o", ms=9, color="white", markeredgecolor=INK, mew=1.6, zorder=4)
ax.text(0, 0.20, "CG", ha="center", color=INK, fontsize=9)
ax.plot([e], [0], "s", ms=8, color=SERIES[1], markeredgecolor="white", mew=1.3, zorder=4)
ax.text(e, -0.22, "P", ha="center", color=SERIES[1], fontsize=9)
ax.annotate("", xy=(e, 0.42), xytext=(e, 0.14),
            arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=2))
ax.text(e + 0.03, 0.44, "force applied here", color=SERIES[1], fontsize=8.5)
ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.62, 0.62); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.show()
''')

md(r"""
## 4.3 · The excitation, and what it can and cannot see

Apply a pure vertical force at $P$: $\{F\}=[F_0,\ 0]^\mathsf{T}$ (a force at $P$
produces no moment *about $P$*). Then

$$\Gamma_r=\{u^r\}^\mathsf{T}\{F\}=u^r_z F_0$$

so mode $r$ is silent iff $u^r_z=0$ — iff that mode is a **pure rotation about
$P$**.

And that has a completely intuitive reading: if a mode's instantaneous centre of
rotation sits exactly at the point where you push, then your push does no work on
it. You are pushing at the pivot.
""")

code(r'''
def analyse(e, F=np.array([1.0, 0.0]), **kw):
    """Frequencies [Hz], mode shapes (largest entry = 1), normalised participations."""
    M, K = bar_system(e=e, **kw)
    w2, V = eigh(K, M)
    V = V / V[np.argmax(np.abs(V), axis=0), np.arange(2)]
    gam = np.abs(V.T @ F) / (np.linalg.norm(V, axis=0) * np.linalg.norm(F))
    return np.sqrt(w2) / 2 / np.pi, V, gam

f_hz, V, gam = analyse(e=0.10)
for r in range(2):
    print(f"mode {r+1}: {f_hz[r]:6.2f} Hz   shape [z, theta] = {V[:, r]}   "
          f"participation = {gam[r]:.4f}")
''')

md(r"""
Both modes respond. Now sweep the reference-point offset $e$ — equivalently, move
where you push along the bar — and watch. `eigh` returns the modes in ascending
frequency, and (as the right-hand panel is about to show) the frequencies do not
move at all here, so "lower mode" and "upper mode" stay unambiguous labels
throughout the sweep.
""")

code(r'''
es = np.linspace(-0.30, 0.30, 500)
G = np.array([analyse(ei)[2] for ei in es])
FZ = np.array([analyse(ei)[0] for ei in es])

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
axes[0].plot(es*1000, G[:, 0], color=SERIES[0], label="lower mode")
axes[0].plot(es*1000, G[:, 1], color=SERIES[1], label="upper mode")
axes[0].axhline(0, color=INK2, lw=0.8, ls="--")
tidy(axes[0], "4.3 · participation vs where you push",
     "offset $e$ of the push point from the CG [mm]", "normalised participation")
axes[0].legend()

axes[1].plot(es*1000, FZ[:, 0], color=SERIES[0], label="lower mode")
axes[1].plot(es*1000, FZ[:, 1], color=SERIES[1], label="upper mode")
axes[1].set_ylim(0, 10)
tidy(axes[1], "the frequencies do not move at all", "offset $e$ [mm]", "frequency [Hz]")
axes[1].legend()
plt.tight_layout(); plt.show()
''')

md(r"""
Two things to notice.

1. The upper mode's participation touches **zero** at one offset in this range.
   That offset is the upper mode's centre of rotation — the classic "oscillation
   centre" of a bar on springs. (The lower mode has one too, but for these
   numbers it lies off the ends of the bar.)
2. **The natural frequencies do not change at all.** Of course they do not:
   moving the reference point is a change of coordinates, not a change of system
   — it is the congruence transform of notebook 1.5, which leaves the eigenvalues
   alone. Only the *shapes*, and hence the participations, are re-expressed.
   That flat line is also a good check that the mass matrix was written
   correctly: if it slopes, the inertia-coupling sign is wrong.

Rather than hunting for the dip numerically, use the condition that notebook 4.5
is about to derive: a mode is silent when $M_{12}/M_{22} = K_{12}/K_{22}$. As a
function of $e$ that is one scalar equation with one unknown.
""")

code(r'''
from scipy.optimize import brentq

def mismatch(e, **kw):
    """M12/M22 - K12/K22.  Zero means one mode is invisible to a force at P."""
    M, K = bar_system(e=e, **kw)
    return M[0, 1]/M[1, 1] - K[0, 1]/K[1, 1]

def find_roots(fn, lo, hi, n=400):
    xs = np.linspace(lo, hi, n)
    vs = np.array([fn(x) for x in xs])
    out = []
    for i in range(n - 1):
        if np.isfinite(vs[i]) and np.isfinite(vs[i+1]) and vs[i]*vs[i+1] < 0:
            out.append(brentq(fn, xs[i], xs[i+1]))
    return out

roots_e = find_roots(mismatch, -0.29, 0.29)
print("silencing offsets [mm]:", np.round(np.array(roots_e)*1000, 3))
e_star = roots_e[0]
print(f"\nusing e = {e_star*1000:.3f} mm")

f_hz, V, gam = analyse(e_star)
print(f"\nfrequencies : {f_hz.round(2)} Hz")
print("mode shapes [z; theta]:\n", V)
print("participations:", gam)
quiet = int(np.argmin(gam))
print(f"\nmode {quiet+1} is a pure rotation: its z component is {V[0, quiet]:.2e}")
''')

code(r'''
# ...and the response is a single-mode response at every frequency
Mx, Kx = bar_system(e=e_star)
w = 2*np.pi*np.linspace(1, 60, 3000)
F = np.array([1.0, 0.0])
Q = np.array([np.linalg.solve(-wi**2*Mx + Kx, F) for wi in w])

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.semilogy(w/2/np.pi, np.abs(Q[:, 0]), color=SERIES[0], label="$z$ at P")
ax.semilogy(w/2/np.pi, np.abs(Q[:, 1]), color=SERIES[1], label=r"$\theta$")
for fr in f_hz:
    ax.axvline(fr, color=GRID, lw=1.2, zorder=0)
    ax.annotate(f"{fr:.2f} Hz", xy=(fr, 1e-6), xytext=(4, 6),
                textcoords="offset points", color=INK2, fontsize=8,
                rotation=90, va="bottom")
tidy(ax, "4.3 · both natural frequencies are there — only one of them answers",
     "frequency [Hz]", "response per unit force")
ax.legend(loc="upper right")
plt.tight_layout(); plt.show()
''')

md(r"""
Two grey lines, one peak. The second mode exists, sits right there in the
frequency range, and does nothing at all.

## 4.4 · The paper's version of the problem is the *inverse* one

In the sweep above we moved the excitation until it stopped seeing a mode. An
engine designer cannot do that: **the excitation is fixed by the engine.** The
crank produces a rocking couple about an axis set by the cylinder geometry, and
no amount of mount design changes it.

So the problem turns around:

> Given a fixed excitation direction $\{F\}$, choose the *system* so that one of
> its mode shapes becomes orthogonal to $\{F\}$.

In the bar: keep the push at $P$ where it is, and change the springs until a
mode's centre of rotation moves onto $P$.
""")

code(r'''
e_fixed = -0.05               # the push point is where the engine says it is
kr_base = 6.0e3

def mismatch_ratio(ratio):
    return mismatch(e_fixed, k_f=ratio*kr_base, k_r=kr_base)

r_star = find_roots(mismatch_ratio, 0.3, 10.0)[0]
print(f"stiffness ratio that silences a mode:  k_f/k_r = {r_star:.4f}")

f_hz, V, gam = analyse(e_fixed, k_f=r_star*kr_base, k_r=kr_base)
print(f"\nfrequencies : {f_hz.round(2)} Hz")
print("mode shapes :\n", V)
print("participations:", gam)

ratios = np.linspace(0.3, 10.0, 600)
lk = np.array([analyse(e_fixed, k_f=r*kr_base, k_r=kr_base)[2].min() for r in ratios])

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.semilogy(ratios, lk, color=SERIES[2])
ax.axvline(r_star, color=INK2, ls="--", lw=1.0)
ax.text(r_star + 0.3, 0.006, f"$k_f/k_r$ = {r_star:.3f}\n"
        f"the quiet mode goes silent", color=INK2, fontsize=8.5, va="center",
        bbox=dict(fc="white", ec="none", alpha=0.9, pad=2))
tidy(ax, "4.4 · tuning the SYSTEM to the excitation, not the other way round",
     "stiffness ratio $k_f / k_r$", "participation of the quiet mode")
plt.tight_layout(); plt.show()
''')

md(r"""
## 4.5 · The condition, in closed form

Do it algebraically rather than numerically, because the answer is literally the
paper's Eq. (16).

Require mode 2 to be pure rotation, $\{u^2\}=[0,\ 1]^\mathsf{T}$. Substitute into
$\left([K]-\omega_2^2[M]\right)\{u^2\}=0$ and read the two rows:

$$\text{row 1:}\quad K_{12}-\omega_2^2 M_{12}=0 \;\Rightarrow\; \omega_2^2=\frac{K_{12}}{M_{12}}$$
$$\text{row 2:}\quad K_{22}-\omega_2^2 M_{22}=0 \;\Rightarrow\; \omega_2^2=\frac{K_{22}}{M_{22}}$$

Both must hold, so

$$\boxed{\ \frac{M_{12}}{M_{22}}=\frac{K_{12}}{K_{22}}\ }$$

**The ratio of inertial coupling to inertia must equal the ratio of elastic
coupling to stiffness.** Put the paper's symbols in — $M_{12}=-I_{zx}$,
$M_{22}=I_z$, $K_{12}=K_{\phi\psi}$, $K_{22}=K_{\psi\psi}$ — and this is

$$-\frac{I_{zx}}{I_z}=\frac{K_{\phi\psi}}{K_{\psi\psi}}
\qquad\Longleftrightarrow\qquad
R=\frac{I_{zx}}{I_z}=-\frac{K_{\phi\psi}}{K_{\psi\psi}}$$

which is Eq. (16) exactly.
""")

code(r'''
Mx, Kx = bar_system(e=e_fixed, k_f=r_star*kr_base, k_r=kr_base)
print(f"M12/M22 = {Mx[0,1]/Mx[1,1]:.8f}")
print(f"K12/K22 = {Kx[0,1]/Kx[1,1]:.8f}")
print("equal?", np.isclose(Mx[0,1]/Mx[1,1], Kx[0,1]/Kx[1,1]))
print(f"\nand w2^2 from both rows: {Kx[0,1]/Mx[0,1]:.4f}  vs  {Kx[1,1]/Mx[1,1]:.4f}")
''')

md(r"""
### Why the same condition can also be read as orthogonality

The paper reaches it a different way — through the orthogonality conditions of
notebook 3 rather than through the eigenvector equation. Both routes give the
same determinant. Worth checking they agree:

$\{u^1\}^\mathsf{T}[M]\{u^2\}=0$ with $\{u^2\}=[0,1]^\mathsf{T}$ gives
$M_{12}u^1_1 + M_{22}u^1_2=0$; the $[K]$ version gives
$K_{12}u^1_1 + K_{22}u^1_2=0$. Two homogeneous equations in the same unknown
ratio — a non-trivial solution needs

$$\begin{vmatrix}M_{12}&M_{22}\\K_{12}&K_{22}\end{vmatrix}=0$$

the same condition. (In the paper this is Eqs. (13)–(15).)
""")

code(r'''
det = Mx[0,1]*Kx[1,1] - Mx[1,1]*Kx[0,1]
print(f"determinant of the orthogonality pair: {det:.3e}  (should be ~0)")

# the surviving mode's shape ratio, from either row
print(f"u1_theta / u1_z  from M:  {-Mx[0,1]/Mx[1,1]:.6f}")
print(f"                 from K:  {-Kx[0,1]/Kx[1,1]:.6f}")
w2v, Vv = eigh(Kx, Mx)
Vv = Vv / Vv[np.argmax(np.abs(Vv), axis=0), np.arange(2)]
live = int(np.argmax(np.abs(Vv.T @ np.array([1.0, 0.0]))))
print(f"                 actual:  {Vv[1, live]/Vv[0, live]:.6f}")
''')

md(r"""
## 4.6 · What this bought, and what it did not

The bar now has two resonances but only one of them can be excited. So the
designer is free to place the *other* one anywhere at all — high, for stiffness —
without paying for it in vibration.

What it did **not** buy: the live mode is still live. You still have to put it
somewhere sensible, and you still have to drive through it. That is the honest
limit of the method, and it is why the paper's own Fig. 14 shows the machine
being *worse* right at the resonance crossing.

---

**Next:** [5 · From point masses to a rigid body](05_rigid_body_and_inertia.ipynb)
""")

# ###########################################################################
# 05
# ###########################################################################
start("05_rigid_body_and_inertia")

md(r"""
# 5 · From point masses to a rigid body

**Math primer — notebook 5 of 7.**

Everything so far had one or two scalar coordinates. An engine on mounts has six,
and its mass matrix contains an *inertia tensor* rather than a number. This
notebook builds that tensor from scratch and shows exactly where the paper's
$I_{zx}$ comes from.

**You should come out knowing:** what a product of inertia is, why principal axes
are the eigenvectors of the inertia tensor, how a tensor transforms under
rotation, and why a tilted principal axis makes roll and yaw talk to each other.
""")

code(HEAD)

md(r"""
## 5.1 · Six coordinates

$$\{q\}=[x,\ y,\ z,\ \phi,\ \theta,\ \psi]^\mathsf{T}$$

Three translations of the centre of gravity, three small rotations about it. With
the CG as origin the kinetic energy splits cleanly:

$$T=\tfrac12 m\,\{\dot u\}^\mathsf{T}\{\dot u\}
   +\tfrac12\{\dot\omega\}^\mathsf{T}[J]\{\dot\omega\}
\qquad\Rightarrow\qquad
[M]=\begin{bmatrix}m[I_3]&0\\0&[J]\end{bmatrix}$$

There is no translation–rotation coupling **only because** the origin is the CG.
Put it anywhere else and the off-diagonal blocks reappear — exactly as in the bar
of notebook 4, where offsetting $P$ from the CG created $M_{12}=me$.
""")

md(r"""
## 5.2 · The inertia tensor, from point masses

For a set of point masses $m_i$ at positions $\{r_i\}$ relative to the CG,

$$[J]=\sum_i m_i\left(\|r_i\|^2[I_3]-\{r_i\}\{r_i\}^\mathsf{T}\right)$$

The diagonal entries are the familiar moments of inertia. The off-diagonal
entries, $-\sum m_i x_i z_i$ and so on, are the **products of inertia**. They are
zero only if the mass is symmetric about the coordinate planes.

Careful with signs: the tensor entry is $J_{xz}=-\sum m_i x_i z_i$, whereas the
quantity the paper calls $I_{zx}$ is $\sum m_i x_i z_i$. So $J_{xz}=-I_{zx}$,
which is why the paper's mass matrix shows $-I_{zx}$ in the $\phi\psi$ slot.
""")

code(r'''
def inertia_from_points(masses, positions):
    """Inertia tensor about the centre of mass of a cloud of point masses."""
    m = np.asarray(masses, float)
    r = np.asarray(positions, float)
    cg = (m[:, None] * r).sum(0) / m.sum()
    d = r - cg
    J = np.zeros((3, 3))
    for mi, di in zip(m, d):
        J += mi * (di @ di * np.eye(3) - np.outer(di, di))
    return J, cg, m.sum()

# a crude "engine": a heavy crankcase low and rearward, two lighter cylinders
# leaning forward and up.
masses = [22.0, 8.0, 8.0, 2.0]
positions = [[-0.05, 0.00, -0.09],     # crankcase
             [ 0.07, 0.035, 0.11],     # cylinder 1
             [ 0.07, -0.035, 0.11],    # cylinder 2
             [-0.18, 0.00, 0.02]]      # gearbox / clutch
J, cg, m_tot = inertia_from_points(masses, positions)
print(f"total mass {m_tot:.1f} kg, CG at {cg} m\n")
print("inertia tensor about the CG [kg m^2]:\n", J)
print("\nproduct of inertia J_xz =", J[0, 2],
      "  ->  the paper's I_zx =", -J[0, 2])
''')

md(r"""
## 5.3 · Principal axes are eigenvectors

There is always some rotated frame in which all products of inertia vanish. That
frame's axes are the **principal axes**, and finding them is an eigenproblem —
the ordinary symmetric one this time, because $[J]$ is symmetric:

$$[J]\{v\}=I\{v\}$$
""")

code(r'''
Iprin, axes3 = np.linalg.eigh(J)
print("principal moments of inertia:", Iprin)
print("principal axes (columns):\n", axes3)
print("\ncheck: axes^T J axes is diagonal ->\n", axes3.T @ J @ axes3)

# the in-plane tilt, measured in the x-z plane
v = axes3[:, int(np.argmax(np.abs(axes3[0, :])))]   # the axis closest to x
if v[0] < 0:
    v = -v
alpha = np.arctan2(-v[2], v[0])
print(f"\nthe principal axis nearest x is tilted by {np.degrees(alpha):.2f} deg in the x-z plane")
''')

md(r"""
## 5.4 · Rotating a tensor — where Eq. (20) comes from

If the principal frame is reached by a rotation $[R]$, then in the working frame

$$[J]=[R]\,\mathrm{diag}(I_\xi,I_\eta,I_\zeta)\,[R]^\mathsf{T}$$

Note it is $R\,\Lambda\,R^\mathsf{T}$, *not* $R\Lambda$ — a tensor transforms on
both sides. Take $[R]$ to be a rotation by $\alpha$ about $y$ and multiply it out:

$$I_x=I_\xi\cos^2\alpha+I_\zeta\sin^2\alpha$$
$$I_z=I_\xi\sin^2\alpha+I_\zeta\cos^2\alpha$$
$$I_{zx}=(I_\xi-I_\zeta)\sin\alpha\cos\alpha$$

which is the paper's Eq. (20). Let's confirm it symbolically and numerically.
""")

code(r'''
import sympy as sp

al, Ixi, Ieta, Izeta = sp.symbols("alpha I_xi I_eta I_zeta", real=True)
R = sp.Matrix([[sp.cos(al), 0, sp.sin(al)],
               [0, 1, 0],
               [-sp.sin(al), 0, sp.cos(al)]])
Jsym = sp.simplify(R * sp.diag(Ixi, Ieta, Izeta) * R.T)
print("J_xx  =", sp.simplify(Jsym[0, 0]))
print("J_zz  =", sp.simplify(Jsym[2, 2]))
print("J_xz  =", sp.simplify(sp.expand_trig(Jsym[0, 2])))
print("\nso I_zx = -J_xz =", sp.simplify(-Jsym[0, 2]))
print("   which is (I_xi - I_zeta) sin(a) cos(a):",
      sp.simplify(-Jsym[0, 2] - (Ixi - Izeta)*sp.sin(al)*sp.cos(al)) == 0)
''')

code(r'''
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])

I_xi, I_eta, I_zeta = 0.95, 1.30, 1.60
for adeg in (0, 10, 30, 45, 90):
    a = np.deg2rad(adeg)
    Jn = rot_y(a) @ np.diag([I_xi, I_eta, I_zeta]) @ rot_y(a).T
    print(f"alpha = {adeg:3d} deg :  I_x = {Jn[0,0]:.4f}  I_z = {Jn[2,2]:.4f}  "
          f"I_zx = {-Jn[0,2]:+.4f}   (closed form {(I_xi-I_zeta)*np.sin(a)*np.cos(a):+.4f})")
''')

md(r"""
Read the last column. $I_{zx}$ is zero when $\alpha=0$ or $90°$ — when the
principal axes line up with the working axes — and largest at $45°$. It is
proportional to $(I_\xi-I_\zeta)$: a body whose two in-plane inertias are equal
has no product of inertia at any tilt.

## 5.5 · Why this couples roll and yaw

The rotational part of the equations of motion is
$[J]\{\dot\omega\}+\ldots=\{M\}$, with $\{\omega\}=[\dot\phi,\dot\theta,\dot\psi]$.
Written out, the $\phi$ row contains $J_{xz}\ddot\psi$: **an angular acceleration
about $z$ produces a moment about $x$.** So roll and yaw are inertially coupled
whenever $I_{zx}\ne 0$.

Physically: spin a tilted body about the vertical and it tries to wobble, because
its mass is not distributed symmetrically about that axis.
""")

code(r'''
alpha = np.deg2rad(30)
J30 = rot_y(alpha) @ np.diag([I_xi, I_eta, I_zeta]) @ rot_y(alpha).T
M6 = np.zeros((6, 6)); M6[:3, :3] = 40.0*np.eye(3); M6[3:, 3:] = J30
lab = ["x", "y", "z", "φ", "θ", "ψ"]

fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.imshow(np.abs(M6) > 1e-12, cmap="Blues", vmin=0, vmax=1.6)
for i in range(6):
    for j in range(6):
        if abs(M6[i, j]) > 1e-12:
            ax.text(j, i, "•", ha="center", va="center", color="white", fontsize=13)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(lab); ax.set_yticklabels(lab); ax.grid(False)
tidy(ax, "5.5 · [M] with a tilted principal axis")
plt.tight_layout(); plt.show()
print("the only off-diagonal entries are the phi-psi pair:", M6[3, 5], M6[5, 3])
''')

md(r"""
Two dots off the diagonal, in the $\phi\psi$ slots. That single pair of entries is
the "dynamic coupling" the paper's abstract refers to, and the thing its elastic
design is going to be matched against.

---

**Next:** [6 · Mounts, stiffness matrices and the elastic centre](06_mounts_and_stiffness.ipynb)
""")

# ###########################################################################
# 06
# ###########################################################################
start("06_mounts_and_stiffness")

md(r"""
# 6 · Mounts, stiffness matrices and the elastic centre

**Math primer — notebook 6 of 7.**

Notebook 1 built a stiffness matrix from springs between scalar coordinates.
Now the springs sit at *positions* on a rigid body, and moving the body moves each
mount by an amount that depends on both translation and rotation.

**You should come out knowing:** how a single mount at $\{r\}$ contributes to the
$6\times6$ $[K]$, what the skew-symmetric matrix is doing there, what the elastic
centre is, and the geometric reason narrow lateral mount spacing collapses roll
stiffness without touching anything else.
""")

code(HEAD)

md(r"""
## 6.1 · How far does a mount deflect?

Small rigid-body motion $\{q\}=[\{u\},\{\theta\}]$ moves the material point at
$\{r\}$ by

$$\{d\}=\{u\}+\{\theta\}\times\{r\}$$

Cross products are linear, so write them as a matrix. For any vector $\{v\}$
define the **skew-symmetric matrix**

$$[S(v)]=\begin{bmatrix}0&-v_z&v_y\\ v_z&0&-v_x\\ -v_y&v_x&0\end{bmatrix}
\qquad\text{so that}\qquad [S(v)]\{w\}=\{v\}\times\{w\}$$

Then $\{\theta\}\times\{r\}=-[S(r)]\{\theta\}$ and

$$\{d\}=\begin{bmatrix}[I_3] & -[S(r)]\end{bmatrix}\{q\}\;\equiv\;[B]\{q\}$$
""")

code(r'''
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

r = np.array([0.23, 0.05, -0.12])
th = np.array([0.01, -0.02, 0.03])
print("theta x r  (numpy cross) :", np.cross(th, r))
print("S(theta) @ r             :", skew(th) @ r)
print("-S(r) @ theta            :", -skew(r) @ th)

def B_of(r):
    return np.hstack([np.eye(3), -skew(r)])

q = np.array([1e-3, 0, 0, 0.01, -0.02, 0.03])
print("\ndeflection of the mount at r for q =", q, ":\n ", B_of(r) @ q)
''')

md(r"""
## 6.2 · One mount's contribution to $[K]$

Same energy argument as notebook 1. The mount stores
$\tfrac12\{d\}^\mathsf{T}[k]\{d\}$ with $[k]$ its $3\times3$ rate matrix, so with
$\{d\}=[B]\{q\}$ the energy is $\tfrac12\{q\}^\mathsf{T}[B]^\mathsf{T}[k][B]\{q\}$
and

$$[K]_{\text{mount}}=[B]^\mathsf{T}[k][B]
=\begin{bmatrix}[k] & -[k][S(r)]\\ [S(r)][k] & -[S(r)][k][S(r)]\end{bmatrix}$$

Exactly the same pattern as $k\{b\}\{b\}^\mathsf{T}$ in notebook 1 — a congruence
transform of the mount's own rate matrix into the body's coordinates. Symmetry is
automatic, and so is positive semi-definiteness.

If the mount's own principal directions are not $x,y,z$ (a bush at an angle),
first rotate its rates: $[k]=[R_m]\,\mathrm{diag}(k_1,k_2,k_3)\,[R_m]^\mathsf{T}$.
""")

code(r'''
def mount_K(r, rates, R_m=None):
    """6x6 stiffness contribution of one mount at position r."""
    R_m = np.eye(3) if R_m is None else R_m
    k = R_m @ np.diag(rates) @ R_m.T
    B = B_of(r)
    return B.T @ k @ B

Km = mount_K([0.23, 0.05, -0.12], [4.9e6, 1.1e6, 4.9e6])
ev = np.linalg.eigvalsh(Km)
print("symmetric?", np.allclose(Km, Km.T))
print("eigenvalues:", ev)
print(f"most negative eigenvalue: {ev.min():.2e}  "
      f"(zero to round-off, relative to {ev.max():.2e})")
print("rank:", np.linalg.matrix_rank(Km), " <- one mount restrains only 3 of 6 DOF")
''')

md(r"""
A single mount has rank 3 — it cannot stop the body rotating about itself. You
need enough mounts, well spread, for $[K]$ to become full rank. That is the
6-DOF version of "the structure must be properly restrained".

## 6.3 · Four mounts, and the block structure appearing

Put four bushes on symmetrically in $y$ and watch which entries survive.
""")

code(r'''
def four_mounts(x_f=0.23, x_r=-0.26, z_f=-0.115, z_r=0.132, y_half=0.05,
                kf=(4.9e6, 1.1e6, 4.9e6), kr=(3.9e6, 0.85e6, 3.9e6), tilt=0.0):
    c, s = np.cos(tilt), np.sin(tilt)
    R_m = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    K = np.zeros((6, 6))
    for x, z, kk in ((x_f, z_f, kf), (x_r, z_r, kr)):
        for sy in (+1, -1):
            K += mount_K([x, sy*y_half, z], kk, R_m)
    return K

K6 = four_mounts()
lab = ["x", "y", "z", "φ", "θ", "ψ"]
np.set_printoptions(precision=1, suppress=True)
print("K (units N/m and N m/rad mixed -- see note below):\n")
print("        " + "".join(f"{l:>12s}" for l in lab))
for i, l in enumerate(lab):
    print(f"{l:>6s}  " + "".join(f"{v:12.3g}" for v in K6[i]))
np.set_printoptions(precision=4, suppress=True)

fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.imshow(np.abs(K6) > 1e-6 * np.abs(K6).max(), cmap="Blues", vmin=0, vmax=1.6)
for i in range(6):
    for j in range(6):
        if abs(K6[i, j]) > 1e-6 * np.abs(K6).max():
            ax.text(j, i, "•", ha="center", va="center", color="white", fontsize=13)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(lab); ax.set_yticklabels(lab); ax.grid(False)
tidy(ax, "6.3 · [K] for four laterally symmetric mounts")
plt.tight_layout(); plt.show()
''')

md(r"""
Lateral symmetry kills every entry that would couple the in-plane motions
($x,z,\theta$) to the out-of-plane ones ($y,\phi,\psi$). What survives is the
$x$–$z$–$\theta$ block and the $y$–$\phi$–$\psi$ block. Combine that with the
inertia structure from notebook 5 and you get the paper's Eq. (7) picture.

> **A note on units.** Mixing translations (m) and rotations (rad) in one vector
> means $[K]$ has mixed units: N/m, N/rad, N·m/rad. That is harmless for the
> algebra but makes the printed numbers hard to compare, and it means "the
> largest entry of a mode shape" is not a meaningful notion until you scale
> rotations by a characteristic length. `orthomount.describe_mode` does exactly
> that, with a 0.25 m default.

## 6.4 · The elastic centre

The translation–rotation coupling block is $-[k][S(r)]$ summed over mounts. There
is generally one point — the **elastic centre** — about which that block
vanishes: push there and the body translates without rotating.

Shift the origin by $\{r_e\}$ and the coupling block becomes
$[K_c]+[K_t][S(r_e)]$. Setting it to zero and solving column by column locates it.
""")

code(r'''
def elastic_centre(K):
    Kt, Kc = K[:3, :3], K[:3, 3:]
    A = np.zeros((9, 3)); b = np.zeros(9)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        A[3*i:3*i+3, :] = Kt @ skew(e)
        b[3*i:3*i+3] = Kc[:, i]
    return np.linalg.lstsq(A, b, rcond=None)[0]

print("elastic centre of the four-mount set:", elastic_centre(four_mounts()) * 1000, "mm")

# deliberately move the front mounts up and watch it shift
print("front mounts 60 mm higher   :",
      elastic_centre(four_mounts(z_f=-0.055)) * 1000, "mm")
print("front mounts twice as stiff :",
      elastic_centre(four_mounts(kf=(9.8e6, 2.2e6, 9.8e6))) * 1000, "mm")
''')

md(r"""
The paper *assumes* the elastic centre coincides with the CG. That is what makes
its $[K]$ block-diagonal and lets the 6-DOF problem collapse to $2\times2$. In a
real layout it is a design target you have to hit, not a given — which is why
`orthomount`'s mount-layout solver carries it as an explicit residual.

## 6.5 · The lever arms — why narrow spacing collapses roll stiffness

Multiply out $[B]^\mathsf{T}[k][B]$ for a mount with rates $(k_x,k_y,k_z)$ at
$(x,y,z)$ and read the rotational diagonal:

$$K_{\phi\phi}=k_z y^2 + k_y z^2,\qquad
  K_{\theta\theta}=k_z x^2 + k_x z^2,\qquad
  K_{\psi\psi}=k_y x^2 + k_x y^2$$

Roll stiffness is the only one carrying $y^2$ against the big vertical rate.
Squeeze the mounts together laterally and $K_{\phi\phi}$ falls as $y^2$ while
bounce, fore/aft and pitch do not move at all.
""")

code(r'''
ys = np.linspace(0.02, 0.20, 200)
diag = np.array([np.diag(four_mounts(y_half=yy)) for yy in ys])

fig, ax = plt.subplots(figsize=(7.2, 3.6))
names = ["$K_{xx}$", "$K_{yy}$", "$K_{zz}$",
         r"$K_{\phi\phi}$ (roll)", r"$K_{\theta\theta}$", r"$K_{\psi\psi}$"]
# The four spacing-independent curves sit exactly on top of each other at 1.0,
# so give them distinct dashes -- otherwise only the last one drawn is visible.
dashes = [(1, 0), (4, 2), (1, 2), (1, 0), (6, 2), (1, 0)]
for j in range(6):
    lw = 2.8 if j == 3 else 1.5
    ax.semilogy(ys*1000, diag[:, j]/diag[0, j], color=SERIES[j], lw=lw,
                dashes=dashes[j], label=names[j])
ax.set_ylim(0.5, 200)
tidy(ax, "6.5 · diagonal stiffness vs lateral half-spacing, normalised to y = 20 mm",
     "bush half-spacing y [mm]", "stiffness relative to y = 20 mm")
ax.legend(ncol=3, loc="upper left")
plt.tight_layout(); plt.show()

print("Only roll and yaw depend on lateral spacing; roll is the one that matters,")
print("because it is the direction the engine's rocking couple pushes in.")
''')

md(r"""
### Cross-check against the library
""")

code(r'''
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import orthomount as om

mounts = []
for x, z, kk in ((0.23, -0.115, (4.9e6, 1.1e6, 4.9e6)),
                 (-0.26, 0.132, (3.9e6, 0.85e6, 3.9e6))):
    for sy in (+1, -1):
        mounts.append(om.Mount([x, sy*0.05, z], kk))
body = om.RigidBody.from_principal(40.0, 0.95, 1.30, 1.60, np.deg2rad(30))
K_lib = om.MountSystem(body, mounts).K

print("hand-built matches orthomount?", np.allclose(K_lib, four_mounts()))
print("max difference:", np.max(np.abs(K_lib - four_mounts())))
''')

md(r"""
---

**Next:** [7 · Putting it together — the paper's 2×2](07_the_papers_2x2.ipynb)
""")

# ###########################################################################
# 07
# ###########################################################################
start("07_the_papers_2x2")

md(r"""
# 7 · Putting it together — the paper's 2×2

**Math primer — notebook 7 of 7.**

Every piece is now on the table. This notebook assembles them into Furusawa's
derivation, symbolically, and checks each step against `orthomount`.

**You should come out knowing:** how Eqs. (16), (22), (25), (27) and (29) follow
from the four preceding notebooks, and which symbol in the paper corresponds to
which line of code.
""")

code(r'''
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import primer
from primer import SERIES, INK, INK2, GRID, tidy
import orthomount as om
primer.use_style()
D, Dg = np.deg2rad, np.rad2deg
''')

md(r"""
## 7.1 · The reduction, in one place

From **notebook 5**: a rigid body with its in-plane principal axes tilted by
$\alpha$ has

$$[M]_{\phi\psi}=\begin{bmatrix}I_x&-I_{zx}\\-I_{zx}&I_z\end{bmatrix}$$

From **notebook 6**: laterally symmetric mounts give a $[K]$ that couples only
$\phi$ with $\psi$ (and $x$ with $z$), and with the elastic principal axes tilted
by $\beta$,

$$[K]_{\phi\psi}=\begin{bmatrix}K_{\phi\phi}&K_{\phi\psi}\\K_{\phi\psi}&K_{\psi\psi}\end{bmatrix}$$

From **notebook 4**: with a pure $\phi$ excitation, mode 2 is silent iff
$M_{12}/M_{22}=K_{12}/K_{22}$.

Put those together and there is nothing left to derive — only algebra to carry
out. Let SymPy do it.
""")

code(r'''
al, be = sp.symbols("alpha beta", real=True)
Ixi, Ieta, Izeta = sp.symbols("I_xi I_eta I_zeta", positive=True)
Kxi, Keta, Kzeta = sp.symbols("K_xi K_eta K_zeta", positive=True)

def Ry(a):
    return sp.Matrix([[sp.cos(a), 0, sp.sin(a)],
                      [0, 1, 0],
                      [-sp.sin(a), 0, sp.cos(a)]])

J = sp.simplify(Ry(al) * sp.diag(Ixi, Ieta, Izeta) * Ry(al).T)
Kb = sp.simplify(Ry(be) * sp.diag(Kxi, Keta, Kzeta) * Ry(be).T)

# the phi-psi 2x2 sub-blocks (rows/cols 0 and 2 of the 3x3 rotational blocks)
Mr = sp.Matrix([[J[0, 0], J[0, 2]], [J[2, 0], J[2, 2]]])
Kr = sp.Matrix([[Kb[0, 0], Kb[0, 2]], [Kb[2, 0], Kb[2, 2]]])

print("M_r =")
sp.pprint(sp.simplify(Mr))
print("\nK_r =")
sp.pprint(sp.simplify(Kr))
''')

md(r"""
## 7.2 · Eq. (16), symbolically

The silencing condition from notebook 4 is $M_{12}/M_{22}=K_{12}/K_{22}$.
Define $R$ as the paper does — remembering $I_{zx}=-J_{xz}$, so
$M_{12}/M_{22}=-I_{zx}/I_z$:

$$R\equiv\frac{I_{zx}}{I_z}=-\frac{M_{12}}{M_{22}},\qquad
R=-\frac{K_{\phi\psi}}{K_{\psi\psi}}=-\frac{K_{12}}{K_{22}}$$
""")

code(r'''
R_inertia  = sp.simplify(-Mr[0, 1] / Mr[1, 1])
R_stiff    = sp.simplify(-Kr[0, 1] / Kr[1, 1])
print("R from inertia   =", sp.simplify(R_inertia))
print("R from stiffness =", sp.simplify(R_stiff))
''')

md(r"""
## 7.3 · Eq. (22): introduce the ratios $p$ and $q$

Divide numerator and denominator through by $I_\zeta\cos^2\alpha$ (and
$K_\zeta\cos^2\beta$), and write $p=I_\xi/I_\zeta$, $q=K_\xi/K_\zeta$:

$$\frac{(p-1)\tan\alpha}{p\tan^2\alpha+1}=R=\frac{(q-1)\tan\beta}{q\tan^2\beta+1}$$
""")

code(r'''
p, q, t = sp.symbols("p q t", positive=True)

# Substituting alpha = atan(t) turns every sin/cos into a rational function of
# t = tan(alpha), which is the form Eq. (22) is written in -- and which SymPy
# can then cancel to zero without any coaxing.
expr_i = sp.simplify(R_inertia.subs(Ixi, p*Izeta))
target = (p - 1)*sp.tan(al) / (p*sp.tan(al)**2 + 1)
diff_i = sp.simplify((expr_i - target).subs(al, sp.atan(t)))
print("R (inertia side) as a function of t = tan(alpha):")
sp.pprint(sp.simplify(expr_i.subs(al, sp.atan(t))))
print("\ninertia side minus Eq.(22) form:", diff_i)

expr_k = sp.simplify(R_stiff.subs(Kxi, q*Kzeta))
target_k = (q - 1)*sp.tan(be) / (q*sp.tan(be)**2 + 1)
diff_k = sp.simplify((expr_k - target_k).subs(be, sp.atan(t)))
print("stiffness side minus Eq.(22) form:", diff_k)
''')

md(r"""
Both differences are zero, so Eq. (22) is confirmed. Note that the two sides are
*identical functions* of their arguments — which is why the paper's Fig. 2 can be
read either way.

## 7.4 · Eqs. (25) and (27): modal inertia and frequency

The surviving mode's shape is $[1,\ R]^\mathsf{T}$ (notebook 4: the ratio comes
straight from the orthogonality row). Its modal inertia is
$m_1=\{u^1\}^\mathsf{T}[M]_{\phi\psi}\{u^1\}$, and its frequency is
$\omega_1^2=k_1/m_1$.
""")

code(r'''
u1 = sp.Matrix([1, sp.symbols("R")])
Rs = sp.symbols("R")
m1 = sp.simplify((u1.T * Mr * u1)[0, 0].subs(Rs, R_inertia))
print("m1 =", sp.simplify(m1))
print("   = I_xi I_zeta / I_z ?",
      sp.simplify(m1 - Ixi*Izeta/Mr[1, 1]) == 0)
print("   with p = I_xi/I_zeta:",
      sp.simplify(m1.subs(Ixi, p*Izeta) - p*Izeta/(p*sp.sin(al)**2 + sp.cos(al)**2)) == 0)

k1 = sp.simplify((u1.T * Kr * u1)[0, 0].subs(Rs, R_stiff))
w1sq = sp.simplify(k1 / m1.subs(Rs, R_inertia))
# express in the paper's form
w1_paper = (Kxi*(p*sp.sin(al)**2 + sp.cos(al)**2) /
            (Ixi*(q*sp.sin(be)**2 + sp.cos(be)**2)))
check = sp.simplify(sp.trigsimp(
    w1sq.subs({Ixi: p*Izeta, Kxi: q*Kzeta}) -
    w1_paper.subs({Ixi: p*Izeta, Kxi: q*Kzeta})))
print("\nomega_1^2 minus Eq.(27):", check)
''')

md(r"""
## 7.5 · Eq. (29): the rocking axis

The mode shape is $[\phi,\psi]=[1,R]$, so the rotation vector points along
$(1,0,R)$ and the axis is inclined by $\delta$ with $\tan\delta=-R$ (the sign
following the same convention as $\alpha$). Substituting Eq. (22) gives

$$\tan\delta=\frac{(1-p)\tan\alpha}{p\tan^2\alpha+1},\qquad
\tan(\alpha-\delta)=p\tan\alpha,\qquad \tan(\beta-\delta)=q\tan\beta$$
""")

code(r'''
# Work in t = tan(alpha) throughout, and use the angle-difference identity
#     tan(a - d) = (tan a - tan d) / (1 + tan a tan d)
tan_alpha = t
tan_delta = -(p - 1)*t / (p*t**2 + 1)          # = -R, from Eq. (22)

tan_a_minus_d = sp.simplify((tan_alpha - tan_delta) / (1 + tan_alpha*tan_delta))
print("tan(alpha - delta) =", sp.simplify(tan_a_minus_d))
print("equals p tan(alpha)?", sp.simplify(tan_a_minus_d - p*t) == 0)

# and the stiffness twin, Eq. (32)
tb = sp.symbols("t_b", positive=True)
tan_delta_b = -(q - 1)*tb / (q*tb**2 + 1)
tan_b_minus_d = sp.simplify((tb - tan_delta_b) / (1 + tb*tan_delta_b))
print("tan(beta - delta)  =", sp.simplify(tan_b_minus_d))
print("equals q tan(beta)?", sp.simplify(tan_b_minus_d - q*tb) == 0)
''')

md(r"""
## 7.6 · Everything at once, numerically

Take the reconstructed example engine, run the whole chain by hand, and compare
with `orthomount` at every step.
""")

code(r'''
I_xi, I_eta, I_zeta = 0.95, 1.30, 1.60
alpha = D(30.0)
K_xi, K_eta, K_zeta = 5.30e4, 4.0e4, 2.45e4
pn, qn = I_xi/I_zeta, K_xi/K_zeta

# --- hand-built, using only what the primer derived ------------------------
def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])

Jn = rot_y(alpha) @ np.diag([I_xi, I_eta, I_zeta]) @ rot_y(alpha).T
Mr_n = Jn[np.ix_([0, 2], [0, 2])]
R_hand = -Mr_n[0, 1] / Mr_n[1, 1]

# solve Eq.(22) for beta:  R q tan^2(b) - (q-1) tan(b) + R = 0
aq, bq, cq = R_hand*qn, -(qn - 1.0), R_hand
disc = bq**2 - 4*aq*cq
t1, t2 = (-bq + np.sqrt(disc))/(2*aq), (-bq - np.sqrt(disc))/(2*aq)
beta_hand = np.arctan(min([t1, t2], key=abs))

Kbn = rot_y(beta_hand) @ np.diag([K_xi, K_eta, K_zeta]) @ rot_y(beta_hand).T
Kr_n = Kbn[np.ix_([0, 2], [0, 2])]

u1n = np.array([1.0, R_hand])
m1_hand = u1n @ Mr_n @ u1n
k1_hand = u1n @ Kr_n @ u1n
f1_hand = np.sqrt(k1_hand/m1_hand)/(2*np.pi)
delta_hand = np.arctan(-R_hand)

print("            hand-derived        orthomount")
print(f"R        {R_hand:16.8f}   {om.R_from_inertia(pn, alpha):14.8f}")
print(f"beta     {Dg(beta_hand):16.8f}   {Dg(om.solve_beta_for_R(R_hand, qn)[0]):14.8f}   [deg]")
print(f"m1       {m1_hand:16.8f}   {om.modal_mass_paper(I_xi, I_zeta, alpha):14.8f}")
print(f"f1       {f1_hand:16.8f}   "
      f"{om.omega1_paper(K_xi, I_xi, pn, qn, alpha, beta_hand)/2/np.pi:14.8f}   [Hz]")
print(f"delta    {Dg(delta_hand):16.8f}   {Dg(om.delta_paper(pn, alpha)):14.8f}   [deg]")
''')

code(r'''
# --- and confirm the mode really is silent, in the full 6-DOF system -------
M6, K6 = om.paper_system(40.0, I_xi, I_eta, I_zeta, alpha,
                         K_xi, K_eta, K_zeta, beta_hand,
                         K_trans=(1.8e7, 3.5e6, 1.7e7))
res = om.modal_from_matrices(M6, K6)
F = np.array([0, 0, 0, 1.0, 0, 0])
part = res.normalised_participation(F)

print("mode   frequency      participation")
for fr, lab, pp in zip(res.f_hz, res.labels, part):
    print(f"  {fr:8.2f} Hz  {lab:<14s}  {pp:.3e}")
print(f"\nnumber of modes that respond at all: {int(np.sum(part > 1e-8))}")
''')

md(r"""
## 7.7 · Symbol map

Where each thing lives, so you can move between the paper, the primer and the
code without re-deriving anything.

| paper | meaning | primer | `orthomount` |
|---|---|---|---|
| Eq. (1) | $[M]\ddot q+[K]q=f$ | NB 1 | `MountSystem.M`, `.K` |
| Eq. (3) | eigenproblem | NB 2 | `MountSystem.modal()` |
| Eqs. (4)–(5) | $[M]$/$[K]$ orthogonality | NB 3 | verified in `test_orthomount.py` |
| Eq. (6) | modal superposition | NB 3–4 | `modal_response()` |
| $\{u^r\}^\mathsf{T}\{F\}$ | participation factor | NB 4 | `ModalResult.participation()` |
| Eq. (11) | $\{u^2\}=[0,1]^\mathsf{T}$ | NB 4.3 | — |
| Eqs. (13)–(16) | the condition on $R$ | NB 4.5 | `R_from_inertia`, `R_from_stiffness` |
| Eq. (20) | inertia tensor rotation | NB 5.4 | `inertia_tensor_from_principal` |
| Eq. (21) | stiffness rotation | NB 6 | `stiffness_block_from_principal` |
| Eq. (22) | $p,q,\alpha,\beta$ relation | NB 7.3 | `solve_beta_for_R` |
| Eq. (25) | modal inertia | NB 7.4 | `modal_mass_paper` |
| Eq. (27) | $\omega_1$ | NB 7.4 | `omega1_paper`, `K_xi_for_target_f1` |
| Eq. (29) | rocking axis $\delta$ | NB 7.5 | `delta_paper` |
| — | mount → $6\times6$ $[K]$ | NB 6.2 | `Mount.K6()` |
| — | elastic centre | NB 6.4 | `MountSystem.elastic_centre` |

---

## Where to go next

Back to the main walkthrough: **`../orthogonal_engine_mounts.ipynb`**, which
takes this machinery and applies it to the paper's own machine, adds real crank
excitation, damping and a mount-layout solver.
""")

# ###########################################################################

for name, cells in NBS.items():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python",
                              "name": "python3"}
    nbf.write(nb, f"{name}.ipynb")
    print(f"wrote {name}.ipynb  ({len(cells)} cells)")
