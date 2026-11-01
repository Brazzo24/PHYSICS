"""Builds slider_crank_hand_vs_mbs.ipynb.  python build_notebook.py [--execute]"""
import sys
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =====================================================================
md(r"""
# Two routes to the same equation of motion
## Slider-crank: hand derivation vs. object-oriented MBS assembly

**Question.** When you sketch a mechanism, solve its kinematics on paper and
apply Lagrange, you get one compact equation. When you drag *bodies*,
*joints* and *forces* into an MBS tool (Dymola, Simpack, Adams, your own
code), nobody derives anything — yet the solver computes the same motion.
**Where does the hand-derived equation hide inside the assembled model, and
how does the assembly reproduce it?**

This notebook answers that by doing both, side by side, and then turning the
assembled model *into* the hand equation term by term:

| Part | content |
|---|---|
| **A** | Hand derivation: loop closure → positions → kinetic energy → Lagrange |
| **B** | The MBS building blocks: `Body`, `Revolute`, `Prismatic`, force elements, `Model.assemble()` |
| **C** | The bridge: from the assembled DAE back to the hand equation, *using only what the objects provide* |
| **D** | Simulation: integrate both, constraint drift, and what the DAE gives you for free (joint forces) |
| **E** | Extending the model: gravity + a gas spring — two lines vs. re-derivation |
| **F** | Summary, the link to Kane's method / `pycamtrain`, exercises |

Everything is planar, rigid, frictionless. Each part ends with a numeric check
— if a check prints `False` or a large number, something is wrong.

**Key idea in one line.** The hand route eliminates the constraints *before*
writing the dynamics; the MBS route writes the dynamics of free bodies and
lets the constraints act through reaction forces. Projecting the MBS equations
onto the motions the joints allow (`Bᵀ·…`) removes the reaction forces and
gives back the hand equation exactly.
""")

code(r"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import display, Math

sp.init_printing(use_latex='mathjax')
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3,
                     'axes.spines.top': False, 'axes.spines.right': False})

def show(label, expr):
    display(Math(label + ' = ' + sp.latex(expr)))
""")

md(r"""
### The mechanism and its parameters

```
            A (crank pin)
           /\
      r   /  \   l (rod)
         /    \
   O ---θ------B=====[piston]==>  x
 (main bearing)       slides on x-axis
```

* **crank** (body 1): rotates about O, length $r$, CoG at distance $c$ from O, mass $m_1$, inertia $J_1$ about its CoG
* **rod** (body 2): from A to B, length $l$, CoG at distance $b$ from A, $m_2$, $J_2$
* **piston** (body 3): slides on the x-axis, $m_3$ (its rotation is locked; $J_3$ only matters formally)
* **loads**: torque $\tau$ on the crank, force $F_p$ on the piston along $x$ (gas force)

Numbers are loosely those of the reference engine of the *engine design basics*
series (48.5 mm stroke, 105 mm rod) — illustrative, not real data.
""")

code(r"""
r, c, l, b = sp.symbols('r c l b', positive=True)
m1, m2, m3, J1, J2, J3 = sp.symbols('m_1 m_2 m_3 J_1 J_2 J_3', positive=True)
tau, Fp = sp.symbols('tau F_p')

par = {r: 0.02425, c: 0.005, l: 0.105, b: 0.032,
       m1: 2.0, m2: 0.35, m3: 0.30, J1: 1.5e-3, J2: 6e-4, J3: 1e-5,
       tau: 20.0, Fp: -2000.0}          # loads used for the static checks in Part C
""")

# =====================================================================
md(r"""
---
# Part A — The hand derivation (minimal coordinates)

The craft here is *kinematics on paper*: express every position through the
single generalised coordinate $\theta$, so that the constraints are satisfied
**by construction** and never appear again.

### A1. Loop closure
The vector loop O→A→B must end on the x-axis:

$$ y_B = r\sin\theta + l\sin\varphi = 0 \quad\Rightarrow\quad \varphi(\theta) = -\arcsin\!\Big(\frac{r}{l}\sin\theta\Big) $$

This single line *is* the information content of the three joints. In Part B
the same information will be spread over eight constraint equations.
""")

code(r"""
th, thd, thdd = sp.symbols('theta thetadot thetaddot')

phi = -sp.asin(r / l * sp.sin(th))                       # rod angle  (loop closure)
A_pin = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])       # crank pin
G1 = c * sp.Matrix([sp.cos(th), sp.sin(th)])              # crank CoG
G2 = A_pin + b * sp.Matrix([sp.cos(phi), sp.sin(phi)])    # rod CoG
xB = r * sp.cos(th) + l * sp.cos(phi)                     # piston position

show(r'\varphi(\theta)', phi)
show(r'x_B(\theta)', xB)
""")

md(r"""
### A2. Velocities by the chain rule — the *velocity coefficients*
Because everything depends on $\theta$ only, every velocity is
$\dot{(\,\cdot\,)} = \frac{d(\cdot)}{d\theta}\,\dot\theta$.
The factors $d(\cdot)/d\theta$ are called **velocity coefficients** (or
*kinematic transfer functions*). Keep an eye on them — they will reappear in
Part C as the columns of a matrix $\mathbf B$.
""")

code(r"""
show(r'\frac{dx_B}{d\theta}', sp.simplify(xB.diff(th)))
show(r'\frac{d\varphi}{d\theta}', sp.simplify(phi.diff(th)))
""")

md(r"""
### A3. Kinetic energy, body by body
For a planar rigid body: $T_i = \tfrac12 m_i |\mathbf v_{G_i}|^2 + \tfrac12 J_i\dot\varphi_i^2$.
With the velocity coefficients every term is $\propto\dot\theta^2$:

$$ T = \tfrac12\,\underbrace{\Big[\sum_i m_i\Big|\frac{d\mathbf r_{G_i}}{d\theta}\Big|^2 + J_i\Big(\frac{d\varphi_i}{d\theta}\Big)^2\Big]}_{M(\theta)}\;\dot\theta^2 $$

$M(\theta)$ is the **reduced (equivalent) inertia** seen at the crank. We keep
the three body contributions separate, because in Part C each will be matched
with one body object.
""")

code(r"""
half = sp.Rational(1, 2)
dG1, dG2, dxB, dphi = G1.diff(th), G2.diff(th), xB.diff(th), phi.diff(th)

M1_hand = m1 * dG1.dot(dG1) + J1 * 1**2            # crank:   d(phi_1)/d(theta) = 1
M2_hand = m2 * dG2.dot(dG2) + J2 * dphi**2         # rod
M3_hand = m3 * dxB**2                              # piston   (no rotation)
M_hand = M1_hand + M2_hand + M3_hand
T_hand = half * M_hand * thd**2

show(r'M_1', sp.simplify(M1_hand))
show(r'M_3', sp.simplify(M3_hand))
print('M(theta) at theta = 0, 90 deg [kg m^2]:',
      [float(M_hand.subs(par).subs(th, a)) for a in (0, np.pi / 2)])
""")

md(r"""
### A4. Generalised force by virtual work
Only forces that do work in a virtual displacement $\delta\theta$ count:

$$ \delta W = \tau\,\delta\theta + F_p\,\delta x_B = \Big(\tau + F_p\frac{dx_B}{d\theta}\Big)\delta\theta \equiv Q\,\delta\theta $$

The bearing force at O, the guide force on the piston and the pin forces at A
and B do **no** virtual work, so they never appear. That is the big advantage
— and the big limitation — of the hand route: you get the motion, but not the
joint loads.

### A5. Lagrange's equation
$\frac{d}{dt}\frac{\partial T}{\partial\dot\theta} - \frac{\partial T}{\partial\theta} = Q$
with $T=\tfrac12 M(\theta)\dot\theta^2$ gives

$$ \boxed{\,M(\theta)\,\ddot\theta + \tfrac12 M'(\theta)\,\dot\theta^2 = Q(\theta)\,} $$

The middle term is the speed-squared ("centrifugal/Coriolis") term — it exists
because the reduced inertia changes with crank angle.
""")

code(r"""
Q_hand = tau + Fp * dxB
dM_hand = M_hand.diff(th)

# Lagrange, done literally with a time-dependent theta(t) as a cross-check
t = sp.symbols('t'); TH = sp.Function('theta')(t)
Tt = T_hand.subs(thd, TH.diff(t)).subs(th, TH)
lag = (Tt.diff(TH.diff(t)).diff(t) - Tt.diff(TH)).subs(TH.diff(t, 2), thdd).subs(TH.diff(t), thd).subs(TH, th)
boxed = M_hand * thdd + half * dM_hand * thd**2
print('Lagrange == M*thdd + 1/2 M\' thd^2 :', sp.simplify(lag - boxed) == 0)

# numeric functions for later comparisons
P = lambda e, p=par: sp.sympify(e).subs(p)
M_hand_f  = sp.lambdify(th, P(M_hand))
dM_hand_f = sp.lambdify(th, P(dM_hand))
Q_hand_f  = sp.lambdify(th, P(Q_hand))
thdd_hand = lambda a, ad: (Q_hand_f(a) - 0.5 * dM_hand_f(a) * ad**2) / M_hand_f(a)
print('hand: thetaddot(theta=1.1, thetadot=500) =', thdd_hand(1.1, 500.))
""")

# =====================================================================
md(r"""
---
# Part B — The object-oriented MBS assembly (maximal coordinates)

Now forget the sketch. We never solve the loop closure. Instead every
component is a small object that knows only **its own physics**:

| object | owns | contributes to the global system |
|---|---|---|
| `Body` | coordinates $(x, y, \varphi)$ of its CoG | a 3×3 mass block $\mathrm{diag}(m, m, J)$ |
| `Revolute` | two attachment points (body-fixed) | 2 constraint equations: *the points coincide* |
| `PrismaticX` | one body | 2 constraint equations: $y = y_0$, $\varphi = 0$ |
| `Torque`, `PointForce` | where and what | generalised forces $\mathbf Q = (\partial\mathbf r_P/\partial\mathbf q)^T\mathbf f$ |
| `Model` | list of the above | stacks everything: $\mathbf M$, $\boldsymbol\Phi(\mathbf q)$, $\mathbf F$ |

Notice what is *not* in this table: no loop closure, no velocity
coefficients, no reduced inertia. Each object is written once and reused for
any mechanism.

### B1. Bodies
In maximal coordinates the kinetic energy of a free body is trivial and
**constant-coefficient**:
$T_i = \tfrac12\,\dot{\mathbf q}_i^T\,\mathrm{diag}(m_i,m_i,J_i)\,\dot{\mathbf q}_i$.
All the configuration dependence that made $M(\theta)$ complicated will live
in the constraints instead.

A body also needs to tell others where its points are:
$\mathbf r_P = \mathbf r_G + \mathbf R(\varphi)\,\mathbf s_P'$ with $\mathbf s_P'$ the
body-fixed (local) vector from the CoG to P.
""")

code(r"""
class Body:
    def __init__(self, name, m, J):
        self.name, self.m, self.J = name, m, J
        self.x, self.y, self.phi = sp.symbols(f'x_{name} y_{name} varphi_{name}')
        self.q = sp.Matrix([self.x, self.y, self.phi])
        self.M = sp.diag(m, m, J)

    def R(self):
        cph, sph = sp.cos(self.phi), sp.sin(self.phi)
        return sp.Matrix([[cph, -sph], [sph, cph]])

    def point(self, s_local):
        # global position of a body-fixed point (local vector from the CoG)
        return sp.Matrix([self.x, self.y]) + self.R() * sp.Matrix(s_local)

crank  = Body('1', m1, J1)
rod    = Body('2', m2, J2)
piston = Body('3', m3, J3)

# body-fixed points (local x-axis along the part)
s_O, s_A1   = (-c, 0), (r - c, 0)       # on the crank
s_A2, s_B2  = (-b, 0), (l - b, 0)       # on the rod
s_B3        = (0, 0)                    # on the piston
show(r'\mathbf r_{A,\mathrm{crank}}', crank.point(s_A1))
""")

md(r"""
### B2. Joints = constraint equations
A joint is nothing but a statement that must stay true:

* **Revolute** between body $i$ (point $P$) and body $j$ (point $Q$): $\;\boldsymbol\Phi = \mathbf r_P^{(i)} - \mathbf r_Q^{(j)} = \mathbf 0$ — 2 equations. To ground, $\mathbf r_Q$ is a fixed point.
* **Prismatic along global x**: $\;\boldsymbol\Phi = [\,y - y_0,\ \varphi\,]^T = \mathbf 0$ — 2 equations.

Each joint removes as many DOF as it has equations.
""")

code(r"""
class Revolute:
    def __init__(self, name, body_i, s_i, body_j=None, s_j=(0, 0)):
        self.name, self.bi, self.si, self.bj, self.sj = name, body_i, s_i, body_j, s_j
    def Phi(self):
        rP = self.bi.point(self.si)
        rQ = self.bj.point(self.sj) if self.bj is not None else sp.Matrix(self.sj)  # None = ground
        return rP - rQ

class PrismaticX:
    def __init__(self, name, body, y0=0):
        self.name, self.body, self.y0 = name, body, y0
    def Phi(self):
        return sp.Matrix([self.body.y - self.y0, self.body.phi])

joints = [Revolute('O: ground-crank', crank, s_O),
          Revolute('A: crank-rod',    crank, s_A1, rod, s_A2),
          Revolute('B: rod-piston',   rod,   s_B2, piston, s_B3),
          PrismaticX('guide: piston', piston)]
for j in joints:
    show(r'\boldsymbol\Phi_{\mathrm{' + j.name.split(':')[0] + r'}}', j.Phi())
""")

md(r"""
### B3. Force elements = generalised forces by virtual work
Exactly the A4 argument, but done *generically* in the full coordinate set:
a force $\mathbf f$ at point P does virtual work
$\delta W = \mathbf f^T\delta\mathbf r_P = \mathbf f^T\frac{\partial\mathbf r_P}{\partial\mathbf q}\delta\mathbf q$,
so its generalised force is $\mathbf Q = \big(\frac{\partial\mathbf r_P}{\partial\mathbf q}\big)^T\mathbf f$.
A torque on body $i$ gives $\mathbf Q = \big(\frac{\partial\varphi_i}{\partial\mathbf q}\big)^T\tau$.
A potential element (spring, gravity) gives $\mathbf Q = -\partial V/\partial\mathbf q$.
The element never needs to know what the rest of the mechanism looks like.
""")

code(r"""
class Torque:
    def __init__(self, body, T): self.body, self.T = body, T
    def Q(self, q): return sp.Matrix([self.body.phi]).jacobian(q).T * self.T

class PointForce:
    def __init__(self, body, s_local, f): self.body, self.s, self.f = body, s_local, sp.Matrix(f)
    def Q(self, q): return self.body.point(self.s).jacobian(q).T * self.f

class Potential:                       # spring, gravity, ... : Q = -dV/dq
    def __init__(self, V): self.V = V
    def Q(self, q): return -sp.Matrix([self.V]).jacobian(q).T

forces = [Torque(crank, tau), PointForce(piston, s_B3, (Fp, 0))]
""")

md(r"""
### B4. The model assembles itself
`assemble()` only *stacks*: coordinates on top of each other, mass blocks on
the diagonal, constraint equations below each other, force vectors added. Then
it builds the constraint Jacobian $\boldsymbol\Phi_{\mathbf q} = \partial\boldsymbol\Phi/\partial\mathbf q$
and the acceleration right-hand side
$\boldsymbol\gamma = -(\boldsymbol\Phi_{\mathbf q}\dot{\mathbf q})_{\mathbf q}\dot{\mathbf q}$
(from differentiating $\boldsymbol\Phi=\mathbf 0$ twice in time:
$\boldsymbol\Phi_{\mathbf q}\ddot{\mathbf q} = \boldsymbol\gamma$).

The equations of motion follow from Lagrange's equations **with multipliers**
(or equally from Newton–Euler per body plus unknown joint forces):

$$
\boxed{\;
\begin{bmatrix}\mathbf M & \boldsymbol\Phi_{\mathbf q}^T\\ \boldsymbol\Phi_{\mathbf q} & \mathbf 0\end{bmatrix}
\begin{bmatrix}\ddot{\mathbf q}\\ \boldsymbol\lambda\end{bmatrix}
=
\begin{bmatrix}\mathbf F\\ \boldsymbol\gamma\end{bmatrix}\;}
$$

Row 1 is Newton–Euler for each free body, with $-\boldsymbol\Phi_{\mathbf q}^T\boldsymbol\lambda$
the joint reaction forces. Row 2 forces the accelerations to respect the joints.
This is a DAE (differential-algebraic equation) — 9 + 8 = 17 unknowns at each instant.
""")

code(r"""
class Model:
    def __init__(self, bodies, joints, forces):
        self.bodies, self.joints, self.forces = bodies, joints, forces

    def assemble(self):
        self.q = sp.Matrix.vstack(*[B.q for B in self.bodies])
        self.M = sp.diag(*[B.M for B in self.bodies])                 # block diagonal, constant
        self.Phi = sp.Matrix.vstack(*[j.Phi() for j in self.joints])
        self.Phiq = self.Phi.jacobian(self.q)
        self.qd = sp.Matrix(sp.symbols(f'qd0:{len(self.q)}'))
        self.gamma = -(self.Phiq * self.qd).jacobian(self.q) * self.qd
        self.F = sum((f.Q(self.q) for f in self.forces), sp.zeros(len(self.q), 1))
        self.rows = []                                                # bookkeeping for plots
        i = 0
        for j in self.joints:
            n = j.Phi().shape[0]; self.rows.append((j.name, i, i + n)); i += n
        return self

mdl = Model([crank, rod, piston], joints, forces).assemble()
n_q, n_c = len(mdl.q), len(mdl.Phi)
print(f'coordinates n = {n_q},  constraints m = {n_c},  DOF = n - m = {n_q - n_c}   (Gruebler)')
show(r'\mathbf q^T', mdl.q.T)
show(r'\mathbf F^T', mdl.F.T)
""")

md(r"""
### B5. Look at the structure
Each joint fills only the columns of the bodies it connects — the global
matrices are **sparse and block-structured**. That locality is exactly why
the method scales: adding a body adds 3 columns; adding a joint adds a few
rows; nothing that already exists changes.
""")

code(r"""
pattern = np.array([[0 if e == 0 else 1 for e in row] for row in mdl.Phiq.tolist()], float)
img = np.full(pattern.shape, np.nan)
for k, (name, i0, i1) in enumerate(mdl.rows):
    img[i0:i1][pattern[i0:i1] > 0] = k

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.imshow(img, cmap='tab10', vmin=0, vmax=9)
ax.set_xticks(range(n_q)); ax.set_xticklabels([f'${sp.latex(s)}$' for s in mdl.q])
ax.set_yticks([(i0 + i1 - 1) / 2 for _, i0, i1 in mdl.rows]); ax.set_yticklabels([n for n, *_ in mdl.rows])
for x in (2.5, 5.5): ax.axvline(x, color='k', lw=.8)
for _, i0, i1 in mdl.rows[:-1]: ax.axhline(i1 - .5, color='k', lw=.4)
ax.grid(False); ax.set_title(r'non-zero pattern of $\Phi_q$  (columns: crank | rod | piston)')
plt.tight_layout(); plt.show()
show(r'\boldsymbol\Phi_{\mathbf q}', mdl.Phiq)
""")

# =====================================================================
md(r"""
---
# Part C — The bridge: from the assembled DAE back to the hand equation

Now the central question. We have two descriptions:

* **hand:** $M(\theta)\ddot\theta + \tfrac12M'(\theta)\dot\theta^2 = Q(\theta)$ — 1 equation
* **MBS:** $\mathbf M\ddot{\mathbf q} + \boldsymbol\Phi_{\mathbf q}^T\boldsymbol\lambda = \mathbf F,\;\boldsymbol\Phi_{\mathbf q}\ddot{\mathbf q} = \boldsymbol\gamma$ — 17 equations

We show that the second contains the first, and — importantly — that we can
**extract it using only the assembled matrices**, i.e. without ever doing the
hand kinematics. This is what MBS solvers based on coordinate partitioning,
null-space projection or Kane's method do internally.

The roadmap:

| step | MBS operation (on assembled objects) | reproduces in the hand route |
|---|---|---|
| C1 | solve $\boldsymbol\Phi(\mathbf q)=\mathbf 0$ for given $\theta$ (Newton) | A1 loop closure |
| C2 | velocity basis $\mathbf B$ from $\boldsymbol\Phi_{\mathbf q}\mathbf B = \mathbf 0$ | A2 velocity coefficients |
| C3 | $\mathbf B^T\mathbf M\mathbf B$ | A3 reduced inertia $M(\theta)$ |
| C4 | $\mathbf B^T\mathbf M\,\mathbf a_p$ with $\boldsymbol\Phi_{\mathbf q}\mathbf a_p = \boldsymbol\gamma$ | A5 term $\tfrac12M'\dot\theta^2$ |
| C5 | $\mathbf B^T\mathbf F$, and $\mathbf B^T\boldsymbol\Phi_{\mathbf q}^T = \mathbf 0$ | A4 virtual work $Q$, reactions drop out |

First, turn the assembled symbolic matrices into fast numeric functions.
""")

code(r"""
def numeric(model, p):
    # lambdify the assembled matrices for a given parameter set
    q, qd = list(model.q), list(model.qd)
    L = lambda e, args: sp.lambdify(args, P(e, p), 'numpy')
    f = dict(Phi=L(model.Phi, q), Phiq=L(model.Phiq, q),
             gamma=L(model.gamma, q + qd), F=L(model.F, q))
    f['M'] = np.array(P(model.M, p), float)
    return f

N = numeric(mdl, par)
iu = 2                                    # independent coordinate: crank angle varphi_1 = theta
iv = [i for i in range(n_q) if i != iu]   # dependent coordinates
""")

md(r"""
### C1. Position level: the solver does the loop closure numerically
Fix the independent coordinate $\varphi_1=\theta$ and solve the 8 nonlinear
equations $\boldsymbol\Phi(\mathbf q)=\mathbf 0$ for the 8 dependent ones with Newton
($\boldsymbol\Phi_{\mathbf q,v}\,\Delta\mathbf q_v = -\boldsymbol\Phi$). This is
"assembly" in Dymola/Adams language. Compare with the closed-form hand result.
""")

code(r"""
def solve_position(theta, q_guess):
    q = np.array(q_guess, float); q[iu] = theta
    for _ in range(30):
        res = N['Phi'](*q).ravel()
        dq = np.linalg.solve(N['Phiq'](*q)[:, iv], -res)
        q[iv] += dq
        if np.linalg.norm(dq) < 1e-14: break
    return q

# hand-route coordinates in the same ordering, for comparison only
q_hand_sym = sp.Matrix([G1[0], G1[1], th, G2[0], G2[1], phi, xB, 0, 0])
q_hand_f = sp.lambdify(th, P(q_hand_sym))

pv = par
rough = [P(c, pv), 0, 0, P(r + b, pv), 0, 0, P(r + l, pv), 0, 0]   # crude 'drawing' of the layout
q_guess = rough
print(' theta  | rod angle MBS    hand        | x_B MBS        hand')
for a in np.linspace(0, 2 * np.pi, 7)[:-1] + 0.2:
    q_guess = solve_position(a, q_guess)                  # continuation keeps the right branch
    qh = np.array(q_hand_f(a), float).ravel()
    print(f'{a:6.2f}  | {q_guess[5]: .10f}  {qh[5]: .10f} | {q_guess[6]:.10f}  {qh[6]:.10f}')
""")

md(r"""
### C2. Velocity level: the velocity coefficients are the null space of $\boldsymbol\Phi_{\mathbf q}$
Differentiating $\boldsymbol\Phi(\mathbf q)=\mathbf 0$ once: $\boldsymbol\Phi_{\mathbf q}\dot{\mathbf q} = \mathbf 0$.
Split $\dot{\mathbf q}$ into independent $\dot\theta$ and dependent $\dot{\mathbf q}_v$:

$$ \boldsymbol\Phi_{\mathbf q,v}\,\dot{\mathbf q}_v + \boldsymbol\Phi_{\mathbf q,u}\,\dot\theta = \mathbf 0
\;\Rightarrow\;
\dot{\mathbf q} = \mathbf B\,\dot\theta,\qquad
\mathbf B = \begin{bmatrix}-\boldsymbol\Phi_{\mathbf q,v}^{-1}\boldsymbol\Phi_{\mathbf q,u}\\ 1\end{bmatrix}\ \text{(re-ordered)} $$

$\mathbf B$ (9×1 here, $n\times\text{DOF}$ in general) lists how fast every
coordinate moves per unit crank speed — i.e. **exactly the hand velocity
coefficients** $d\mathbf q/d\theta$ of A2. Its column spans the null space of
$\boldsymbol\Phi_{\mathbf q}$: every motion the joints allow.

In Kane's method these are the **partial velocities**.
""")

code(r"""
def B_mbs(q):
    A = N['Phiq'](*q)
    B = np.zeros(n_q); B[iu] = 1.0
    B[iv] = -np.linalg.solve(A[:, iv], A[:, iu])
    return B

B_hand_sym = q_hand_sym.diff(th)                   # hand velocity coefficients d q / d theta
B_hand_f = sp.lambdify(th, P(B_hand_sym))

a = 1.1
q_a = solve_position(a, q_guess)
Bm, Bh = B_mbs(q_a), np.array(B_hand_f(a), float).ravel()
print('coordinate     B from Phi_q (MBS)    dq/dtheta (hand)')
for s, x1, x2 in zip(mdl.q, Bm, Bh):
    print(f'{str(s):12s} {x1: .12f}   {x2: .12f}')
print('max |difference| =', np.abs(Bm - Bh).max(), '   ||Phi_q B|| =', np.linalg.norm(N['Phiq'](*q_a) @ Bm))
""")

md(r"""
### C3. Mass: $\mathbf B^T\mathbf M\mathbf B = M(\theta)$ — body by body
Insert $\dot{\mathbf q}=\mathbf B\dot\theta$ into the trivial maximal-coordinate kinetic energy:

$$ T = \tfrac12\dot{\mathbf q}^T\mathbf M\dot{\mathbf q} = \tfrac12\,(\mathbf B^T\mathbf M\mathbf B)\,\dot\theta^2
\quad\Rightarrow\quad M(\theta) = \mathbf B^T\mathbf M\mathbf B = \sum_{i}\mathbf B_i^T\,\mathrm{diag}(m_i,m_i,J_i)\,\mathbf B_i $$

With $\mathbf B_i = [\,dx_{G_i}/d\theta,\ dy_{G_i}/d\theta,\ d\varphi_i/d\theta\,]^T$ each body block gives
$m_i|d\mathbf r_{G_i}/d\theta|^2 + J_i(d\varphi_i/d\theta)^2$ — **literally the term written
by hand in A3** for that body. The projection does the bookkeeping that you
did with pencil. We check it symbolically, body by body.
""")

code(r"""
blocks = {'crank': slice(0, 3), 'rod': slice(3, 6), 'piston': slice(6, 9)}
hand_terms = {'crank': M1_hand, 'rod': M2_hand, 'piston': M3_hand}
for name, s in blocks.items():
    Bi = B_hand_sym[s, :]
    proj = (Bi.T * mdl.M[s, s] * Bi)[0]
    print(f'{name:7s}:  B_i^T M_i B_i - (hand term) = ', sp.simplify(proj - hand_terms[name]))

# and numerically with B obtained purely from the assembled Phi_q:
print('\nnumeric, B from Phi_q:  B^T M B =', Bm @ N['M'] @ Bm, '   hand M(theta) =', M_hand_f(a))
""")

md(r"""
### C4. The speed-squared term from $\boldsymbol\gamma$
Differentiate $\dot{\mathbf q}=\mathbf B\dot\theta$ once more:
$\ddot{\mathbf q} = \mathbf B\ddot\theta + \mathbf a_p$, with $\mathbf a_p = \dot{\mathbf B}\dot\theta = \frac{d\mathbf B}{d\theta}\dot\theta^2$
— the acceleration the mechanism has even at $\ddot\theta = 0$ (centripetal
accelerations etc.).

The MBS model never differentiates $\mathbf B$. It gets $\mathbf a_p$ from the
second-derivative constraint $\boldsymbol\Phi_{\mathbf q}\ddot{\mathbf q}=\boldsymbol\gamma$: with
$\ddot\theta=0$ set, $\boldsymbol\Phi_{\mathbf q,v}\,\mathbf a_{p,v} = \boldsymbol\gamma$.

Projecting the inertia forces of this acceleration:

$$ \mathbf B^T\mathbf M\,\mathbf a_p = \mathbf B^T\mathbf M\frac{d\mathbf B}{d\theta}\,\dot\theta^2
= \tfrac12\frac{d}{d\theta}\big(\mathbf B^T\mathbf M\mathbf B\big)\dot\theta^2 = \tfrac12 M'(\theta)\,\dot\theta^2 $$

(the middle step uses that $\mathbf M$ is constant and symmetric). That is the
second term of the hand equation.
""")

code(r"""
def a_p_mbs(q, qd):
    A = N['Phiq'](*q); g = N['gamma'](*q, *qd).ravel()
    ap = np.zeros(n_q); ap[iv] = np.linalg.solve(A[:, iv], g)
    return ap

ad = 500.0                                          # rad/s  (~4800 rpm)
qd_a = Bm * ad
ap = a_p_mbs(q_a, qd_a)
dB_hand_f = sp.lambdify(th, P(B_hand_sym.diff(th)))
print('a_p from gamma  vs  dB/dtheta * thetadot^2 (hand):  max diff =',
      np.abs(ap - np.array(dB_hand_f(a), float).ravel() * ad**2).max(), ' (|a_p| ~', np.abs(ap).max(), ')')
print('B^T M a_p =', Bm @ N['M'] @ ap, '    1/2 M\'(theta) thetadot^2 (hand) =', 0.5 * dM_hand_f(a) * ad**2)

# symbolic identity  B^T M dB/dtheta == 1/2 dM/dtheta
lhs = (B_hand_sym.T * mdl.M * B_hand_sym.diff(th))[0]
print('symbolic:  B^T M dB/dtheta - 1/2 M\'(theta) =', sp.simplify(lhs - half * dM_hand))
""")

md(r"""
### C5. Forces: the applied loads project to $Q$, the joint forces vanish
Project the full force balance $\mathbf M\ddot{\mathbf q} = \mathbf F - \boldsymbol\Phi_{\mathbf q}^T\boldsymbol\lambda$ with $\mathbf B^T$:

* $\mathbf B^T\mathbf F = \tau\cdot 1 + F_p\cdot\frac{dx_B}{d\theta} = Q$ — the virtual work of A4, now computed by a matrix product.
* $\mathbf B^T\boldsymbol\Phi_{\mathbf q}^T\boldsymbol\lambda = (\boldsymbol\Phi_{\mathbf q}\mathbf B)^T\boldsymbol\lambda = \mathbf 0$ — **the joint reactions do no work on admissible motions**, so they drop out. This is d'Alembert's principle in matrix form, and it is why the hand route never sees them.
""")

code(r"""
Fm = N['F'](*q_a).ravel()
print('B^T F =', Bm @ Fm, '    hand Q(theta) =', Q_hand_f(a))
print('B^T Phi_q^T =', Bm @ N['Phiq'](*q_a).T, ' -> all zero: reactions drop out')
print('symbolic B^T F - Q =', sp.simplify((B_hand_sym.T * mdl.F.subs(dict(zip(mdl.q, q_hand_sym))))[0] - Q_hand))
""")

md(r"""
### C6. Putting it together — three ways to the same $\ddot\theta$
1. **hand**: $\ddot\theta = (Q - \tfrac12M'\dot\theta^2)/M$
2. **MBS, reduced** (C1–C5, from the assembled objects only): $\ddot\theta = \mathbf B^T(\mathbf F - \mathbf M\mathbf a_p)\,/\,\mathbf B^T\mathbf M\mathbf B$
3. **MBS, full DAE**: solve the 17×17 saddle-point system and read off $\ddot q_{\varphi_1}$
""")

code(r"""
def thdd_reduced(q, qd):
    B, ap, F = B_mbs(q), a_p_mbs(q, qd), N['F'](*q).ravel()
    return B @ (F - N['M'] @ ap) / (B @ N['M'] @ B)

def solve_dae(q, qd, f=None):
    A = N['Phiq'](*q); g = N['gamma'](*q, *qd).ravel()
    F = N['F'](*q).ravel() if f is None else f
    K = np.block([[N['M'], A.T], [A, np.zeros((n_c, n_c))]])
    x = np.linalg.solve(K, np.r_[F, g])
    return x[:n_q], x[n_q:]                            # qdd, lambda

print(' theta  thetadot |        hand          MBS reduced         MBS full DAE   | max diff')
qg = rough
for a_, ad_ in [(0.3, 100.), (1.1, 500.), (2.5, -800.), (4.0, 1500.), (5.5, 300.)]:
    qg = solve_position(a_, qg); qd_ = B_mbs(qg) * ad_
    r1, r2, r3 = thdd_hand(a_, ad_), thdd_reduced(qg, qd_), solve_dae(qg, qd_)[0][iu]
    print(f'{a_:5.1f} {ad_:8.0f}  | {r1:16.6f} {r2:16.6f} {r3:16.6f}  | {max(abs(r1-r2), abs(r1-r3)):.1e}')
""")

md(r"""
**What just happened.** Route 2 never used the sketch: it only called
`Phi`, `Phiq`, `gamma`, `M`, `F` — the outputs of the objects. Yet each
intermediate quantity equals its hand counterpart (C1–C5), so the result is the
hand equation, evaluated numerically instead of written in closed form.

So the two methods are not two different physics. They differ in **when the
constraints are eliminated**:

* hand: *before* writing the dynamics (analytically, once, by you)
* MBS full DAE: *never* — they act via $\boldsymbol\lambda$
* MBS reduced: *after* assembly (numerically, at every time step, by the code)
""")

# =====================================================================
md(r"""
---
# Part D — Simulation: same motion, plus joint forces for free

Scenario: the engine coasts freely ($\tau = 0$, $F_p = 0$) from 9500 rpm for
20 ms (~3 revolutions). The crank speed fluctuates twice per revolution
because $M(\theta)$ does — energy $\tfrac12M\dot\theta^2$ is conserved.

* **hand ODE**: 2 states $(\theta,\dot\theta)$
* **MBS DAE**: 18 states $(\mathbf q,\dot{\mathbf q})$, at each step solve the 17×17 system.
  Because only $\ddot{\mathbf q}$ is constrained, numerical errors let $\boldsymbol\Phi$
  drift away from zero. **Baumgarte stabilisation** replaces $\boldsymbol\gamma$ by
  $\boldsymbol\gamma - 2\alpha\boldsymbol\Phi_{\mathbf q}\dot{\mathbf q} - \beta^2\boldsymbol\Phi$ to pull it back.
""")

code(r"""
coast = dict(par); coast[tau] = 0.0; coast[Fp] = 0.0
N = numeric(mdl, coast)
M_hand_f, dM_hand_f, Q_hand_f = (sp.lambdify(th, P(e, coast)) for e in (M_hand, dM_hand, Q_hand))
thdd_hand = lambda a, ad: (Q_hand_f(a) - 0.5 * dM_hand_f(a) * ad**2) / M_hand_f(a)

w0, t_end = 9500 * 2 * np.pi / 60, 0.02
t_eval = np.linspace(0, t_end, 2001)
opts = dict(method='DOP853', rtol=1e-10, atol=1e-12, t_eval=t_eval)

sol_h = solve_ivp(lambda t, z: [z[1], thdd_hand(z[0], z[1])], (0, t_end), [0.0, w0], **opts)

def dae_rhs(t, z, alpha=0.0, beta=0.0):
    q, qd = z[:n_q], z[n_q:]
    A = N['Phiq'](*q)
    g = N['gamma'](*q, *qd).ravel() - 2 * alpha * (A @ qd) - beta**2 * N['Phi'](*q).ravel()
    K = np.block([[N['M'], A.T], [A, np.zeros((n_c, n_c))]])
    return np.r_[qd, np.linalg.solve(K, np.r_[N['F'](*q).ravel(), g])[:n_q]]

q0 = solve_position(0.0, rough); z0 = np.r_[q0, B_mbs(q0) * w0]
opts_dae = dict(opts, rtol=1e-8, atol=1e-10)
sol_baum = solve_ivp(lambda t, z: dae_rhs(t, z, 300., 300.), (0, t_end), z0, **opts_dae)
drift = lambda s: [np.linalg.norm(N['Phi'](*s.y[:n_q, k])) for k in range(s.y.shape[1])]

# drift demo: everyday tolerance (rtol 1e-5), 0.2 s ~ 30 revolutions
t_long = np.linspace(0, 0.2, 401)
loose = dict(method='RK45', rtol=1e-5, atol=1e-7, t_eval=t_long)
sol_raw_l  = solve_ivp(dae_rhs, (0, 0.2), z0, **loose)
sol_baum_l = solve_ivp(lambda t, z: dae_rhs(t, z, 300., 300.), (0, 0.2), z0, **loose)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
rpm = 60 / (2 * np.pi)
ax[0].plot(np.degrees(sol_h.y[0]), sol_h.y[1] * rpm, lw=2.5, label='hand ODE (2 states)')
ax[0].plot(np.degrees(sol_baum.y[iu]), sol_baum.y[n_q + iu] * rpm, '--', label='MBS DAE (18 states)')
ax[0].set(xlabel='crank angle [deg]', ylabel='crank speed [rpm]', title='same motion'); ax[0].legend()
ax[1].semilogy(t_eval * 1e3, np.abs(sol_baum.y[iu] - sol_h.y[0]) + 1e-18)
ax[1].set(xlabel='t [ms]', ylabel='|theta_DAE - theta_hand| [rad]', title='difference (integration tolerance)')
ax[2].semilogy(t_long * 1e3, np.array(drift(sol_raw_l)) + 1e-18, label='no stabilisation')
ax[2].semilogy(t_long * 1e3, np.array(drift(sol_baum_l)) + 1e-18, label='Baumgarte (alpha=beta=300)')
ax[2].set(xlabel='t [ms]', ylabel='||Phi(q)|| [m]', title='constraint drift, rtol 1e-5, 0.2 s'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

E = lambda s: [0.5 * s.y[n_q:, k] @ N['M'] @ s.y[n_q:, k] for k in range(s.y.shape[1])]
Eh = 0.5 * M_hand_f(sol_h.y[0]) * sol_h.y[1]**2
print(f'energy: hand rel. change {np.ptp(Eh)/Eh[0]:.1e},   MBS (Baumgarte) rel. change {np.ptp(E(sol_baum))/Eh[0]:.1e}')
print(f'speed fluctuation: {sol_h.y[1].min()*rpm:.0f} ... {sol_h.y[1].max()*rpm:.0f} rpm')
print(f'drift after 0.2 s at rtol 1e-5:  unstabilised {drift(sol_raw_l)[-1]*1e3:.2f} mm,  Baumgarte {drift(sol_baum_l)[-1]*1e6:.2f} um')
""")

md(r"""
**Reading the plots.** Left: the two routes give the same crank speed curve
(the dip twice per revolution is $M(\theta)$ rising when rod and piston move fast).
Middle: the difference is at integration-tolerance level. Right: with ordinary
tolerances, the unstabilised DAE lets the pin joints come apart by millimetres
within 30 revolutions — the hand ODE cannot have this problem because the
constraints are not equations there, they are built into $\theta$.
""")

md(r"""
### D2. What the DAE gives you for free: the joint forces $\boldsymbol\lambda$
In the hand route the joint forces vanished in C5. In the DAE they are part of
the solution: the force that joint $k$ applies to the bodies is
$-\boldsymbol\Phi_{\mathbf q,k}^T\boldsymbol\lambda_k$. For a revolute joint the
Jacobian block on the first body is the identity, so $-\boldsymbol\lambda_k$ is
directly the pin force (in global x, y) on that body.

**Independent check.** Newton for the whole mechanism: the sum of external
forces (bearing O, piston guide, applied loads) must equal
$\sum_i m_i\,\mathbf a_{G_i}$. We take the CoG accelerations from the *hand*
kinematics ($\mathbf a_G = \frac{d^2\mathbf r_G}{d\theta^2}\dot\theta^2 + \frac{d\mathbf r_G}{d\theta}\ddot\theta$)
and the forces from the *MBS* $\boldsymbol\lambda$ — two different routes again.
""")

code(r"""
d2q_hand_f = sp.lambdify(th, P(B_hand_sym.diff(th), coast))
masses = np.array([coast[m1], coast[m2], coast[m3]])
ang, F_O, N_guide, F_A, resid = [], [], [], [], []
for k in range(0, len(t_eval), 4):
    q, qd = sol_baum.y[:n_q, k], sol_baum.y[n_q:, k]
    qdd, lam = solve_dae(q, qd)
    a_, ad_ = sol_h.y[0, k], sol_h.y[1, k]
    add_ = thdd_hand(a_, ad_)
    acc = np.array(d2q_hand_f(a_), float).ravel() * ad_**2 + np.array(B_hand_f(a_), float).ravel() * add_
    sum_ma = sum(masses[i] * acc[3 * i:3 * i + 2] for i in range(3))
    ext = np.array([-lam[0], -lam[1]]) + np.array([0, -lam[6]]) + np.array([coast[Fp], 0])
    ang.append(np.degrees(q[iu]) % 360); F_O.append(-lam[0:2]); N_guide.append(-lam[6]); F_A.append(-lam[2:4])
    resid.append(np.linalg.norm(sum_ma - ext) / np.linalg.norm(sum_ma))
ang, F_O, F_A = np.array(ang), np.array(F_O), np.array(F_A)
o = np.argsort(ang[:len(ang)//3])  # first revolution only

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(ang[o], np.linalg.norm(F_A, axis=1)[o] / 1e3, label='|crank-pin force A|')
ax[0].plot(ang[o], np.linalg.norm(F_O, axis=1)[o] / 1e3, label='|main bearing force O|')
ax[0].plot(ang[o], np.array(N_guide)[o] / 1e3, label='piston side force (guide)')
ax[0].set(xlabel='crank angle [deg]', ylabel='force [kN]', title='joint forces from lambda (inertia only, ~9500 rpm, no counterweight)'); ax[0].legend(fontsize=8)
ax[1].semilogy(ang, np.array(resid) + 1e-18, '.', ms=3)
ax[1].set(xlabel='crank angle [deg]', ylabel='relative residual', title='Newton check:  sum(m a_G) vs  external forces from lambda')
plt.tight_layout(); plt.show()
print('max relative residual:', max(resid))
""")

# =====================================================================
md(r"""
---
# Part E — Extending the model: where the two routes really differ

Add **gravity** (engine upright, $-y$) and a **gas spring** pushing on the
piston (a linear spring from a fixed point above TDC — a crude stand-in for
cylinder pressure).

**MBS route** — append two force objects, re-assemble. Nothing else changes:
""")

code(r"""
g, k_s, L0 = sp.symbols('g k_s L_0', positive=True)
anchor = sp.Matrix([r + l + 0.05, 0])                 # spring anchor, beyond TDC
d = piston.point(s_B3) - anchor
V_spring = half * k_s * (sp.sqrt(d.dot(d)) - L0)**2
V_grav = g * (m1 * crank.y + m2 * rod.y + m3 * piston.y)

mdl2 = Model([crank, rod, piston], joints,
             forces + [Potential(V_spring), Potential(V_grav)]).assemble()      # <- the whole change
""")

md(r"""
**Hand route** — the new forces are conservative, so add their potential
$V(\theta)$ and $Q \to Q - dV/d\theta$. That means writing $y_{G_i}(\theta)$ and the
spring length as functions of $\theta$ — again by hand. For this example it is
short; for a different topology (a second cylinder, a balance shaft, a flexible
bearing) you start over from the loop closure.
""")

code(r"""
xA = anchor[0]
V_hand = half * k_s * (sp.sqrt((xB - xA)**2) - L0)**2 + g * (m1 * G1[1] + m2 * G2[1] + m3 * 0)
Q2_hand = Q_hand - V_hand.diff(th)

p2 = dict(par); p2.update({g: 9.81, k_s: 2.0e5, L0: 0.10})
M_f, dM_f, Q2_f = (sp.lambdify(th, P(e, p2)) for e in (M_hand, dM_hand, Q2_hand))
N = numeric(mdl2, p2)                              # thdd_reduced / solve_dae now use model 2

print(' theta  thetadot |        hand          MBS reduced        MBS full DAE')
qg = rough
for a_, ad_ in [(0.3, 0.), (1.1, 300.), (2.5, -600.), (4.0, 1200.)]:
    qg = solve_position(a_, qg); qd_ = B_mbs(qg) * ad_
    h = (Q2_f(a_) - 0.5 * dM_f(a_) * ad_**2) / M_f(a_)
    print(f'{a_:5.1f} {ad_:8.0f}  | {h:16.6f} {thdd_reduced(qg, qd_):16.6f} {solve_dae(qg, qd_)[0][iu]:16.6f}')
""")

# =====================================================================
md(r"""
---
# Part F — Summary

| | hand / minimal coordinates | MBS / maximal coordinates (DAE) |
|---|---|---|
| unknowns | $\theta$ (DOF only) | all body coordinates $\mathbf q$ + multipliers $\boldsymbol\lambda$ |
| kinematics | solved analytically by you (loop closure) | stated per joint as $\boldsymbol\Phi(\mathbf q)=\mathbf 0$, solved numerically |
| mass matrix | $M(\theta)$, configuration dependent, derived | $\mathbf M$ constant, block diagonal, just stacked |
| velocity terms | $\tfrac12M'\dot\theta^2$, derived | hidden in $\boldsymbol\gamma$, generated by differentiating $\boldsymbol\Phi$ |
| applied forces | virtual work along $\theta$ | $(\partial\mathbf r_P/\partial\mathbf q)^T\mathbf f$ per element |
| joint forces | gone (no virtual work) | $\boldsymbol\lambda$ — part of the solution |
| new component | re-derive | add an object |
| numerics | small ODE, no drift | larger, sparse DAE; needs index reduction / stabilisation |

**The bridge in one equation.** With $\mathbf B$ spanning the null space of $\boldsymbol\Phi_{\mathbf q}$ ($\dot{\mathbf q} = \mathbf B\dot\theta$):

$$
\mathbf B^T\Big[\mathbf M\ddot{\mathbf q} + \boldsymbol\Phi_{\mathbf q}^T\boldsymbol\lambda - \mathbf F\Big] = 0
\;\;\Longrightarrow\;\;
\underbrace{\mathbf B^T\mathbf M\mathbf B}_{M(\theta)}\ddot\theta
+ \underbrace{\mathbf B^T\mathbf M\dot{\mathbf B}\dot\theta}_{\frac12M'\dot\theta^2}
= \underbrace{\mathbf B^T\mathbf F}_{Q}
\quad\big(\mathbf B^T\boldsymbol\Phi_{\mathbf q}^T = \mathbf 0\big)
$$

### Where real tools sit
* **Adams, Simpack (partly), RecurDyn**: maximal (absolute) coordinates + DAE, as in Part B/D.
* **Dymola/Modelica**: assemble objects acausally, then *symbolically* eliminate and
  index-reduce at translation time — close to doing C1–C5 symbolically, which is why
  the generated code looks like a minimal-coordinate model. When it fails, you see
  "structural singularity" or nonlinear systems in the loop closure (= our C1 Newton).
* **Simpack, recursive/relative-coordinate codes**: joint coordinates for tree structures
  (constraints satisfied by construction, like the hand route), cut joints with $\boldsymbol\lambda$ only where loops close.
* **Kane's method / `sympy.physics.mechanics` (your `pycamtrain`)**: builds $\mathbf B$
  (*partial velocities*) per body from the object description and forms
  $\sum_i \mathbf B_i^T(\mathbf F_i - \mathbf M_i\mathbf a_i) = 0$ directly — the C-route done
  symbolically, so you get the compact hand equation automatically. That is why
  `pycamtrain` needs no constraint stabilisation, and why prescribed motion is
  "a coordinate that was never created".

### Exercises
1. **Desaxé piston** (pin offset $e$): change `PrismaticX(..., y0=e)`. Re-run Part C. Then
   redo A1 by hand ($\sin\varphi = (e - r\sin\theta)/l$) and see how much of Part A you must touch.
2. **Choose the piston as independent coordinate** (`iu = 6`). Evaluate `B_mbs` near TDC.
   What happens to $\boldsymbol\Phi_{\mathbf q,v}$, and why do MBS codes switch independent
   coordinates during a run?
3. **Flexible main bearing**: replace `Revolute('O', ...)` by two linear springs (a
   `Potential` with stiffness 1e8 N/m in x and y). How many DOF now? Compare the
   bearing force with the $\boldsymbol\lambda_O$ from Part D, and look at the step size the
   integrator needs.
4. **Second cylinder** at 90° bank angle on the same crank pin: count how many lines you
   add in the MBS route vs. how much of the hand derivation changes.
5. Show by hand that $\mathbf B^T\mathbf M\,d\mathbf B/d\theta = \tfrac12\,d(\mathbf B^T\mathbf M\mathbf B)/d\theta$
   requires $\mathbf M$ to be constant — and think about what happens in a code that uses
   body-fixed rotational coordinates in 3D, where it is not.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = 'slider_crank_hand_vs_mbs.ipynb'
if '--execute' in sys.argv:
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=900, kernel_name='python3').execute()
nbf.write(nb, out)
print('written', out)
