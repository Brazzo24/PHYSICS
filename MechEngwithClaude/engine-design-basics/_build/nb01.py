from common import NB, HEAD

nb = NB("01_crank_kinematics")

nb.md(r"""
# 1 · Crank-slider kinematics

**Engine design basics — notebook 1.**

Every load in the crank drive starts here. The gas force is applied to a piston
whose *position* sets the cylinder volume, and the inertia force is the piston's
*acceleration* times its mass. So before pistons, rods, crankshafts or bearings,
we need $x(\theta)$, $\dot x(\theta)$ and $\ddot x(\theta)$ — exactly, and in the
familiar series form that explains first- and second-order forces.

**You should come out knowing:**
- the exact piston motion and the 2nd-order approximation, and how big the error is;
- why $\lambda = r/l$ is the single most important kinematic number;
- the orders contained in piston acceleration (1, 2, 4, 6 … but no odd ones above 1);
- the magnitudes in a high-revving race engine: mean piston speed, peak acceleration in *g*.
""")

nb.code(HEAD)

nb.md(r"""
## 1.1 · The reference engine

All notebooks share one *illustrative* engine: 1000 cc, four cylinders, very
over-square, high-revving. The numbers are round and plausible — they are not the
data of any real engine. Swap them for your own in `ec.Engine(...)` whenever you
want.
""")

nb.code(r'''
eng = ec.Engine()
print(eng.summary())
''')

nb.md(r"""
## 1.2 · Geometry

Crank radius $r = s/2$, rod length $l$ (centre to centre), crank angle $\theta$
from TDC, rod angle $\beta$. The piston pin lies on the cylinder axis, the crank
pin on a circle of radius $r$. The horizontal offset of the crank pin is shared by
both links:

$$r\sin\theta = l\sin\beta \quad\Rightarrow\quad \sin\beta = \lambda\sin\theta,
\qquad \lambda = \frac{r}{l}$$

The distance from crank centre to piston pin is $r\cos\theta + l\cos\beta$; at TDC
it is $r + l$. Hence the piston travel from TDC:

$$\boxed{x(\theta) = r(1-\cos\theta) + l\left(1-\sqrt{1-\lambda^2\sin^2\theta}\right)}$$
""")

nb.code(r'''
fig, axs = plt.subplots(1, 4, figsize=(9, 3.6))
for ax, deg in zip(axs, (0, 60, 90, 180)):
    ec.draw_crank_slider(ax, eng, np.radians(deg))
plt.tight_layout(); plt.show()
''')

nb.md(r"""
Notice at $\theta = 90°$: the piston is **not** at half stroke. The rod is inclined,
so the piston has already travelled further than $r$. That asymmetry between the
top and bottom halves of the stroke is the whole origin of second-order effects.
""")

nb.code(r'''
x90, _, _ = ec.piston_kinematics(eng, np.pi / 2)
print(f"x(90°) = {x90*1e3:.2f} mm   vs   r = {eng.r*1e3:.2f} mm")
print(f"Extra travel l(1-cos beta) = {(x90-eng.r)*1e3:.2f} mm  ≈ r·λ/2 = {eng.r*eng.lam/2*1e3:.2f} mm")
''')

nb.md(r"""
## 1.3 · Velocity and acceleration

At constant crank speed $\omega$, $\dot x = \omega\,dx/d\theta$ and
$\ddot x = \omega^2\,d^2x/d\theta^2$. With $s=\sin\theta$, $c=\cos\theta$,
$D=\sqrt{1-\lambda^2 s^2}=\cos\beta$:

$$\frac{dx}{d\theta} = r\left(s + \frac{\lambda\,s\,c}{D}\right)
\qquad
\frac{d^2x}{d\theta^2} = r\left(c + \lambda\left[\frac{c^2-s^2}{D}
+ \frac{\lambda^2 s^2c^2}{D^3}\right]\right)$$

Expanding $\sqrt{1-\lambda^2 s^2}\approx 1 - \tfrac12\lambda^2s^2$ and using
$s^2 = \tfrac12(1-\cos2\theta)$ gives the classic textbook forms

$$x \approx r\left[(1-\cos\theta) + \tfrac{\lambda}{4}(1-\cos2\theta)\right]$$
$$\dot x \approx r\omega\left[\sin\theta + \tfrac{\lambda}{2}\sin2\theta\right]
\qquad
\ddot x \approx r\omega^2\left[\cos\theta + \lambda\cos2\theta\right]$$

The first term is a pure harmonic motion (infinitely long rod); the $\lambda$ term
is the rod-obliquity correction at **twice** crank frequency.
""")

nb.code(r'''
rpm = 16000
w = ec.rpm_to_omega(rpm)
th = np.radians(np.linspace(0, 360, 1441))
x, v, a = ec.piston_kinematics(eng, th, w)
xs, vs, as_ = ec.piston_kinematics_series(eng, th, w)
deg = np.degrees(th)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, ex, ap, lab, k in zip(axs, (x, v, a), (xs, vs, as_),
                              ("x [mm]", "v [m/s]", "a [km/s²]"), (1e3, 1, 1e-3)):
    ax.plot(deg, ex * k, label="exact")
    ax.plot(deg, ap * k, ls="--", label="2nd-order series")
    tidy(ax, title=lab.split()[0] + f"  @ {rpm} rpm", xlabel="crank angle [°]", ylabel=lab)
    ax.set_xticks(range(0, 361, 90))
axs[0].legend()
plt.tight_layout(); plt.show()

print(f"max |error| series vs exact: x {np.max(abs(x-xs))*1e6:.1f} µm,  "
      f"a {np.max(abs(a-as_))/np.max(abs(a))*100:.2f} % of peak")
''')

nb.md(r"""
The series is excellent for *understanding* (and for balancing, where only
orders 1 and 2 matter), but it is not exact. For load calculations use the exact
form — it costs nothing in code.

## 1.4 · The numbers that matter

| quantity | formula | comment |
|---|---|---|
| mean piston speed | $c_m = 2 s n$ | the classic "how hard is this engine working" number |
| peak acceleration (TDC) | $r\omega^2(1+\lambda)$ | sets the inertia force — grows with $n^2$ |
| acceleration at BDC | $-r\omega^2(1-\lambda)$ | smaller: the rod helps at BDC |
| peak velocity | $\approx 1.6\,c_m$ | occurs *before* 90° ATDC, close to where the rod is perpendicular to the crank |
""")

nb.code(r'''
n = rpm / 60
c_m = 2 * eng.stroke * n
i_vmax = np.argmax(v)
print(f"Engine speed            {rpm} rpm")
print(f"mean piston speed c_m   {c_m:.1f} m/s")
print(f"peak piston speed       {v[i_vmax]:.1f} m/s  at {deg[i_vmax]:.1f}° ATDC  (= {v[i_vmax]/c_m:.2f} c_m)")
print(f"accel. at TDC           {a[0]/1e3:.1f} km/s²  = {a[0]/9.81:,.0f} g")
print(f"accel. at BDC           {a[720]/1e3:.1f} km/s²  = {a[720]/9.81:,.0f} g")
print(f"ratio TDC/BDC           {a[0]/-a[720]:.2f}  = (1+λ)/(1-λ) = {(1+eng.lam)/(1-eng.lam):.2f}")

# v max is close to rod ⟂ crank:  tan(theta) = l/r  ->  theta = atan(1/λ)
print(f"rod ⟂ crank at          {np.degrees(np.arctan(1/eng.lam)):.1f}°  (close, not exact)")
''')

nb.md(r"""
**Orientation values for mean piston speed** (rough, for feel):

| application | $c_m$ at rated speed |
|---|---|
| passenger-car diesel | 10–13 m/s |
| passenger-car gasoline | 14–18 m/s |
| superbike (production) | 22–25 m/s |
| MotoGP / F1 (historic V10) | ≈ 25–26 m/s |

$c_m$ is limited by flow (valve area vs. piston area) and by friction/wear;
peak acceleration — and thus the *inertia load* — is what limits the parts.
For a fixed $c_m$, shorter stroke means higher rpm and therefore higher $\omega^2 r$:

$$r\omega^2 = \frac{s}{2}\left(\frac{\pi c_m}{s}\right)^2 = \frac{\pi^2 c_m^2}{2s}$$

Halving the stroke at the same mean piston speed **doubles** the acceleration.
That is the price of the over-square, high-revving layout.
""")

nb.code(r'''
cm = 25.0
strokes = np.linspace(0.040, 0.080, 50)
acc = np.pi**2 * cm**2 / (2 * strokes) * (1 + eng.lam) / 9.81
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(strokes * 1e3, acc / 1e3)
ax.axvline(eng.stroke * 1e3, color=GRID, ls="--")
tidy(ax, title=f"TDC acceleration at constant c_m = {cm:.0f} m/s",
     xlabel="stroke [mm]", ylabel="acceleration [1000 g]")
plt.tight_layout(); plt.show()
''')

nb.md(r"""
## 1.5 · Orders contained in the piston motion

Because the kinematics repeat every revolution, the acceleration is a Fourier
series in the crank angle. It contains only **cosines** (the motion is symmetric
about TDC) and, apart from the first order, only **even** orders:

$$\frac{\ddot x}{r\omega^2} = \cos\theta + A_2\cos2\theta - A_4\cos4\theta + A_6\cos6\theta - \dots$$
$$A_2 \approx \lambda + \tfrac{\lambda^3}{4},\qquad A_4\approx\tfrac{\lambda^3}{4},\qquad A_6 \approx \tfrac{9\lambda^5}{128}$$

So the reciprocating mass produces free forces in orders 1, 2, 4, 6 … — order 2 is
about $\lambda$ times order 1, order 4 is already tiny. Odd orders above 1 do
**not** come from the piston acceleration; in the torque they appear through the
multiplication with the lever arm (notebook 3).
""")

nb.code(r'''
N = 4096
thf = np.linspace(0, 2*np.pi, N, endpoint=False)
_, _, af = ec.piston_kinematics(eng, thf, 1.0)
coef = (np.fft.rfft(af) / N * 2).real / eng.r
lam = eng.lam
approx = {1: 1.0, 2: lam + lam**3/4, 4: -lam**3/4, 6: 9*lam**5/128}
print(" order   FFT coeff     approx")
for k in range(1, 8):
    print(f"   {k}   {coef[k]:+.6f}   {approx.get(k, 0):+.6f}")

fig, ax = plt.subplots(figsize=(5.5, 3))
ks = np.arange(1, 9)
ax.bar(ks, np.abs(coef[1:9]), color=SERIES[0])
ax.set_yscale("log"); ax.set_ylim(1e-8, 2)
tidy(ax, title="|harmonic amplitude| of piston acceleration / rω²", xlabel="order", ylabel="[-]")
ax.set_xticks(ks)
plt.tight_layout(); plt.show()
''')

nb.md(r"""
## 1.6 · Effect of $\lambda$ (rod length)

A shorter rod (larger $\lambda$) means a lighter, stiffer rod and a lower engine —
but higher TDC acceleration, larger rod angles (→ more piston side force and
friction) and a larger second-order force. Typical values: $\lambda \approx 0.25$–$0.33$
for most engines; race engines often go longer (≈ 0.22–0.27).
""")

nb.code(r'''
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
for i, lam_i in enumerate((0.20, 0.25, 0.30, 0.35)):
    e_i = ec.Engine(rod_length=eng.r / lam_i)
    _, _, a_i = ec.piston_kinematics(e_i, th, w)
    axs[0].plot(deg, a_i / 9.81 / 1e3, label=f"λ = {lam_i:.2f}")
    axs[1].plot(deg, np.degrees(ec.rod_angle(e_i, th)), label=f"λ = {lam_i:.2f}")
tidy(axs[0], title=f"piston acceleration @ {rpm} rpm", xlabel="crank angle [°]", ylabel="[1000 g]")
tidy(axs[1], title="rod angle β", xlabel="crank angle [°]", ylabel="[°]")
for ax in axs: ax.set_xticks(range(0, 361, 90))
axs[0].legend()
plt.tight_layout(); plt.show()
''')

nb.md(r"""
## 1.7 · Rod angular motion

The rod swings as it translates. Its angular velocity and acceleration are needed
whenever the rod's own inertia matters (notebook 3, and the MBS model, where it is
exact by construction):

$$\dot\beta = \omega\,\frac{\lambda\cos\theta}{\cos\beta}\qquad
\ddot\beta = -\omega^2\,\frac{\lambda(1-\lambda^2)\sin\theta}{\cos^3\beta}$$

Peak $|\ddot\beta|$ is reached near 90°, where the rod's *rotational* inertia
creates a couple about the small end — a load the simple two-mass rod model misses.
""")

nb.code(r'''
bd, bdd = ec.rod_angular_kinematics(eng, th, w)
fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(deg, bdd / 1e3)
tidy(ax, title=f"rod angular acceleration @ {rpm} rpm", xlabel="crank angle [°]", ylabel="[10³ rad/s²]")
ax.set_xticks(range(0, 361, 90))
plt.tight_layout(); plt.show()
print(f"max |beta_ddot| = {np.max(abs(bdd)):.3g} rad/s²  -> couple J_cg·beta_ddot = {eng.J_rod_cg*np.max(abs(bdd)):.0f} N·m")
''')

nb.md(r"""
## 1.8 · Take-aways

- $\sin\beta = \lambda\sin\theta$ and $x = r(1-\cos\theta)+l(1-\cos\beta)$ — everything else is differentiation.
- Peak acceleration $r\omega^2(1+\lambda)$ at TDC; it scales with $n^2$ and, at fixed $c_m$, with $1/s$.
- Piston acceleration contains orders 1, 2, 4, 6 …; order 2 ≈ $\lambda\,\times$ order 1.
- The 2nd-order series is the language of balancing; use the exact form for loads.

**Exercises**

1. Find the rpm at which this engine reaches $c_m = 25$ m/s. What is the TDC acceleration in *g* there?
2. Keep displacement and bore/stroke ratio fixed but change the rod length from 105 to 95 mm. By how much does the TDC acceleration change? And the peak rod angle?
3. A piston-pin offset $e$ (pin moved towards the thrust side) changes $\sin\beta = (r\sin\theta - e)/l$. Modify `piston_kinematics` for the offset case and look at what happens around TDC.

**Next:** notebook 2 — where the *gas* force comes from.
""")
