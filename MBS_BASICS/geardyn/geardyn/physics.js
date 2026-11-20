// geardyn-lite: gear-pair dynamics for the browser animation.
// Same conventions as geardyn.py: theta1 CCW+, theta2 CW+, s = rb1*th1 - rb2*th2,
// drive flank force on gear 2 along n_d = (sin a, cos a), coast n_r = (sin a, -cos a).
// DOFs: q = [phi_in, th1, th2, phi_out, x1, y1, x2, y2]
"use strict";

function makeGearPair(o = {}) {
  const gp = Object.assign({
    z1: 24, z2: 48, m: 2.5e-3, alphaDeg: 20, b: 15e-3, ha: 1.0, hf: 1.25,
    cPrime: 14.0, zeta: 0.05, rho: 7850, tvRatio: 1.6, tvSmooth: 0.03,
    m1Extra: 0.3, m2Extra: 0.6,
  }, o);
  const al = gp.alphaDeg * Math.PI / 180;
  gp.alpha = al; gp.sa = Math.sin(al); gp.ca = Math.cos(al);
  gp.r1 = gp.m * gp.z1 / 2; gp.r2 = gp.m * gp.z2 / 2;
  gp.rb1 = gp.r1 * gp.ca; gp.rb2 = gp.r2 * gp.ca;
  gp.ra1 = gp.r1 + gp.ha * gp.m; gp.ra2 = gp.r2 + gp.ha * gp.m;
  gp.a = gp.r1 + gp.r2; gp.i = gp.z2 / gp.z1; gp.pb = Math.PI * gp.m * gp.ca;
  gp.eps = (Math.sqrt(gp.ra1 ** 2 - gp.rb1 ** 2) + Math.sqrt(gp.ra2 ** 2 - gp.rb2 ** 2) - gp.a * gp.sa) / gp.pb;
  const md1 = gp.rho * Math.PI * gp.r1 ** 2 * gp.b, md2 = gp.rho * Math.PI * gp.r2 ** 2 * gp.b;
  gp.J1 = 0.5 * md1 * gp.r1 ** 2; gp.J2 = 0.5 * md2 * gp.r2 ** 2;
  gp.m1 = md1 + gp.m1Extra; gp.m2 = md2 + gp.m2Extra;
  gp.meq = 1 / (gp.rb1 ** 2 / gp.J1 + gp.rb2 ** 2 / gp.J2);
  gp.cGamma = gp.cPrime * (0.75 * gp.eps + 0.25);
  gp.k = gp.cGamma * gp.b * 1e3 * 1e6;
  gp.c = 2 * gp.zeta * Math.sqrt(gp.k * gp.meq);
  // time-varying stiffness: smoothed box, mean = k
  const e = Math.min(gp.eps, 2), h = 0.5 * (e - 1), sw = Math.max(gp.tvSmooth, 1e-4);
  const box = (s) => {
    const d = ((s - h + 0.5) % 1 + 1) % 1 - 0.5, A = (d + h) / sw, B = (d - h) / sw;
    const ch2a = 1 / Math.cosh(A) ** 2, ch2b = 1 / Math.cosh(B) ** 2;
    return [0.5 * (Math.tanh(A) - Math.tanh(B)), 0.5 / sw * (ch2a - ch2b)];
  };
  let mean = 0; const N = 4000;
  for (let j = 0; j < N; j++) mean += 1 + (gp.tvRatio - 1) * box(j / N)[0];
  gp.kSingle = gp.k / (mean / N);
  gp.kOfS = (s) => { const [w, dw] = box(s); return [gp.kSingle * (1 + (gp.tvRatio - 1) * w), gp.kSingle * (gp.tvRatio - 1) * dw]; };
  return gp;
}

// options: backlash [m], tv (bool), bearings (bool), bending (bool), inertias (bool),
// shafts (bool, torsional flexibility), kb, kBend, ks1, ks2, Jin, Jout, zetaS, zetaB
function makeSim(gp, opt) {
  const o = Object.assign({
    backlash: 60e-6, tv: false, bearings: false, bending: false, inertias: false, shafts: false,
    kb: 3e8, kBend: 3e8, ks1: 1.62e4, ks2: 4.25e4, Jin: 0.02, Jout: 0.05, zetaS: 0.02, zetaB: 0.03,
  }, opt);
  const sim = { gp, o, t: 0, Ediss: 0, Wext: 0 };
  const g = o.backlash / 2;
  const shaftFlex = o.inertias && o.shafts;
  const lateral = o.bearings || o.bending;
  let kl = 0;
  if (lateral) {
    const parts = [];
    parts.push(o.bearings ? o.kb : Infinity);
    parts.push(o.bending ? o.kBend : Infinity);
    kl = 1 / parts.reduce((s, k) => s + 1 / k, 0);
  }
  sim.kl = kl; sim.shaftFlex = shaftFlex; sim.lateral = lateral; sim.gap = g;
  const I = [o.Jin, gp.J1, gp.J2, o.Jout, gp.m1, gp.m1, gp.m2, gp.m2];
  if (o.inertias && !shaftFlex) { I[1] += o.Jin; I[2] += o.Jout; }
  sim.I = I;
  sim.inDof = shaftFlex ? 0 : 1; sim.outDof = shaftFlex ? 3 : 2;
  // active dofs
  const act = [shaftFlex, true, true, shaftFlex, lateral, lateral, lateral, lateral];
  sim.act = act;
  const cs1 = 2 * o.zetaS * Math.sqrt(o.ks1 * (o.Jin * gp.J1 / (o.Jin + gp.J1)));
  const cs2 = 2 * o.zetaS * Math.sqrt(o.ks2 * (o.Jout * gp.J2 / (o.Jout + gp.J2)));
  const cl1 = lateral ? 2 * o.zetaB * Math.sqrt(kl * gp.m1) : 0, cl2 = lateral ? 2 * o.zetaB * Math.sqrt(kl * gp.m2) : 0;
  sim.q = new Float64Array(8); sim.v = new Float64Array(8);
  sim.clampOut = false;        // output-end dof fixed
  sim.Tin = () => 0;           // torque on input-end dof (t)
  sim.Tout = () => 0;          // load torque on output-end dof (opposing CW rotation)
  sim.presc = null;            // {f, fd, fdd} prescribed input-end motion

  // mesh evaluation
  function mesh(q, v) {
    const th1 = q[1], th2 = q[2];
    const s = gp.rb1 * th1 - gp.rb2 * th2, sd = gp.rb1 * v[1] - gp.rb2 * v[2];
    const dx = q[4] - q[6], dy = q[5] - q[7], dxd = v[4] - v[6], dyd = v[5] - v[7];
    const pd = s + dx * gp.sa + dy * gp.ca - g, pr = -s + dx * gp.sa - dy * gp.ca - g;
    const pdd = sd + dxd * gp.sa + dyd * gp.ca, prd = -sd + dxd * gp.sa - dyd * gp.ca;
    let k = gp.k, dk = 0;
    if (o.tv) { const r = gp.kOfS(th1 * gp.z1 / (2 * Math.PI)); k = r[0]; dk = r[1] * gp.z1 / (2 * Math.PI); }
    const Fd = pd > 0 ? Math.max(k * pd + gp.c * pdd, 0) : 0;
    const Fr = pr > 0 ? Math.max(k * pr + gp.c * prd, 0) : 0;
    const pdp = Math.max(pd, 0), prp = Math.max(pr, 0);
    const U = 0.5 * k * (pdp * pdp + prp * prp);
    const P = (Fd - k * pdp) * pdd + (Fr - k * prp) * prd;
    return { Fd, Fr, F: Fd - Fr, pd, pr, pdd, prd, s, k, U, P, dUdth: 0.5 * dk * (pdp * pdp + prp * prp) };
  }
  sim.mesh = mesh;

  function forces(t, q, v) {
    const Q = new Float64Array(8);
    const m = mesh(q, v);
    Q[1] += -gp.rb1 * m.F - m.dUdth; Q[2] += gp.rb2 * m.F;
    if (lateral) {
      const fx2 = (m.Fd + m.Fr) * gp.sa, fy2 = (m.Fd - m.Fr) * gp.ca;
      Q[4] -= fx2; Q[5] -= fy2; Q[6] += fx2; Q[7] += fy2;
      Q[4] -= kl * q[4] + cl1 * v[4]; Q[5] -= kl * q[5] + cl1 * v[5];
      Q[6] -= kl * q[6] + cl2 * v[6]; Q[7] -= kl * q[7] + cl2 * v[7];
    }
    let Pd = m.P + (lateral ? cl1 * (v[4] ** 2 + v[5] ** 2) + cl2 * (v[6] ** 2 + v[7] ** 2) : 0);
    if (shaftFlex) {
      const t1 = o.ks1 * (q[0] - q[1]) + cs1 * (v[0] - v[1]);
      const t2 = o.ks2 * (q[2] - q[3]) + cs2 * (v[2] - v[3]);
      Q[0] -= t1; Q[1] += t1; Q[2] -= t2; Q[3] += t2;
      Pd += cs1 * (v[0] - v[1]) ** 2 + cs2 * (v[2] - v[3]) ** 2;
    }
    const Ti = sim.Tin(t), To = sim.Tout(t);
    Q[sim.inDof] += Ti; Q[sim.outDof] -= To;
    const Pw = Ti * v[sim.inDof] - To * v[sim.outDof];
    return { Q, m, Pd, Pw };
  }
  sim.forces = forces;

  function applyConstraints(t, q, v) {
    if (sim.presc) { const p = sim.presc; q[sim.inDof] = p.f(t); v[sim.inDof] = p.fd(t); }
    if (sim.clampOut) { v[sim.outDof] = 0; }
    if (!shaftFlex) { q[0] = q[1]; v[0] = v[1]; q[3] = q[2]; v[3] = v[2]; }
  }
  sim.applyConstraints = applyConstraints;

  function deriv(t, q, v) {
    applyConstraints(t, q, v);
    const { Q, Pd, Pw } = forces(t, q, v);
    const a = new Float64Array(8);
    for (let j = 0; j < 8; j++) a[j] = act[j] ? Q[j] / I[j] : 0;
    if (sim.presc) a[sim.inDof] = sim.presc.fdd(t);
    if (sim.clampOut) a[sim.outDof] = 0;
    if (!shaftFlex) { a[0] = a[1]; a[3] = a[2]; }
    return { a, Pd, Pw };
  }

  sim.step = function (dt) {
    const t = sim.t, q = sim.q, v = sim.v;
    applyConstraints(t, q, v);
    const k1 = deriv(t, q.slice(), v.slice());
    const q2 = new Float64Array(8), v2 = new Float64Array(8);
    for (let j = 0; j < 8; j++) { q2[j] = q[j] + 0.5 * dt * v[j]; v2[j] = v[j] + 0.5 * dt * k1.a[j]; }
    const k2 = deriv(t + dt / 2, q2, v2);
    const q3 = new Float64Array(8), v3 = new Float64Array(8);
    for (let j = 0; j < 8; j++) { q3[j] = q[j] + 0.5 * dt * v2[j]; v3[j] = v[j] + 0.5 * dt * k2.a[j]; }
    const k3 = deriv(t + dt / 2, q3, v3);
    const q4 = new Float64Array(8), v4 = new Float64Array(8);
    for (let j = 0; j < 8; j++) { q4[j] = q[j] + dt * v3[j]; v4[j] = v[j] + dt * k3.a[j]; }
    const k4 = deriv(t + dt, q4, v4);
    for (let j = 0; j < 8; j++) {
      q[j] += dt / 6 * (v[j] + 2 * v2[j] + 2 * v3[j] + v4[j]);
      v[j] += dt / 6 * (k1.a[j] + 2 * k2.a[j] + 2 * k3.a[j] + k4.a[j]);
    }
    sim.Ediss += dt / 6 * (k1.Pd + 2 * k2.Pd + 2 * k3.Pd + k4.Pd);
    sim.Wext += dt / 6 * (k1.Pw + 2 * k2.Pw + 2 * k3.Pw + k4.Pw);
    sim.t = t + dt;
    applyConstraints(sim.t, q, v);
  };

  // energies
  sim.energies = function () {
    const q = sim.q, v = sim.v, m = mesh(q, v);
    let T = 0; for (let j = 0; j < 8; j++) if (act[j]) T += 0.5 * I[j] * v[j] ** 2;
    const Ub = lateral ? 0.5 * kl * (q[4] ** 2 + q[5] ** 2 + q[6] ** 2 + q[7] ** 2) : 0;
    const Us = shaftFlex ? 0.5 * o.ks1 * (q[0] - q[1]) ** 2 + 0.5 * o.ks2 * (q[2] - q[3]) ** 2 : 0;
    return { T, Um: m.U, Ub, Us, Ed: sim.Ediss, W: sim.Wext };
  };

  // static state transmitting torque T (>0 drive flank, <0 coast flank) with the
  // output end held at angle phiOut (rotating frame for steady running)
  sim.staticState = function (T, phiOut = 0) {
    const q = sim.q; q.fill(0);
    const F = T / gp.rb1;                  // signed mesh force
    const Fd = Math.max(F, 0), Fr = Math.max(-F, 0);
    if (lateral) {
      const fx2 = (Fd + Fr) * gp.sa, fy2 = (Fd - Fr) * gp.ca;
      q[4] = -fx2 / kl; q[5] = -fy2 / kl; q[6] = fx2 / kl; q[7] = fy2 / kl;
    }
    let k = gp.k;
    const T2 = F * gp.rb2;                 // torque on gear 2 (CW) from the mesh
    q[3] = phiOut;
    q[2] = phiOut + (shaftFlex ? T2 / o.ks2 : 0);
    // penetration on the loaded flank (iterate once for tv stiffness)
    for (let it = 0; it < 3; it++) {
      if (o.tv) k = gp.kOfS(q[1] * gp.z1 / (2 * Math.PI))[0];
      const dx = q[4] - q[6], dy = q[5] - q[7];
      if (F >= 0) q[1] = (Math.abs(F) / k + g - dx * gp.sa - dy * gp.ca + gp.rb2 * q[2]) / gp.rb1;
      else q[1] = (-(Math.abs(F) / k + g) + dx * gp.sa - dy * gp.ca + gp.rb2 * q[2]) / gp.rb1;
    }
    q[0] = q[1] + (shaftFlex ? T / o.ks1 : 0);
    if (!shaftFlex) { q[0] = q[1]; q[3] = q[2]; }
    sim.v.fill(0); sim.Ediss = 0; sim.Wext = 0;
  };
  return sim;
}

if (typeof module !== "undefined") module.exports = { makeGearPair, makeSim };
