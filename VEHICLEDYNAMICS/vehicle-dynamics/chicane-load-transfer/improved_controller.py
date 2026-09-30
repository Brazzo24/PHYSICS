"""
Improved Corner-Exit Controller — Complete Fix
==============================================
Three compounding problems identified and resolved:

  1. UNSTABLE GAINS (Kp=120 << mgh=1252 Nm)
     → Tuned to Kp=1800, Kd=200, Ki=60 for ζ≈0.78

  2. MISSING GRAVITY FEEDFORWARD
     At any lean angle, gravity creates a toppling moment mgh·sin(φ).
     A pure PID cannot reject this without steady-state error unless
     the integral winds up (slow) or a feedforward term cancels it.
     → Added: tau_ff = -mgh·sin(phi_ref)  (gravity cancellation)
     → Also added inertia FF: tau_inertia = I·phi_ddot_ref

  3. DISCONTINUOUS REFERENCE (hard ramp → step in phi_dot)
     → Replaced with S-curve sigmoid that is smooth at t=0 and t=T.

  BONUS: Improved Fz model:
     - CoM longitudinal offset (l_f ≠ L/2)
     - Centripetal front/rear load transfer
     - Mozzi-pitch inertial coupling
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ══════════════════════════════════════════════════════════════════
@dataclass
class BikeParams:
    m         : float = 220.0
    h_com     : float = 0.58
    wheelbase : float = 1.40
    l_f       : float = 0.65    # CoM-to-front distance (slightly fwd of mid)
    I_roll    : float = 30.0
    g         : float = 9.81
    v_fwd     : float = 14.0
    B:float=10.0; C:float=1.9; D:float=1.0; E:float=-1.0
    Kp:float=1800.0; Kd:float=200.0; Ki:float=60.0
    use_ff:bool=True
    label:str="Improved"; color:str="#f0a500"

    @property
    def l_r(self): return self.wheelbase - self.l_f
    @property
    def mgh(self): return self.m * self.g * self.h_com
    @property
    def omega_n(self):
        ex=self.Kp-self.mgh; return np.sqrt(max(ex,0)/self.I_roll)
    @property
    def zeta(self):
        ex=self.Kp-self.mgh
        if ex<=0: return float('nan')
        return self.Kd/(2*np.sqrt(self.I_roll*ex))

# ══════════════════════════════════════════════════════════════════
# 2.  S-CURVE REFERENCE
# ══════════════════════════════════════════════════════════════════
def make_scurve_ref(phi_apex_deg, t_total, steepness=5.0):
    phi_apex = np.radians(phi_apex_deg)
    t_mid    = t_total / 2.0
    k        = steepness

    def _s(t):   return 1.0/(1.0+np.exp(k*(t-t_mid)))
    def phi_ref(t):
        return phi_apex * _s(t)
    def phi_dot_ref(t):
        s=_s(t); return phi_apex*(-k*s*(1.0-s))
    def phi_ddot_ref(t):
        s=_s(t); ds=-k*s*(1.0-s)
        return phi_apex*(-k*ds*(1.0-2.0*s))
    return phi_ref, phi_dot_ref, phi_ddot_ref

# ══════════════════════════════════════════════════════════════════
# 3.  IMPROVED NORMAL FORCE MODEL
# ══════════════════════════════════════════════════════════════════
def mozzi_pitch_h(phi, phi_dot, psi_dot, p):
    omega = np.array([phi_dot*np.cos(phi), phi_dot*np.sin(phi), psi_dot])
    r_com = np.array([0.0, -p.h_com*np.sin(phi), p.h_com*np.cos(phi)])
    v_com = np.array([p.v_fwd, 0.0, phi_dot*p.h_com*np.cos(phi)])
    w2    = float(omega@omega)
    if w2 < 1e-10: return 0.0, None
    h  = float(omega@v_com)/w2
    ra = r_com + np.cross(omega, v_com)/w2
    return h, ra

def vertical_loads(phi, phi_dot, psi_dot, p):
    L=p.wheelbase; Fz0=p.m*p.g
    Fz_f = Fz0*p.l_r/L
    Fz_r = Fz0*p.l_f/L
    if abs(psi_dot)>1e-4:
        a_lat = p.v_fwd*abs(psi_dot)
        dFc   = p.m*a_lat*p.h_com/L*np.cos(phi)
        Fz_f += dFc; Fz_r -= dFc
    h_sc, ra = mozzi_pitch_h(phi, phi_dot, psi_dot, p)
    if ra is not None:
        ap  = np.arctan2(ra[2], ra[0]+1e-9)
        ar  = phi_dot**2*p.h_com*np.cos(phi)
        dFm = np.clip(0.25*p.m*ar*np.tan(ap), -Fz0*0.35, Fz0*0.35)
        Fz_f += dFm; Fz_r -= dFm
    return max(Fz_f,10.0), max(Fz_r,10.0)

def pacejka_Fy(alpha, Fz, p):
    Fz0=p.m*p.g/2.0; mu=p.D*(Fz/Fz0)
    phi_=(1-p.E)*alpha+(p.E/p.B)*np.arctan(p.B*alpha)
    return mu*Fz*np.sin(p.C*np.arctan(p.B*phi_))

# ══════════════════════════════════════════════════════════════════
# 4.  ODE
# ══════════════════════════════════════════════════════════════════
def make_ode(p, ref_fn, dref_fn, ddref_fn, throttle_Fx=600.0):
    def rhs(t, y):
        phi, phi_dot, e_int = y
        ref=ref_fn(t); refd=dref_fn(t); refdd=ddref_fn(t)
        err=ref-phi; errd=refd-phi_dot
        ei_c = np.clip(e_int, -np.radians(15), np.radians(15))
        tau_pid = p.Kp*err + p.Kd*errd + p.Ki*ei_c
        tau_ff  = (-p.mgh*np.sin(ref) + p.I_roll*refdd) if p.use_ff else 0.0
        tau     = tau_pid + tau_ff
        psi_dot = p.v_fwd*np.tan(phi)/p.wheelbase
        Fz_f, Fz_r = vertical_loads(phi, phi_dot, psi_dot, p)
        alpha   = np.arctan2(psi_dot*p.wheelbase, p.v_fwd)
        Fy_f    = pacejka_Fy( alpha, Fz_f, p)
        Fy_r    = pacejka_Fy(-alpha, Fz_r, p)
        lat_mom = (Fy_f+Fy_r)*p.h_com*np.cos(phi)
        grav_m  = p.mgh*np.sin(phi)
        phi_ddot= (tau+grav_m-lat_mom)/p.I_roll
        return [phi_dot, phi_ddot, err]
    return rhs

def run(p, phi_apex_deg=32.0, t_total=2.5, steepness=5.0,
        throttle_Fx=600.0, n_pts=500):
    ref_fn, dref_fn, ddref_fn = make_scurve_ref(phi_apex_deg, t_total, steepness)
    y0    = [ref_fn(0.0), 0.0, 0.0]
    t_ev  = np.linspace(0.0, t_total, n_pts)
    ode   = make_ode(p, ref_fn, dref_fn, ddref_fn, throttle_Fx)
    sol   = solve_ivp(ode,(0.0,t_total),y0,t_eval=t_ev,
                      method='RK45',rtol=1e-8,atol=1e-10,max_step=0.005)
    t=sol.t; phi=sol.y[0]; phid=sol.y[1]; e_int=sol.y[2]
    pr = np.array([ref_fn(ti)  for ti in t])
    dr = np.array([dref_fn(ti) for ti in t])
    Ff_a=np.zeros(len(t)); Fr_a=np.zeros(len(t))
    Fxa=np.zeros(len(t));  tma=np.zeros(len(t))
    pha=np.zeros(len(t));  taua=np.zeros(len(t))
    tff=np.zeros(len(t));  tpid=np.zeros(len(t))
    for i in range(len(t)):
        psid=p.v_fwd*np.tan(phi[i])/p.wheelbase
        Ff,Fr=vertical_loads(phi[i],phid[i],psid,p)
        Ff_a[i]=Ff; Fr_a[i]=Fr; Fxa[i]=p.D*Fr; tma[i]=p.D*Fr-throttle_Fx
        h,_=mozzi_pitch_h(phi[i],phid[i],psid,p)
        pha[i]=h if h else 0.0
        ref=pr[i]; err=ref-phi[i]; errd=dr[i]-phid[i]
        refdd=ddref_fn(t[i])
        ei_c=np.clip(e_int[i],-np.radians(15),np.radians(15))
        tp=p.Kp*err+p.Kd*errd+p.Ki*ei_c
        tf=(-p.mgh*np.sin(ref)+p.I_roll*refdd) if p.use_ff else 0.0
        tpid[i]=tp; tff[i]=tf; taua[i]=tp+tf
    return dict(t=t,phi=phi,phid=phid,phi_ref=pr,phid_ref=dr,
                Fz_f=Ff_a,Fz_r=Fr_a,Fx_avail=Fxa,trac_margin=tma,
                pitch=pha,tau=taua,tau_ff=tff,tau_pid=tpid,
                label=p.label,color=p.color,status=sol.status)

# ══════════════════════════════════════════════════════════════════
# 5.  SCENARIOS
# ══════════════════════════════════════════════════════════════════
APEX=32.0; T=2.5; THR=600.0
P_ref = BikeParams()

print(f"mgh = {P_ref.mgh:.0f} Nm  (stability threshold for Kp)")
print(f"Tuned: ωn={P_ref.omega_n:.2f} rad/s, ζ={P_ref.zeta:.3f}\n")

scenarios = [
    (BikeParams(Kp=120,Kd=25,Ki=0,use_ff=False,
                label="① Old (Kp=120, no FF)",color="#6b7385"),  5.0),
    (BikeParams(Kp=1800,Kd=200,Ki=60,use_ff=False,
                label="② Tuned gains, no FF", color="#ea580c"),  5.0),
    (BikeParams(Kp=1800,Kd=200,Ki=60,use_ff=True,
                label="③ Tuned + FF",          color="#2563eb"),  5.0),
    (BikeParams(Kp=1800,Kd=200,Ki=60,use_ff=True,
                label="④ Tuned+FF+slow S (k=3)",color="#16a34a"), 3.0),
    (BikeParams(Kp=1800,Kd=200,Ki=60,use_ff=True,
                label="⑤ Tuned+FF+fast S (k=7)",color="#dc2626"), 7.0),
]

results = []
for p, steep in scenarios:
    r = run(p, APEX, T, steep, THR)
    err=np.degrees(r['phi']-r['phi_ref'])
    spin=np.sum(r['trac_margin']<0)*(T/len(r['t']))
    print(f"{r['label']:38s}  RMS={np.sqrt(np.mean(err**2)):.1f}°  "
          f"max={np.max(np.abs(err)):.1f}°  spin={spin:.2f}s  "
          f"minFzr={np.min(r['Fz_r']):.0f}N  [{r['status']}]")
    results.append(r)

# ══════════════════════════════════════════════════════════════════
# 6.  FIGURE 1 — Stepwise fix (3 rows × 5 columns)
# ══════════════════════════════════════════════════════════════════
F_st = P_ref.m*P_ref.g/2.0
fig1 = plt.figure(figsize=(18,10))
fig1.suptitle("Corner-Exit Controller: Stepwise Improvement\n"
              "Each column adds one fix — read left to right",
              fontsize=13,fontweight='bold')
gs = gridspec.GridSpec(3,5,figure=fig1,hspace=0.52,wspace=0.32)

for col, r in enumerate(results):
    # Row 0: lean tracking
    ax=fig1.add_subplot(gs[0,col])
    ax.plot(r['t'],np.degrees(r['phi']),    color=r['color'],lw=2,  label='actual')
    ax.plot(r['t'],np.degrees(r['phi_ref']),color='white',   lw=1.2,ls='--',alpha=0.55,label='ref')
    ax.set_title(r['label'],fontsize=7.5,pad=3)
    if col==0: ax.set_ylabel("φ [°]")
    ax.set_ylim(-5,45); ax.grid(True,alpha=0.25); ax.axhline(0,color='grey',lw=0.5)

    # Row 1: tracking error
    ax=fig1.add_subplot(gs[1,col])
    err=np.degrees(r['phi']-r['phi_ref'])
    ax.plot(r['t'],err,color=r['color'],lw=2)
    ax.axhline(0,color='k',lw=0.7,ls='--')
    ax.fill_between(r['t'],-3,3,alpha=0.09,color='green')
    ax.set_ylim(-35,20); ax.grid(True,alpha=0.25)
    if col==0: ax.set_ylabel("Δφ [°]")
    rms=np.sqrt(np.mean(err**2))
    ax.text(0.97,0.97,f"RMS={rms:.1f}°",transform=ax.transAxes,
            ha='right',va='top',fontsize=7.5,color=r['color'])

    # Row 2: rear tyre load
    ax=fig1.add_subplot(gs[2,col])
    ax.plot(r['t'],r['Fz_r'],color=r['color'],lw=2)
    ax.axhline(F_st,color='white',lw=0.8,ls='--',alpha=0.45)
    ax.fill_between(r['t'],r['Fz_r'],F_st,
                    where=np.array(r['Fz_r'])<F_st,
                    alpha=0.20,color='red')
    ax.set_ylim(0,2200); ax.grid(True,alpha=0.25)
    ax.set_xlabel("t [s]")
    if col==0: ax.set_ylabel("F_z rear [N]")
    mfr=np.min(r['Fz_r'])
    ax.text(0.97,0.04,f"min={mfr:.0f}N",transform=ax.transAxes,
            ha='right',va='bottom',fontsize=7.5,color=r['color'])

# Row labels
for row,txt in enumerate(["Lean tracking","Tracking error","Rear tyre load"]):
    ax=fig1.add_subplot(gs[row,0])
    ax.text(-0.35,0.5,txt,transform=ax.transAxes,
            va='center',ha='right',fontsize=8.5,color='grey',
            rotation=90,fontweight='bold')

plt.savefig("/mnt/user-data/outputs/controller_stepwise_fix.png",
            dpi=150,bbox_inches='tight')
plt.close(); print("\nFig 1 saved.")

# ══════════════════════════════════════════════════════════════════
# 7.  FIGURE 2 — Old vs Best: deep dive
# ══════════════════════════════════════════════════════════════════
old=results[0]; best=results[3]   # slow S-curve best balance

fig2,axes2=plt.subplots(2,3,figsize=(15,9))
fig2.suptitle(f"Deep Dive: Old Baseline  vs.  {best['label']}\n"
              "Torque decomposition · Fz model · Traction margin",
              fontsize=12,fontweight='bold')
c0=old['color']; cb=best['color']

# (0,0) Lean tracking
ax=axes2[0,0]
for r,c in[(old,c0),(best,cb)]:
    ax.plot(r['t'],np.degrees(r['phi']),    color=c,lw=2.5,label=r['label'][:25])
    ax.plot(r['t'],np.degrees(r['phi_ref']),color=c,lw=1.5,ls='--',alpha=0.5)
ax.set_title("Lean Tracking  (dashed=ref)"); ax.set_ylabel("φ [°]")
ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# (0,1) Tracking error
ax=axes2[0,1]
for r,c in[(old,c0),(best,cb)]:
    err=np.degrees(r['phi']-r['phi_ref'])
    ax.plot(r['t'],err,color=c,lw=2.5,label=r['label'][:25])
ax.axhline(0,color='k',lw=0.8,ls='--')
ax.fill_between(old['t'],-3,3,alpha=0.07,color='green',label='±3° band')
ax.set_title("Tracking Error Δφ"); ax.set_ylabel("Δφ [°]")
ax.set_ylim(-30,15); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# (0,2) Torque decomposition — best
ax=axes2[0,2]
ax.plot(best['t'],best['tau'],    color=cb,        lw=2.5,label='Total τ')
ax.plot(best['t'],best['tau_pid'],color='steelblue',lw=1.8,ls='--',label='τ_PID')
ax.plot(best['t'],best['tau_ff'], color='coral',    lw=1.8,ls=':',label='τ_FF (grav+inertia)')
ax.axhline(0,color='k',lw=0.5)
ax.set_title("Torque Decomposition (best)"); ax.set_ylabel("τ [Nm]")
ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

# (1,0) Fz
ax=axes2[1,0]
ax.plot(old['t'],  old['Fz_r'],  color=c0, lw=2,  ls='--',label='Old Fz_rear')
ax.plot(best['t'], best['Fz_r'], color=cb, lw=2.5,        label='Best Fz_rear')
ax.plot(best['t'], best['Fz_f'], color='tomato',lw=1.8,ls=':',label='Best Fz_front')
ax.axhline(F_st,color='grey',lw=0.8,ls='--',label=f'Static ({F_st:.0f}N)')
ax.set_title("Vertical Tyre Loads (improved Fz model)"); ax.set_ylabel("F_z [N]")
ax.set_xlabel("t [s]"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# (1,1) Mozzi pitch
ax=axes2[1,1]
for r,c in[(old,c0),(best,cb)]:
    ax.plot(r['t'],r['pitch'],color=c,lw=2.5,label=r['label'][:25])
ax.axhline(0,color='k',lw=0.8,ls='--')
ax.fill_between(best['t'],best['pitch'],0,
                where=np.array(best['pitch'])>0,alpha=0.15,color='gold')
ax.set_title("Mozzi Screw Pitch"); ax.set_ylabel("h [m/rad]")
ax.set_xlabel("t [s]"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

# (1,2) Traction margin
ax=axes2[1,2]
for r,c in[(old,c0),(best,cb)]:
    ax.plot(r['t'],r['trac_margin'],color=c,lw=2.5,label=r['label'][:25])
    spin=np.sum(r['trac_margin']<0)*(T/len(r['t']))
    ax.text(0.02,0.97-0.13*(r is best),
            f"spin={spin:.2f}s",transform=ax.transAxes,
            fontsize=8,color=c,va='top')
ax.axhline(0,color='k',lw=1.0)
ax.fill_between(old['t'], np.minimum(np.array(old['trac_margin']),0),0,
                alpha=0.18,color='red')
ax.fill_between(best['t'],np.minimum(np.array(best['trac_margin']),0),0,
                alpha=0.18,color='gold')
ax.set_title("Traction Margin"); ax.set_ylabel("[N]")
ax.set_xlabel("t [s]"); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/controller_old_vs_best.png",
            dpi=150,bbox_inches='tight')
plt.close(); print("Fig 2 saved.")

# ══════════════════════════════════════════════════════════════════
# 8.  FIGURE 3 — S-curve tuning guide
# ══════════════════════════════════════════════════════════════════
fig3,axes3=plt.subplots(1,2,figsize=(12,5))
fig3.suptitle("S-Curve Reference — Steepness Tuning Guide",fontsize=12,fontweight='bold')
t_g=np.linspace(0,T,300)
steeps=[2.5,3.5,5.0,6.5,8.0]
cmap=plt.cm.plasma(np.linspace(0.15,0.9,len(steeps)))
for k,s in zip(cmap,steeps):
    fn,dfn,_=make_scurve_ref(APEX,T,s)
    axes3[0].plot(t_g,[np.degrees(fn(ti))  for ti in t_g],color=k,lw=2,label=f'k={s}')
    axes3[1].plot(t_g,[np.degrees(dfn(ti)) for ti in t_g],color=k,lw=2,label=f'k={s}')
# Old ramp
rp=[np.degrees(max(np.radians(APEX)-np.radians(40)*ti,0)) for ti in t_g]
rd=[-40.0 if np.radians(APEX)-np.radians(40)*ti>0 else 0.0 for ti in t_g]
axes3[0].plot(t_g,rp,'k--',lw=1.5,label='Old ramp')
axes3[1].plot(t_g,rd,'k--',lw=1.5,label='Old ramp')
for ax,yl,ttl in zip(axes3,["φ_ref [°]","φ̇_ref [°/s]"],
                              ["Reference Lean","Reference Roll Rate"]):
    ax.axhline(0,color='grey',lw=0.5); ax.set_xlabel("t [s]")
    ax.set_ylabel(yl); ax.set_title(ttl); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/scurve_tuning_guide.png",
            dpi=150,bbox_inches='tight')
plt.close(); print("Fig 3 saved.")

# Final summary
print("\n"+"═"*78)
print(f"{'Scenario':<38} {'RMS [°]':>8} {'Max [°]':>8} {'Spin [s]':>9} {'Min Fzr':>9}")
print("═"*78)
for r in results:
    e=np.degrees(r['phi']-r['phi_ref'])
    sp=np.sum(r['trac_margin']<0)*(T/len(r['t']))
    print(f"{r['label']:<38} {np.sqrt(np.mean(e**2)):>8.1f} {np.max(np.abs(e)):>8.1f} "
          f"{sp:>9.2f} {np.min(r['Fz_r']):>9.0f}N")
print("═"*78)
print("\n✓ Complete.")
