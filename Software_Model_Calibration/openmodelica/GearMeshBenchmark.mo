within ;
package GearMeshBenchmark
  "ValDyn gear-mesh benchmark (Stage 1: gear pair, Stage 2: gear on shaft with bearing clearance) - OpenModelica twin of gearmesh_model.py"
  extends GearMeshBenchmark.Icons.Package;
  constant Real pi = 3.141592653589793;

  annotation (
    preferredView="info",
    Documentation(info="<html>
<p>Planar (2D) re-implementation of the ValDyn benchmark models from
<i>Benchmark-Daten-Kalibrierung.key</i>, equivalent to the Python file
<code>gearmesh_model.py</code>. The package is self-contained: the Modelica
Standard Library is <b>not</b> required.</p>
<h4>Conventions</h4>
<ul>
<li>All quantities in SI units (m, kg, s, N, rad). The ValDyn mask values
(N/mm, mm, um) are converted in the parameter defaults.</li>
<li>Connector <code>Frame</code>: potentials are the <b>displacements</b>
u, v and rotation phi of a node relative to its nominal position (keeps
micrometre motions well-conditioned for the solver). Flows fx, fy, t are the
cut forces/torque <b>acting on the component</b> (same convention as
Modelica.Mechanics).</li>
<li>Rotation angle in the ValDyn plots = driver angle = <code>angle_deg</code>.</li>
</ul>
<h4>Models</h4>
<ul>
<li><code>Examples.Stage1</code>: 2 gears (x,y,phi), elastic supports, drive spring, drag torque, GEARMESH2D.</li>
<li><code>Examples.Stage2</code>: gear 2 rides on a non-rotating shaft through a radial bearing with clearance;
shaft ends on non-linear Block/Case supports; load 10*sin(driver angle). Mask values (does not match ValDyn).</li>
<li><code>Examples.Stage2Calibrated</code>: Stage 2 with backlash 0.012 mm and mesh stiffness 25000 N/mm.</li>
</ul>
<p>Recommended simulation settings: DASSL, tolerance 1e-8, output interval 1 us (Stage 1) / 2 us (Stage 2),
see the <code>experiment</code> annotations.</p>
</html>"));

  package Icons
    partial package Package
      annotation (Icon(graphics={Rectangle(extent={{-100,100},{100,-100}}, lineColor={0,0,0},
            fillColor={240,240,240}, fillPattern=FillPattern.Solid, radius=20),
          Ellipse(extent={{-70,40},{10,-40}}, lineColor={0,0,0}), Ellipse(extent={{10,30},{70,-30}}, lineColor={0,0,0})}));
    end Package;
  end Icons;

  // ---------------------------------------------------------------------------
  connector Frame "Planar node: displacements from nominal position (potential), cut forces (flow)"
    Real u(unit="m") "x displacement";
    Real v(unit="m") "y displacement";
    Real phi(unit="rad") "rotation";
    flow Real fx(unit="N") "cut force x acting on the component";
    flow Real fy(unit="N") "cut force y acting on the component";
    flow Real t(unit="N.m") "cut torque acting on the component";
    annotation (Icon(graphics={Ellipse(extent={{-60,60},{60,-60}}, lineColor={0,0,0},
          fillColor={95,95,95}, fillPattern=FillPattern.Solid)}));
  end Frame;

  // ---------------------------------------------------------------------------
  package Functions
    function interp1 "Piecewise-linear interpolation with linear extrapolation"
      input Real x;
      input Real xs[:];
      input Real ys[size(xs, 1)];
      output Real y;
    protected
      Integer n = size(xs, 1);
      Integer i;
    algorithm
      if x <= xs[1] then
        y := ys[1] + (ys[2] - ys[1])/(xs[2] - xs[1])*(x - xs[1]);
      elseif x >= xs[n] then
        y := ys[n] + (ys[n] - ys[n - 1])/(xs[n] - xs[n - 1])*(x - xs[n]);
      else
        i := 1;
        while xs[i + 1] < x loop
          i := i + 1;
        end while;
        y := ys[i] + (ys[i + 1] - ys[i])/(xs[i + 1] - xs[i])*(x - xs[i]);
      end if;
      annotation (Inline=false);
    end interp1;

    function solve "Solve A*X = B (Gauss elimination with partial pivoting)"
      input Real A[:, size(A, 1)];
      input Real B[size(A, 1), :];
      output Real X[size(A, 1), size(B, 2)];
    protected
      Integer n = size(A, 1);
      Real M[size(A, 1), size(A, 1)] = A;
      Real R[size(A, 1), size(B, 2)] = B;
      Integer p;
      Real f;
      Real tmp;
    algorithm
      for k in 1:n loop
        p := k;
        for i in k + 1:n loop
          if abs(M[i, k]) > abs(M[p, k]) then
            p := i;
          end if;
        end for;
        if p <> k then
          for j in 1:n loop
            tmp := M[k, j]; M[k, j] := M[p, j]; M[p, j] := tmp;
          end for;
          for j in 1:size(B, 2) loop
            tmp := R[k, j]; R[k, j] := R[p, j]; R[p, j] := tmp;
          end for;
        end if;
        for i in k + 1:n loop
          f := M[i, k]/M[k, k];
          for j in k:n loop
            M[i, j] := M[i, j] - f*M[k, j];
          end for;
          for j in 1:size(B, 2) loop
            R[i, j] := R[i, j] - f*R[k, j];
          end for;
        end for;
      end for;
      for i in n:-1:1 loop
        for j in 1:size(B, 2) loop
          X[i, j] := (R[i, j] - sum(M[i, l]*X[l, j] for l in i + 1:n))/M[i, i];
        end for;
      end for;
    end solve;

    function shaftCondensedK
      "3x3 lateral stiffness (end a, middle, end b) of a 2-element Euler-Bernoulli shaft, nodal rotations condensed out"
      input Real E(unit="N/m2");
      input Real dAo(unit="m");
      input Real dAi(unit="m");
      input Real L1(unit="m");
      input Real L2(unit="m");
      output Real K[3, 3](each unit="N/m");
    protected
      Real I = GearMeshBenchmark.pi/64*(dAo^4 - dAi^4);
      Real Kf[6, 6] "dofs v1, p1, v2, p2, v3, p3";
      Real Ktt[3, 3];
      Real Ktr[3, 3];
      Real Krr[3, 3];
      Integer it[3] = {1, 3, 5};
      Integer ir[3] = {2, 4, 6};
      Real ke[4, 4];
      Real L;
    algorithm
      Kf := zeros(6, 6);
      for e in 1:2 loop
        L := if e == 1 then L1 else L2;
        ke := E*I/L^3*[12, 6*L, -12, 6*L; 6*L, 4*L^2, -6*L, 2*L^2; -12, -6*L, 12, -6*L; 6*L, 2*L^2, -6*L, 4*L^2];
        for i in 1:4 loop
          for j in 1:4 loop
            Kf[2*(e - 1) + i, 2*(e - 1) + j] := Kf[2*(e - 1) + i, 2*(e - 1) + j] + ke[i, j];
          end for;
        end for;
      end for;
      for i in 1:3 loop
        for j in 1:3 loop
          Ktt[i, j] := Kf[it[i], it[j]];
          Ktr[i, j] := Kf[it[i], ir[j]];
          Krr[i, j] := Kf[ir[i], ir[j]];
        end for;
      end for;
      K := Ktt - Ktr*solve(Krr, transpose(Ktr));
    end shaftCondensedK;
  end Functions;

  // ---------------------------------------------------------------------------
  package Components
    model Fixed "Ground node"
      Frame frame annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
    equation
      frame.u = 0; frame.v = 0; frame.phi = 0;
      annotation (Icon(graphics={Line(points={{-60,-20},{60,-20}}, thickness=1),
            Line(points={{-50,-20},{-70,-50}}), Line(points={{0,-20},{-20,-50}}), Line(points={{50,-20},{30,-50}})}));
    end Fixed;

    model ConstantSpeed "Prescribed rotation phi = Omega*time (no translation)"
      parameter Real Omega(unit="rad/s") "speed";
      Frame frame annotation (Placement(transformation(extent={{90,-10},{110,10}})));
    equation
      frame.u = 0; frame.v = 0; frame.phi = Omega*time;
      annotation (Icon(graphics={Ellipse(extent={{-60,60},{60,-60}}, lineColor={255,0,0}, thickness=1),
            Text(extent={{-100,-70},{100,-100}}, textString="%Omega rad/s")}));
    end ConstantSpeed;

    model Body "Rigid gear body: translations + rotation"
      parameter Real m(unit="kg") "mass";
      parameter Real J(unit="kg.m2") "polar inertia";
      parameter Real w0(unit="rad/s") = 0 "initial angular velocity";
      Frame frame annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
      Real u(unit="m", start=0, fixed=true);
      Real v(unit="m", start=0, fixed=true);
      Real phi(unit="rad", start=0, fixed=true);
      Real du(unit="m/s", start=0, fixed=true);
      Real dv(unit="m/s", start=0, fixed=true);
      Real w(unit="rad/s", start=w0, fixed=true);
    equation
      frame.u = u; frame.v = v; frame.phi = phi;
      der(u) = du; der(v) = dv; der(phi) = w;
      m*der(du) = frame.fx;
      m*der(dv) = frame.fy;
      J*der(w) = frame.t;
      annotation (Icon(graphics={Ellipse(extent={{-80,80},{80,-80}}, lineColor={0,0,0},
            fillColor={200,200,200}, fillPattern=FillPattern.Solid), Text(extent={{-100,-85},{100,-110}}, textString="%name")}));
    end Body;

    model PointMass "Translational node (no rotational dof, phi = 0)"
      parameter Real m(unit="kg");
      Frame frame annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
      Real u(unit="m", start=0, fixed=true);
      Real v(unit="m", start=0, fixed=true);
      Real du(unit="m/s", start=0, fixed=true);
      Real dv(unit="m/s", start=0, fixed=true);
    equation
      frame.u = u; frame.v = v; frame.phi = 0;
      der(u) = du; der(v) = dv;
      m*der(du) = frame.fx;
      m*der(dv) = frame.fy;
      annotation (Icon(graphics={Ellipse(extent={{-40,40},{40,-40}}, fillColor={0,0,255}, fillPattern=FillPattern.Solid),
            Text(extent={{-100,-45},{100,-70}}, textString="%name")}));
    end PointMass;

    model SpringDamper2D "Linear x/y spring-damper between two frames (no torque)"
      parameter Real kx(unit="N/m");
      parameter Real ky(unit="N/m") = kx;
      parameter Real dx(unit="N.s/m");
      parameter Real dy(unit="N.s/m") = dx;
      Frame frame_a annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Frame frame_b annotation (Placement(transformation(extent={{90,-10},{110,10}})));
      Real fx(unit="N") "force acting on frame_b side (tension positive)";
      Real fy(unit="N");
    equation
      fx = kx*(frame_b.u - frame_a.u) + dx*(der(frame_b.u) - der(frame_a.u));
      fy = ky*(frame_b.v - frame_a.v) + dy*(der(frame_b.v) - der(frame_a.v));
      frame_b.fx = fx; frame_b.fy = fy; frame_a.fx = -fx; frame_a.fy = -fy;
      frame_a.t = 0; frame_b.t = 0;
      annotation (Icon(graphics={Line(points={{-90,0},{-60,0},{-50,20},{-30,-20},{-10,20},{10,-20},{30,20},{50,-20},{60,0},{90,0}})}));
    end SpringDamper2D;

    model TorsionSpringDamper "Rotational spring-damper between two frames"
      parameter Real k(unit="N.m/rad");
      parameter Real d(unit="N.m.s/rad");
      Frame frame_a annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Frame frame_b annotation (Placement(transformation(extent={{90,-10},{110,10}})));
      Real tau(unit="N.m") "torque transmitted from a to b";
    equation
      tau = k*(frame_a.phi - frame_b.phi) + d*(der(frame_a.phi) - der(frame_b.phi));
      frame_b.t = -tau; frame_a.t = tau;
      frame_a.fx = 0; frame_a.fy = 0; frame_b.fx = 0; frame_b.fy = 0;
      annotation (Icon(graphics={Line(points={{-90,0},{-60,0},{-50,20},{-30,-20},{-10,20},{10,-20},{30,20},{50,-20},{60,0},{90,0}}, color={255,0,0})}));
    end TorsionSpringDamper;

    model ConstantTorque "Torque acting on the frame (ValDyn: zero-stiffness spring with preload)"
      parameter Real T(unit="N.m");
      Frame frame annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
    equation
      frame.t = -T; frame.fx = 0; frame.fy = 0;
      annotation (Icon(graphics={Line(points={{-90,0},{-40,0}}), Line(points={{-40,40},{40,40},{40,-40}}, color={255,0,0}, arrow={Arrow.None,Arrow.Filled}),
            Text(extent={{-100,-50},{100,-80}}, textString="%name")}));
    end ConstantTorque;

    model SineAngleTorque "Torque T = A*sin(driver angle), driver angle = Omega*time (SINUS10.txt)"
      parameter Real A(unit="N.m");
      parameter Real Omega(unit="rad/s");
      Frame frame annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Real T(unit="N.m");
    equation
      T = A*sin(Omega*time);
      frame.t = -T; frame.fx = 0; frame.fy = 0;
      annotation (Icon(graphics={Line(points={{-90,0},{-40,0}}), Line(points={{-40,40},{40,40},{40,-40}}, color={255,0,0}, arrow={Arrow.None,Arrow.Filled}),
            Text(extent={{-100,-50},{100,-80}}, textString="%name")}));
    end SineAngleTorque;

    model GearMesh2D "Planar spur-gear mesh with backlash (ValDyn GEARMESH2D equivalent)"
      parameter Real k(unit="N/m") = 3.08e8 "tooth stiffness (ValDyn 308000 N/mm)";
      parameter Real c(unit="N.s/m") = 4300 "tooth damping";
      parameter Integer z1 = 40 "teeth gear a (i)";
      parameter Integer z2 = 30 "teeth gear b (j)";
      parameter Real a(unit="m") = 0.140 "centre distance (for pitch radii)";
      parameter Real alpha_deg = 20 "pressure angle [deg]";
      parameter Real jn(unit="m") = 0.25e-3 "initial NORMAL backlash";
      parameter Real gapSplit = 0.5 "share of backlash on side A at start (0.5 = centred)";
      parameter Real xb(unit="m") = 0.13156 "nominal position of gear b relative to gear a";
      parameter Real yb(unit="m") = 0.04788;
      parameter Boolean clipTension = false "true: contact force >= 0 (damping clipped)";
      final parameter Real alpha = alpha_deg*GearMeshBenchmark.pi/180;
      final parameter Real ex = xb/sqrt(xb^2 + yb^2) "centre line";
      final parameter Real ey = yb/sqrt(xb^2 + yb^2);
      final parameter Real tx = -ey;
      final parameter Real ty = ex;
      final parameter Real nAx = cos(alpha)*tx + sin(alpha)*ex "drive flank: force direction on gear b";
      final parameter Real nAy = cos(alpha)*ty + sin(alpha)*ey;
      final parameter Real nBx = -cos(alpha)*tx + sin(alpha)*ex "coast flank";
      final parameter Real nBy = -cos(alpha)*ty + sin(alpha)*ey;
      final parameter Real rb1(unit="m") = a*z1/(z1 + z2)*cos(alpha) "base radius a";
      final parameter Real rb2(unit="m") = a*z2/(z1 + z2)*cos(alpha) "base radius b";
      final parameter Real gA0 = gapSplit*jn;
      final parameter Real gB0 = (1 - gapSplit)*jn;
      Frame frame_a "gear i (driving)" annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Frame frame_b "gear j" annotation (Placement(transformation(extent={{90,-10},{110,10}})));
      Real dA(unit="m") "penetration flank A (>0: contact)";
      Real dB(unit="m") "penetration flank B";
      Real FA(unit="N") "tooth force side A";
      Real FB(unit="N") "tooth force side B";
      Real Ft(unit="N") "tangential force (A positive)";
      Real lashA_mm "lash side A on pitch circle [mm] (ValDyn LASH)";
      Real lashB_mm "lash side B on pitch circle [mm]";
    protected
      Real rot(unit="m");
      Real FAraw;
      Real FBraw;
    equation
      rot = rb1*frame_a.phi + rb2*frame_b.phi;
      dA = rot + (frame_a.u - frame_b.u)*nAx + (frame_a.v - frame_b.v)*nAy - gA0;
      dB = -rot + (frame_a.u - frame_b.u)*nBx + (frame_a.v - frame_b.v)*nBy - gB0;
      FAraw = k*dA + c*der(dA);
      FBraw = k*dB + c*der(dB);
      FA = if dA > 0 then (if clipTension then max(FAraw, 0) else FAraw) else 0;
      FB = if dB > 0 then (if clipTension then max(FBraw, 0) else FBraw) else 0;
      // cut forces acting on the mesh element (= minus the forces it applies to the gears)
      frame_b.fx = -(FA*nAx + FB*nBx);
      frame_b.fy = -(FA*nAy + FB*nBy);
      frame_a.fx = FA*nAx + FB*nBx;
      frame_a.fy = FA*nAy + FB*nBy;
      frame_a.t = (FA - FB)*rb1;
      frame_b.t = (FA - FB)*rb2;
      Ft = (FA - FB)*cos(alpha);
      lashA_mm = -dA/cos(alpha)*1e3;
      lashB_mm = -dB/cos(alpha)*1e3;
      annotation (Icon(graphics={Ellipse(extent={{-90,50},{10,-50}}, lineColor={0,0,0}, pattern=LinePattern.Dash),
            Ellipse(extent={{0,40},{80,-40}}, lineColor={0,0,0}, pattern=LinePattern.Dash),
            Text(extent={{-100,-60},{100,-90}}, textString="GEARMESH2D")}));
    end GearMesh2D;

    model RadialBearingClearance "Simplified radial bearing with clearance (frame_a: shaft/inner, frame_b: outer/gear)"
      parameter Real k(unit="N/m") = 5e8 "radial stiffness (500000 N/mm)";
      parameter Real c(unit="N.s/m") = 1.26 "radial damping";
      parameter Real clr(unit="m") = 30e-6 "radial clearance (free radius)";
      parameter Integer mode = 0 "0: circular, 1: per plane (XZ/YZ), 2: clearance only in x";
      parameter Real mu = 0.001 "rotational friction coefficient";
      parameter Real dFric(unit="m") = 0.028 "diameter for friction";
      parameter Real wReg(unit="rad/s") = 1 "regularisation speed for friction";
      Frame frame_a annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Frame frame_b annotation (Placement(transformation(extent={{90,-10},{110,10}})));
      Real rx(unit="m"); Real ry(unit="m"); Real r(unit="m");
      Real Fn(unit="N") "normal (radial) force magnitude";
      Real Fx(unit="N") "force on outer ring (gear) x";
      Real Fy(unit="N");
      Real Tf(unit="N.m") "friction torque on outer ring";
    protected
      Real Fn_c;
      Real cy = if mode == 2 then 0 else clr;
    equation
      rx = frame_b.u - frame_a.u;
      ry = frame_b.v - frame_a.v;
      r = sqrt(rx^2 + ry^2 + 1e-24);
      Fn_c = if r > clr then max(k*(r - clr) + c*(rx*der(rx) + ry*der(ry))/r, 0) else 0;
      if mode == 0 then
        Fx = -Fn_c*rx/r;
        Fy = -Fn_c*ry/r;
      else
        Fx = if rx > clr then -(k*(rx - clr) + c*der(rx)) elseif rx < -clr then -(k*(rx + clr) + c*der(rx)) else 0;
        Fy = if mode == 2 then -(k*ry + c*der(ry)) else
             (if ry > cy then -(k*(ry - cy) + c*der(ry)) elseif ry < -cy then -(k*(ry + cy) + c*der(ry)) else 0);
      end if;
      Fn = sqrt(Fx^2 + Fy^2);
      Tf = -mu*Fn*dFric/2*tanh(der(frame_b.phi)/wReg);
      frame_b.fx = -Fx; frame_b.fy = -Fy; frame_b.t = -Tf;
      frame_a.fx = Fx; frame_a.fy = Fy; frame_a.t = Tf;
      annotation (Icon(graphics={Rectangle(extent={{-60,60},{60,-60}}, lineColor={0,0,0}),
            Line(points={{-60,60},{60,-60}}), Line(points={{-60,-60},{60,60}})}));
    end RadialBearingClearance;

    model ShaftCondensed "Non-rotating hollow shaft, 2 beam elements, 3 lateral nodes (rotations condensed)"
      parameter Real E(unit="N/m2") = 2.1e11;
      parameter Real dAo(unit="m") = 0.026;
      parameter Real dAi(unit="m") = 0.020;
      parameter Real L1(unit="m") = 0.050;
      parameter Real L2(unit="m") = 0.050;
      parameter Real cBend(unit="N.s/m") = 100 "bending damping (stiffness proportional, = value at mid node)";
      final parameter Real K[3, 3] = Functions.shaftCondensedK(E, dAo, dAi, L1, L2);
      final parameter Real C[3, 3] = cBend*K/K[2, 2];
      Frame frame_a "block end" annotation (Placement(transformation(extent={{-110,-10},{-90,10}})));
      Frame frame_m "middle (bearing)" annotation (Placement(transformation(extent={{-10,-110},{10,-90}})));
      Frame frame_b "case end" annotation (Placement(transformation(extent={{90,-10},{110,10}})));
    protected
      Real U[3]; Real V[3];
    equation
      U = {frame_a.u, frame_m.u, frame_b.u};
      V = {frame_a.v, frame_m.v, frame_b.v};
      {frame_a.fx, frame_m.fx, frame_b.fx} = K*U + C*der(U);
      {frame_a.fy, frame_m.fy, frame_b.fy} = K*V + C*der(V);
      frame_a.t = 0; frame_m.t = 0; frame_b.t = 0;
      annotation (Icon(graphics={Rectangle(extent={{-90,15},{90,-15}}, lineColor={0,0,0},
            fillColor={180,180,180}, fillPattern=FillPattern.Solid)}));
    end ShaftCondensed;

    model NonlinearSupport2D "Non-linear x/y support to ground from ValDyn tables {disp [mm], force [N], damping [N.s/m]}"
      parameter Real tabX[:, 3] "x table (mm, N, N.s/m)";
      parameter Real tabY[:, 3] "y table (mm, N, N.s/m)";
      parameter Real dampScale = 1 "multiplier on damping column";
      Frame frame annotation (Placement(transformation(extent={{-10,90},{10,110}})));
      Real Fx(unit="N") "support (table) force x; force on the shaft = -Fx";
      Real Fy(unit="N");
    equation
      Fx = Functions.interp1(frame.u*1e3, tabX[:, 1], tabX[:, 2])
         + dampScale*Functions.interp1(frame.u*1e3, tabX[:, 1], tabX[:, 3])*der(frame.u);
      Fy = Functions.interp1(frame.v*1e3, tabY[:, 1], tabY[:, 2])
         + dampScale*Functions.interp1(frame.v*1e3, tabY[:, 1], tabY[:, 3])*der(frame.v);
      frame.fx = Fx; frame.fy = Fy; frame.t = 0;
      annotation (Icon(graphics={Line(points={{0,90},{0,60},{20,50},{-20,30},{20,10},{-20,-10},{0,-20},{0,-40}}),
            Line(points={{-40,-40},{40,-40}}, thickness=1)}));
    end NonlinearSupport2D;
  end Components;

  // ---------------------------------------------------------------------------
  package Examples
    model Stage1 "ValDyn GEAR PAIR MODEL with GEARMESH2D element (exact mask values)"
      import GearMeshBenchmark.Components.*;
      parameter Real speed_rpm = 2500;
      final parameter Real Omega = speed_rpm*GearMeshBenchmark.pi/30;
      parameter Real w2_ratio = -1.37 "initial speed gear 2 [deg/deg]";
      parameter Real T_drag(unit="N.m") = 10;
      parameter Real k_mesh(unit="N/m") = 3.08e8;
      parameter Real c_mesh(unit="N.s/m") = 4300;
      parameter Real jn(unit="m") = 0.25e-3 "normal backlash";
      parameter Boolean clipTension = false;
      parameter Real gapSplit = 0.5;
      ConstantSpeed driver(Omega=Omega) annotation (Placement(transformation(extent={{-100,-10},{-80,10}})));
      TorsionSpringDamper driveSpring(k=4.3e7, d=172) annotation (Placement(transformation(extent={{-70,-10},{-50,10}})));
      Body gear1(m=2.67, J=0.01709, w0=Omega) annotation (Placement(transformation(extent={{-40,-10},{-20,10}})));
      Body gear2(m=1.5, J=0.0054, w0=w2_ratio*Omega) annotation (Placement(transformation(extent={{20,-10},{40,10}})));
      Fixed ground1 annotation (Placement(transformation(extent={{-70,-70},{-50,-50}})));
      Fixed ground2 annotation (Placement(transformation(extent={{60,-70},{80,-50}})));
      SpringDamper2D support1(kx=3.6e9, dx=10218) "BUSH_1 x/y" annotation (Placement(transformation(extent={{-55,-50},{-35,-30}})));
      SpringDamper2D support2(kx=3.6e9, dx=7397) "BUSH_2 x/y" annotation (Placement(transformation(extent={{40,-50},{60,-30}})));
      ConstantTorque drag(T=T_drag) annotation (Placement(transformation(extent={{55,10},{75,30}})));
      GearMesh2D mesh(k=k_mesh, c=c_mesh, jn=jn, clipTension=clipTension, gapSplit=gapSplit) annotation (Placement(transformation(extent={{-10,-10},{10,10}})));
      // outputs in ValDyn plot units
      Real angle_deg = Omega*time*180/GearMeshBenchmark.pi "rotation angle (driver)";
      Real lashA_mm = mesh.lashA_mm;
      Real lashB_mm = mesh.lashB_mm;
      Real FA = mesh.FA;
      Real FB = mesh.FB;
      Real x1_mm = gear1.u*1e3;
      Real y1_mm = gear1.v*1e3;
      Real x2_mm = 131.56 + gear2.u*1e3;
      Real y2_mm = 47.88 + gear2.v*1e3;
      Real Fs1x = support1.fx "support force on gear 1 seen by ValDyn plot 'X Masse 1' (sign: -)";
    equation
      connect(driver.frame, driveSpring.frame_a) annotation (Line(points={{-80,0},{-70,0}}));
      connect(driveSpring.frame_b, gear1.frame) annotation (Line(points={{-50,0},{-30,0}}));
      connect(ground1.frame, support1.frame_a) annotation (Line(points={{-60,-60},{-55,-40}}));
      connect(support1.frame_b, gear1.frame) annotation (Line(points={{-35,-40},{-30,0}}));
      connect(ground2.frame, support2.frame_a) annotation (Line(points={{70,-60},{40,-40}}));
      connect(support2.frame_b, gear2.frame) annotation (Line(points={{60,-40},{30,0}}));
      connect(drag.frame, gear2.frame) annotation (Line(points={{55,20},{30,0}}));
      connect(mesh.frame_a, gear1.frame) annotation (Line(points={{-10,0},{-30,0}}));
      connect(mesh.frame_b, gear2.frame) annotation (Line(points={{10,0},{30,0}}));
      annotation (experiment(StartTime=0, StopTime=0.024, Interval=1e-6, Tolerance=1e-8, __OpenModelica_Algorithm="dassl"),
        Documentation(info="<html><p>360 deg of driver rotation at 2500 rpm = 24 ms.</p></html>"));
    end Stage1;

    model Stage2 "Gear pair with gear 2 on a fixed shaft (bearing with clearance, non-linear supports) - mask values"
      import GearMeshBenchmark.Components.*;
      parameter Real speed_rpm = 3000 "result files labelled 03000";
      final parameter Real Omega = speed_rpm*GearMeshBenchmark.pi/30;
      parameter Real w2_ratio = -40/30;
      parameter Real T_amp(unit="N.m") = 10 "SINUS10.txt";
      parameter Real k_mesh(unit="N/m") = 3.08e8;
      parameter Real c_mesh(unit="N.s/m") = 4300;
      parameter Real jn(unit="m") = 0.25e-3;
      parameter Real clr(unit="m") = 30e-6 "bearing radial clearance (free radius)";
      parameter Integer clrMode = 0 "0 circular, 1 per plane, 2 x only";
      parameter Real dampScale = 1 "support table damping unit factor";
      // shaft and node masses (as typed in the ValDyn masks; MASS_8 without /2*pi/4)
      parameter Real rhoA = 7850, dAo = 0.026, dAi = 0.020, lA1 = 0.050, lA2 = 0.050;
      parameter Real m8 = rhoA*(dAo^2 - dAi^2)*lA1;
      parameter Real mM = rhoA*(dAo^2 - dAi^2)*(lA1 + lA2)/2*GearMeshBenchmark.pi/4;
      parameter Real m9 = rhoA*(dAo^2 - dAi^2)*lA2/2*GearMeshBenchmark.pi/4;
      parameter Real tabBlock[:, 3] = [-2, -2e6, 200; -1, -2e5, 20; -0.002, -10, 20; 0, 0, 20; 0.002, 10, 20; 1, 2e5, 20; 2, 2e6, 200];
      parameter Real tabCase[:, 3] = [-2, -1e6, 200; -1, -1e5, 20; -0.002, -10, 20; 0, 0, 20; 0.002, 10, 20; 1, 1e5, 20; 2, 1e6, 200];
      ConstantSpeed driver(Omega=Omega) annotation (Placement(transformation(extent={{-100,-50},{-80,-30}})));
      TorsionSpringDamper driveSpring(k=4.3e7, d=172) annotation (Placement(transformation(extent={{-70,-50},{-50,-30}})));
      Body gear1(m=2.67, J=0.01709, w0=Omega) annotation (Placement(transformation(extent={{-40,-50},{-20,-30}})));
      Body gear2(m=1.5, J=0.0054, w0=w2_ratio*Omega) annotation (Placement(transformation(extent={{20,-50},{40,-30}})));
      Fixed ground1 annotation (Placement(transformation(extent={{-70,-100},{-50,-80}})));
      SpringDamper2D support1(kx=3.6e9, dx=10218) annotation (Placement(transformation(extent={{-55,-80},{-35,-60}})));
      SineAngleTorque load(A=T_amp, Omega=Omega) annotation (Placement(transformation(extent={{50,-70},{70,-50}})));
      GearMesh2D mesh(k=k_mesh, c=c_mesh, jn=jn) annotation (Placement(transformation(extent={{-10,-50},{10,-30}})));
      RadialBearingClearance bearing(clr=clr, mode=clrMode) annotation (Placement(transformation(extent={{20,-10},{40,10}})));
      ShaftCondensed shaft(dAo=dAo, dAi=dAi, L1=lA1, L2=lA2) annotation (Placement(transformation(extent={{20,40},{40,60}})));
      PointMass node8(m=m8) "shaft end at block (MASS_8)" annotation (Placement(transformation(extent={{-10,40},{10,60}})));
      PointMass nodeM(m=mM) "shaft node under the bearing (R_BEARING_1 nodes)" annotation (Placement(transformation(extent={{20,20},{40,40}})));
      PointMass node9(m=m9) "shaft end at case (MASS_9)" annotation (Placement(transformation(extent={{50,40},{70,60}})));
      NonlinearSupport2D supBlock(tabX=tabBlock, tabY=tabBlock, dampScale=dampScale) annotation (Placement(transformation(extent={{-10,10},{10,30}})));
      NonlinearSupport2D supCase(tabX=tabCase, tabY=tabCase, dampScale=dampScale) annotation (Placement(transformation(extent={{50,10},{70,30}})));
      // outputs in ValDyn plot units
      Real angle_deg = Omega*time*180/GearMeshBenchmark.pi;
      Real Tload = load.T;
      Real Ft = mesh.Ft;
      Real lashA_mm = mesh.lashA_mm;
      Real lashB_mm = mesh.lashB_mm;
      Real gearX_mm = 131.56 + gear2.u*1e3 "compare with ValDyn 'BearingX'";
      Real gearY_mm = 47.88 + gear2.v*1e3;
      Real midX_mm = 131.56 + nodeM.u*1e3;
      Real midY_mm = 47.88 + nodeM.v*1e3 "compare with ValDyn 'BearingY'";
      Real blockX_mm = 131.56 + node8.u*1e3;
      Real blockY_mm = 47.88 + node8.v*1e3;
      Real caseX_mm = 131.56 + node9.u*1e3;
      Real caseY_mm = 47.88 + node9.v*1e3;
      Real FXBlock = -supBlock.Fx "force on the shaft (ValDyn sign)";
      Real FYBlock = -supBlock.Fy;
      Real FXCase = -supCase.Fx;
      Real FYCase = -supCase.Fy;
    equation
      connect(driver.frame, driveSpring.frame_a) annotation (Line(points={{-80,-40},{-70,-40}}));
      connect(driveSpring.frame_b, gear1.frame) annotation (Line(points={{-50,-40},{-30,-40}}));
      connect(ground1.frame, support1.frame_a) annotation (Line(points={{-60,-90},{-55,-70}}));
      connect(support1.frame_b, gear1.frame) annotation (Line(points={{-35,-70},{-30,-40}}));
      connect(mesh.frame_a, gear1.frame) annotation (Line(points={{-10,-40},{-30,-40}}));
      connect(mesh.frame_b, gear2.frame) annotation (Line(points={{10,-40},{30,-40}}));
      connect(load.frame, gear2.frame) annotation (Line(points={{50,-60},{30,-40}}));
      connect(bearing.frame_b, gear2.frame) annotation (Line(points={{40,0},{30,-40}}));
      connect(bearing.frame_a, nodeM.frame) annotation (Line(points={{20,0},{30,30}}));
      connect(shaft.frame_m, nodeM.frame) annotation (Line(points={{30,40},{30,30}}));
      connect(shaft.frame_a, node8.frame) annotation (Line(points={{20,50},{0,50}}));
      connect(shaft.frame_b, node9.frame) annotation (Line(points={{40,50},{60,50}}));
      connect(supBlock.frame, node8.frame) annotation (Line(points={{0,30},{0,50}}));
      connect(supCase.frame, node9.frame) annotation (Line(points={{60,30},{60,50}}));
      annotation (experiment(StartTime=0, StopTime=0.08, Interval=2e-6, Tolerance=1e-8, __OpenModelica_Algorithm="dassl"),
        Documentation(info="<html><p>1440 deg of driver rotation at 3000 rpm = 80 ms. With the mask values
(backlash 0.25 mm, 308000 N/mm) the result does NOT match ValDyn - see Stage2Calibrated.</p></html>"));
    end Stage2;

    model Stage2Calibrated "Stage 2 with calibrated backlash and mesh stiffness"
      extends Stage2(jn=0.012e-3, k_mesh=2.5e7);
      annotation (experiment(StartTime=0, StopTime=0.08, Interval=2e-6, Tolerance=1e-8, __OpenModelica_Algorithm="dassl"));
    end Stage2Calibrated;
  end Examples;
end GearMeshBenchmark;
