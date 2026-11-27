within ;
package CamtrainProto "Prototype: structuring pattern for camtrains built from GearMeshBenchmark components"
  import GearMeshBenchmark.Components;
  constant Real pi = GearMeshBenchmark.pi;

  package Data "Parameter records"
    record GearData
      parameter Integer z = 30 "teeth";
      parameter Real m(unit="kg") = 1.5;
      parameter Real J(unit="kg.m2") = 0.0054;
    end GearData;
    record MeshData
      parameter Real k(unit="N/m") = 3.08e8;
      parameter Real c(unit="N.s/m") = 4300;
      parameter Real jn(unit="m") = 0.25e-3;
      parameter Real alpha_deg = 20;
    end MeshData;
    record BearingData
      parameter Real k(unit="N/m") = 5e8;
      parameter Real c(unit="N.s/m") = 1.26;
      parameter Real clr(unit="m") = 30e-6;
    end BearingData;
    record ShaftData
      parameter Real dAo(unit="m") = 0.026;
      parameter Real dAi(unit="m") = 0.020;
      parameter Real L1(unit="m") = 0.050;
      parameter Real L2(unit="m") = 0.050;
      parameter Real E(unit="N/m2") = 2.1e11;
      parameter Real rho(unit="kg/m3") = 7850;
      parameter Real cBend(unit="N.s/m") = 100;
    end ShaftData;
    record SupportData
      parameter Real tabBlock[:, 3] = [-2, -2e6, 200; -1, -2e5, 20; -0.002, -10, 20; 0, 0, 20; 0.002, 10, 20; 1, 2e5, 20; 2, 2e6, 200];
      parameter Real tabCase[:, 3] = [-2, -1e6, 200; -1, -1e5, 20; -0.002, -10, 20; 0, 0, 20; 0.002, 10, 20; 1, 1e5, 20; 2, 1e6, 200];
    end SupportData;
    record GearShaftData "one gear on its shaft in the crankcase"
      parameter Real x(unit="m") = 0 "shaft centre in the case";
      parameter Real y(unit="m") = 0;
      GearData gear;
      ShaftData shaft;
      BearingData bearing;
      SupportData support;
    end GearShaftData;
    record Stage2Data "system record: layout + all components"
      parameter Real speed_rpm = 3000;
      parameter Real T_amp = 10;
      GearShaftData g2(x=0.13156, y=0.04788);
      MeshData mesh;
    end Stage2Data;
    record Stage2Calibrated = Stage2Data(mesh(k=2.5e7, jn=0.012e-3)) "variant: only the differences";
  end Data;

  model Environment "inner: global crank speed and angle, used by all assemblies"
    parameter Real speed_rpm = 3000;
    final parameter Real Omega = speed_rpm*pi/30;
    Real angle(unit="rad") = Omega*time;
    annotation (defaultComponentName="env", defaultComponentPrefixes="inner");
  end Environment;

  package Interfaces
    partial model PartialGearAssembly "anything that carries a meshing gear"
      parameter Data.GearShaftData data;
      parameter Real ratio "speed of this gear / crank speed (signed)";
      outer Environment env;
      GearMeshBenchmark.Frame gearFrame "connect meshes and loads here";
    end PartialGearAssembly;
  end Interfaces;

  package Assemblies
    model IdlerOnFixedShaft "gear on bearing with clearance, non-rotating shaft on Block/Case tables (Stage 2 pattern)"
      extends Interfaces.PartialGearAssembly;
      Components.Body gear(m=data.gear.m, J=data.gear.J, w0=ratio*env.Omega);
      Components.RadialBearingClearance bearing(k=data.bearing.k, c=data.bearing.c, clr=data.bearing.clr);
      Components.ShaftCondensed shaft(dAo=data.shaft.dAo, dAi=data.shaft.dAi, L1=data.shaft.L1, L2=data.shaft.L2,
        E=data.shaft.E, cBend=data.shaft.cBend);
      Components.PointMass nodeBlock(m=data.shaft.rho*(data.shaft.dAo^2 - data.shaft.dAi^2)*data.shaft.L1);
      Components.PointMass nodeM(m=data.shaft.rho*(data.shaft.dAo^2 - data.shaft.dAi^2)*(data.shaft.L1 + data.shaft.L2)/2*pi/4);
      Components.PointMass nodeCase(m=data.shaft.rho*(data.shaft.dAo^2 - data.shaft.dAi^2)*data.shaft.L2/2*pi/4);
      Components.NonlinearSupport2D supBlock(tabX=data.support.tabBlock, tabY=data.support.tabBlock);
      Components.NonlinearSupport2D supCase(tabX=data.support.tabCase, tabY=data.support.tabCase);
    equation
      connect(gear.frame, gearFrame);
      connect(bearing.frame_b, gear.frame);
      connect(bearing.frame_a, nodeM.frame);
      connect(shaft.frame_m, nodeM.frame);
      connect(shaft.frame_a, nodeBlock.frame);
      connect(shaft.frame_b, nodeCase.frame);
      connect(supBlock.frame, nodeBlock.frame);
      connect(supCase.frame, nodeCase.frame);
    end IdlerOnFixedShaft;

    model ElasticBody "gear on linear x/y supports (Stage 1 pattern)"
      extends Interfaces.PartialGearAssembly;
      parameter Real k(unit="N/m") = 3.6e9;
      parameter Real d(unit="N.s/m") = 7397;
      Components.Body gear(m=data.gear.m, J=data.gear.J, w0=ratio*env.Omega);
      Components.SpringDamper2D support(kx=k, dx=d);
      Components.Fixed ground;
    equation
      connect(gear.frame, gearFrame);
      connect(ground.frame, support.frame_a);
      connect(support.frame_b, gear.frame);
    end ElasticBody;
  end Assemblies;

  model GearMesh "mesh between two assemblies: geometry taken from their data records"
    parameter Data.MeshData data;
    parameter Data.GearShaftData a;
    parameter Data.GearShaftData b;
    GearMeshBenchmark.Frame frame_a;
    GearMeshBenchmark.Frame frame_b;
    Components.GearMesh2D mesh(k=data.k, c=data.c, jn=data.jn, alpha_deg=data.alpha_deg, z1=a.gear.z, z2=b.gear.z,
      a=sqrt((b.x - a.x)^2 + (b.y - a.y)^2), xb=b.x - a.x, yb=b.y - a.y);
  equation
    connect(mesh.frame_a, frame_a);
    connect(mesh.frame_b, frame_b);
  end GearMesh;

  model Stage2Rebuilt "Stage 2 rebuilt from the assembly + records; must equal GearMeshBenchmark.Examples.Stage2"
    replaceable parameter Data.Stage2Data data constrainedby Data.Stage2Data annotation (choicesAllMatching=true);
    inner Environment env(speed_rpm=data.speed_rpm);
    parameter Data.GearShaftData g1(gear(z=40, m=2.67, J=0.01709));
    Components.ConstantSpeed driver(Omega=env.Omega);
    Components.TorsionSpringDamper driveSpring(k=4.3e7, d=172);
    Assemblies.ElasticBody gear1(data=g1, ratio=1, k=3.6e9, d=10218);
    replaceable Assemblies.IdlerOnFixedShaft gear2(data=data.g2, ratio=-g1.gear.z/data.g2.gear.z)
      constrainedby Interfaces.PartialGearAssembly annotation (choicesAllMatching=true);
    GearMesh mesh(data=data.mesh, a=g1, b=data.g2);
    Components.SineAngleTorque load(A=data.T_amp, Omega=env.Omega);
    Real angle_deg = env.angle*180/pi;
  equation
    connect(driver.frame, driveSpring.frame_a);
    connect(driveSpring.frame_b, gear1.gearFrame);
    connect(mesh.frame_a, gear1.gearFrame);
    connect(mesh.frame_b, gear2.gearFrame);
    connect(load.frame, gear2.gearFrame);
    annotation (experiment(StopTime=0.08, Interval=2e-6, Tolerance=1e-8));
  end Stage2Rebuilt;

  model Stage2RebuiltCal = Stage2Rebuilt(redeclare parameter Data.Stage2Calibrated data);
end CamtrainProto;
