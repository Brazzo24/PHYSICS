model ArrTest "N gears in a row: arrays of assemblies + records, meshes connected in a for-loop"
  import CamtrainProto.*;
  parameter Integer N = 4;
  parameter Data.GearShaftData g[N](x={0.06*(i-1) for i in 1:N}, each y=0, gear(z={20, 40, 20, 40}, each m=0.5, each J=2e-4));
  parameter Data.MeshData md(jn=0.05e-3);
  inner Environment env(speed_rpm=6000);
  GearMeshBenchmark.Components.ConstantSpeed driver(Omega=env.Omega);
  GearMeshBenchmark.Components.TorsionSpringDamper spring(k=5e4, d=5);
  Assemblies.ElasticBody gears[N](data=g, ratio={(-1)^(i-1)*g[1].gear.z/g[i].gear.z for i in 1:N}, each k=1e9, each d=2000);
  GearMesh meshes[N-1](each data=md, a=g[1:N-1], b=g[2:N]);
  GearMeshBenchmark.Components.SineAngleTorque load(A=5, Omega=2*env.Omega);
equation
  connect(driver.frame, spring.frame_a);
  connect(spring.frame_b, gears[1].gearFrame);
  for i in 1:N-1 loop
    connect(meshes[i].frame_a, gears[i].gearFrame);
    connect(meshes[i].frame_b, gears[i+1].gearFrame);
  end for;
  connect(load.frame, gears[N].gearFrame);
  annotation (experiment(StopTime=0.02, Interval=2e-6, Tolerance=1e-8));
end ArrTest;
