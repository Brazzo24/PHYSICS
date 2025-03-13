MDOF_ANALYSIS_PACK_V01
    For mass-systems connected to GROUND via Spring/Damper


MDOF_ANALYSIS_PACK_V02
    Systems connected to GROUND via Spring/Damper
    includes reactive Power for the Inertias/Masses


MDOF_ANALYSIS_PACK_V02_1
    for TORSIONAL Systems without Spring/Damper connection to Ground.
    Uses lagrangian Multiplier instead, creating a torque-boundary condition, where the last 
    inertia in the chain is the Zero displacement reference node (augmented_system()).
    Is ought to represent the Road. Inertia N-1 would be the tyre, then.
    Inlcudes the same post-processing as V01 and V02.
    Includes are small Unit-test-package.