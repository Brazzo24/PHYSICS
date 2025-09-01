import numpy as np
from torsional_filter_design import TorsionalSystem, two_mass_target_k, two_mass_c_for_zeta

# Example: two-mass chain (engine ↔ soft coupling ↔ gearbox)
I1, I2 = 0.163, 0.181      # kg·m^2
k = two_mass_target_k(I1, I2, f_n_target_hz=12.0)  # place mode at ~12 Hz
c = two_mass_c_for_zeta(I1, I2, k, zeta=0.15)

sys = TorsionalSystem([I1, I2])
e_soft = sys.add_kv(0, 1, k=k, c=c, label="soft coupling")
sys.transmission_element = e_soft
sys.input_node = 0

rpm = 3000.0
orders = np.arange(0.5, 21.0, 0.5)
ords, mags = sys.order_sweep(rpm, orders)

for o, m in zip(ords, mags):
    print(f"order {o:>4.1f}: |T_t/T_in| = {m:8.5f}")
