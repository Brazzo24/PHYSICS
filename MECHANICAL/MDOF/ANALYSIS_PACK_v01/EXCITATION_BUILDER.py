import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

class ForcingBuilder:
    """
    A helper class that collects piecewise forcing segments (ramp, sinusoidal, etc.)
    for a single DOF. When 'build()' is called, it returns a function f(t)
    that evaluates the forcing at any time t.
    """
    def __init__(self):
        self.segments = []
    
    def add_ramp(self, t_start, t_end, F_start, F_end):
        """
        Add a ramp segment:
          from time t_start to t_end,
          force goes linearly from F_start to F_end.
        """
        self.segments.append({
            'type': 'ramp',
            't_start': t_start,
            't_end': t_end,
            'F_start': F_start,
            'F_end': F_end
        })
    
    def add_sinusoidal(self, t_start, t_end, amplitude, frequency, offset=0.0, phase=0.0):
        """
        Add a sinusoidal segment:
          from time t_start to t_end,
          force = offset + amplitude * sin(2*pi*frequency * t + phase)
        """
        self.segments.append({
            'type': 'sinusoidal',
            't_start': t_start,
            't_end': t_end,
            'amplitude': amplitude,
            'frequency': frequency,
            'offset': offset,
            'phase': phase
        })
    
    def add_constant(self, t_start, t_end, value):
        """
        Add a constant force segment:
          from time t_start to t_end,
          force = value
        """
        self.segments.append({
            'type': 'constant',
            't_start': t_start,
            't_end': t_end,
            'value': value
        })
    
    def build(self, default_value=0.0):
        """
        Return a function f(t) that, for a given time t, returns the force
        defined by whichever segment t falls into. If t is outside all segments,
        it returns default_value (e.g. 0.0).
        
        If time overlaps multiple segments, the first matching segment in
        self.segments is used. Adjust logic if you need a different rule.
        """
        def forcing_func(t):
            for seg in self.segments:
                if seg['t_start'] <= t < seg['t_end']:
                    if seg['type'] == 'ramp':
                        # Linear interpolation between F_start and F_end
                        duration = seg['t_end'] - seg['t_start']
                        frac = (t - seg['t_start']) / duration
                        return seg['F_start'] + frac * (seg['F_end'] - seg['F_start'])
                    
                    elif seg['type'] == 'sinusoidal':
                        amp = seg['amplitude']
                        freq = seg['frequency']
                        offset = seg['offset']
                        phase = seg['phase']
                        return offset + amp * np.sin(2*np.pi*freq * t + phase)
                    
                    elif seg['type'] == 'constant':
                        return seg['value']
            # If no segment matches, return default
            return default_value
        
        return forcing_func


def single_dof_system(t, y, m, c, k, forcing_func):
    """
    single_dof_system(t, y, m, c, k, forcing_func)
    y = [x, v]
      x: displacement
      v: velocity
    """
    x, v = y
    f_t = forcing_func(t)
    # Equation of motion: m*a + c*v + k*x = f(t)
    # => a = (f(t) - c*v - k*x) / m
    a = (f_t - c*v - k*x) / m
    return [v, a]

def main_piecewise_forcing_demo():
    # 1) Create the forcing builder and define segments
    fb = ForcingBuilder()
    
    # 0-2 s: ramp from 0 to 100 N
    fb.add_ramp(t_start=0.0, t_end=2.0, F_start=0.0, F_end=100.0)
    
    # 2-3 s: sinusoidal, amplitude=20 N, freq=5 Hz, offset=100 N
    fb.add_sinusoidal(t_start=2.0, t_end=3.0, amplitude=20.0, frequency=5.0, offset=100.0)
    
    # 3-5 s: ramp back down from 100 N to 0 N
    fb.add_ramp(t_start=3.0, t_end=5.0, F_start=100.0, F_end=0.0)
    
    # 5-10 s: constant zero force (could also just let default_value=0.0 handle it)
    fb.add_constant(t_start=5.0, t_end=10.0, value=0.0)
    
    # 2) Build the forcing function
    forcing_func = fb.build(default_value=0.0)
    
    # 3) Single-DOF system parameters
    m = 1.0   # mass
    c = 0.5   # damping
    k = 50.0  # stiffness
    
    # 4) Initial conditions
    x0 = 0.0
    v0 = 0.0
    y0 = [x0, v0]
    
    # 5) Time span for simulation
    t_span = (0, 10)  # 0 to 10 seconds
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    # 6) Solve
    sol = solve_ivp(
        fun=lambda t, y: single_dof_system(t, y, m, c, k, forcing_func),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval
    )
    
    # 7) Extract results
    x_sol = sol.y[0, :]
    v_sol = sol.y[1, :]
    
    # 8) Evaluate the forcing at each time for plotting
    f_sol = np.array([forcing_func(ti) for ti in sol.t])
    
    # 9) Plot the piecewise forcing
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, f_sol, 'r', label='Forcing [N]')
    plt.title('Piecewise Forcing Profile')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 10) Plot displacement and velocity
    plt.figure(figsize=(10, 4))
    plt.plot(sol.t, x_sol, label='Displacement x(t)')
    plt.plot(sol.t, v_sol, label='Velocity v(t)')
    plt.title('Single-DOF Response with Piecewise Forcing')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_piecewise_forcing_demo()
