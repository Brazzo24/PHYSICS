import cmath
import matplotlib.pyplot as plt

class MechanicalComponent:
    """
    Represents a mechanical component (Mass, Damper, Spring).
    """
    def __init__(self, component_type, value):
        self.type = component_type.lower()
        self.value = value
    
    def impedance(self, frequency):
        """Returns the mechanical impedance at a given frequency."""
        omega = 2 * cmath.pi * frequency
        if self.type == "mass":
            return complex(0, omega * self.value)  # jωm
        elif self.type == "damper":
            return complex(self.value, 0)  # c
        elif self.type == "spring":
            return complex(0, -1 / (omega * self.value))  # -j/(ωk)
        else:
            raise ValueError("Invalid component type")

class MechanicalSystem:
    """
    Represents a mechanical system with force input and components.
    """
    def __init__(self, force_magnitude, force_phase, frequency):
        self.F = cmath.rect(force_magnitude, cmath.pi * force_phase / 180)  # Convert to rectangular form
        self.frequency = frequency
        self.components = []
    
    def add_component(self, component):
        """Adds a component to the mechanical system."""
        self.components.append(component)
    
    def total_impedance(self):
        """Calculates total mechanical impedance."""
        return sum(comp.impedance(self.frequency) for comp in self.components)
    
    def velocity(self):
        """Calculates the velocity phasor in the system."""
        Z = self.total_impedance()
        return self.F / Z
    
    def complex_power(self):
        """Computes mechanical complex power (S = P + jQ)."""
        v = self.velocity()
        S = self.F * v.conjugate()
        return {"Complex Power (S)": S, "Real Power (P)": S.real, "Reactive Power (Q)": S.imag}
    
    def plot_phasor_diagram(self):
        """Plots the phasor diagram of force, velocity, and power components."""
        v = self.velocity()
        S = self.complex_power()["Complex Power (S)"]
        P = S.real  # Real Power
        Q = S.imag  # Reactive Power
        
        plt.figure(figsize=(6, 6))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        # Plot Force Phasor
        plt.arrow(0, 0, self.F.real, self.F.imag, head_width=5, head_length=5, color='r', label='Force (F)')
        
        # Plot Velocity Phasor
        plt.arrow(0, 0, v.real * 20, v.imag * 20, head_width=5, head_length=5, color='b', label='Velocity (v)')
        
        # Plot Complex Power Phasor
        plt.arrow(0, 0, P / 5, 0, head_width=5, head_length=5, color='g', label='Real Power (P)')
        plt.arrow(P / 5, 0, 0, Q / 5, head_width=5, head_length=5, color='m', label='Reactive Power (Q)')
        
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.title("Mechanical Phasor Diagram")
        plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis")
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Define the system parameters
    force_magnitude = 10  # N
    force_phase = 30  # degrees
    frequency = 1000  # Hz

    # Create Mechanical System Object
    system = MechanicalSystem(force_magnitude, force_phase, frequency)
    
    # Add components (Mass, Damper, Spring)
    system.add_component(MechanicalComponent("mass", 2))  # 2 kg
    system.add_component(MechanicalComponent("damper", 5))  # 5 Ns/m
    system.add_component(MechanicalComponent("spring", 500))  # 500 N/m
    
    # Calculate and display results
    print("Total Impedance:", system.total_impedance())
    print("Velocity Phasor:", system.velocity())
    print("Complex Power Analysis:", system.complex_power())
    
    # Plot phasor diagram
    system.plot_phasor_diagram()
