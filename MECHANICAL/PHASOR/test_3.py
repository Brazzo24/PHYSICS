import cmath
import matplotlib.pyplot as plt

class Component:
    """
    Represents a circuit component (Resistor, Inductor, Capacitor).
    """
    def __init__(self, component_type, value):
        self.type = component_type.lower()
        self.value = value
    
    def impedance(self, frequency):
        """Returns the impedance of the component at a given frequency."""
        omega = 2 * cmath.pi * frequency
        if self.type == "resistor":
            return complex(self.value, 0)
        elif self.type == "inductor":
            return complex(0, omega * self.value)
        elif self.type == "capacitor":
            return complex(0, -1 / (omega * self.value))
        else:
            raise ValueError("Invalid component type")

class Circuit:
    """
    Represents an AC circuit with voltage source and components.
    """
    def __init__(self, voltage_magnitude, voltage_phase, frequency):
        self.V = cmath.rect(voltage_magnitude, cmath.pi * voltage_phase / 180)  # Convert polar to rectangular
        self.frequency = frequency
        self.components = []
    
    def add_component(self, component):
        """Adds a component to the circuit."""
        self.components.append(component)
    
    def total_impedance(self):
        """Calculates total impedance of the circuit."""
        return sum(comp.impedance(self.frequency) for comp in self.components)
    
    def current(self):
        """Calculates the current phasor in the circuit."""
        Z = self.total_impedance()
        return self.V / Z
    
    def complex_power(self):
        """Computes complex power (S = P + jQ)."""
        I = self.current()
        S = self.V * I.conjugate()
        return {"Complex Power (S)": S, "Real Power (P)": S.real, "Reactive Power (Q)": S.imag}
    
    def plot_phasor_diagram(self):
        """Plots the phasor diagram of voltage, current, and power components."""
        I = self.current()
        S = self.complex_power()["Complex Power (S)"]
        P = S.real  # Real Power
        Q = S.imag  # Reactive Power
        
        plt.figure(figsize=(6, 6))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        # Plot Voltage Phasor
        plt.arrow(0, 0, self.V.real, self.V.imag, head_width=5, head_length=5, color='r', label='Voltage (V)')
        
        # Plot Current Phasor
        plt.arrow(0, 0, I.real * 20, I.imag * 20, head_width=5, head_length=5, color='b', label='Current (I)')
        
        # Plot Complex Power Phasor
        plt.arrow(0, 0, P / 5, 0, head_width=5, head_length=5, color='g', label='Real Power (P)')
        plt.arrow(P / 5, 0, 0, Q / 5, head_width=5, head_length=5, color='m', label='Reactive Power (Q)')
        
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.title("Phasor Diagram with Power Components")
        plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis")
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Define the circuit parameters
    voltage_magnitude = 100  # V
    voltage_phase = 30  # degrees
    frequency = 1000  # Hz

    # Create Circuit Object
    circuit = Circuit(voltage_magnitude, voltage_phase, frequency)
    
    # Add components (Resistor and Inductor)
    circuit.add_component(Component("resistor", 10))  # 10 Ohm
    circuit.add_component(Component("inductor", 20e-3))  # 20 mH
    
    # Calculate and display results
    print("Total Impedance:", circuit.total_impedance())
    print("Current Phasor:", circuit.current())
    print("Complex Power Analysis:", circuit.complex_power())
    
    # Plot phasor diagram
    circuit.plot_phasor_diagram()
