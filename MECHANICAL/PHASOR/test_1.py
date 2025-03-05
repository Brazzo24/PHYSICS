import cmath

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
    
    def power_analysis(self):
        """Computes real, reactive, and apparent power."""
        I = self.current()
        S = self.V * I.conjugate()
        P = S.real  # Real Power (W)
        Q = S.imag  # Reactive Power (VAR)
        PF = P / abs(S)  # Power Factor
        return {"Apparent Power (VA)": abs(S), "Real Power (W)": P, "Reactive Power (VAR)": Q, "Power Factor": PF}

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
    print("Power Analysis:", circuit.power_analysis())
