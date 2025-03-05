import sys
import networkx as nx
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QInputDialog, QMessageBox

class TorsionalSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Torsional System Builder")
        self.setGeometry(100, 100, 800, 600)

        # System storage
        self.graph = nx.Graph()
        self.node_counter = 1  # Unique ID for inertias

        # UI Layout
        self.layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.layout)

        # Buttons
        self.add_inertia_btn = QPushButton("➕ Add Inertia", self)
        self.add_spring_btn = QPushButton("🔗 Connect Inertias with Spring", self)
        self.add_damper_btn = QPushButton("💧 Connect Inertias with Damper", self)
        self.visualize_btn = QPushButton("🖼 Visualize System", self)
        self.export_btn = QPushButton("📤 Export to Python", self)

        self.layout.addWidget(self.add_inertia_btn)
        self.layout.addWidget(self.add_spring_btn)
        self.layout.addWidget(self.add_damper_btn)
        self.layout.addWidget(self.visualize_btn)
        self.layout.addWidget(self.export_btn)

        # Button Actions
        self.add_inertia_btn.clicked.connect(self.add_inertia)
        self.add_spring_btn.clicked.connect(self.add_spring)
        self.add_damper_btn.clicked.connect(self.add_damper)
        self.visualize_btn.clicked.connect(self.visualize_system)
        self.export_btn.clicked.connect(self.export_to_python)

    def add_inertia(self):
        """Adds an inertia (node) to the system"""
        inertia_value, ok = QInputDialog.getDouble(self, "Add Inertia", "Enter inertia value (kg·m²):", 1.0, 0.01, 1000.0, 2)
        if ok:
            self.graph.add_node(self.node_counter, inertia=inertia_value)
            QMessageBox.information(self, "Added", f"Inertia {self.node_counter} added (I = {inertia_value} kg·m²)")
            self.node_counter += 1

    def add_spring(self):
        """Adds a spring (edge) between two nodes"""
        if len(self.graph.nodes) < 2:
            QMessageBox.warning(self, "Error", "You need at least 2 inertias to add a spring.")
            return
        
        nodeA, ok1 = QInputDialog.getInt(self, "Connect Spring", "Enter first node number:", 1, 1, self.node_counter-1, 1)
        nodeB, ok2 = QInputDialog.getInt(self, "Connect Spring", "Enter second node number:", 1, 1, self.node_counter-1, 1)
        stiffness, ok3 = QInputDialog.getDouble(self, "Spring Stiffness", "Enter stiffness (Nm/rad):", 1000.0, 1.0, 100000.0, 2)

        if ok1 and ok2 and ok3 and nodeA != nodeB:
            self.graph.add_edge(nodeA, nodeB, stiffness=stiffness, type="spring")
            QMessageBox.information(self, "Added", f"Spring added between {nodeA} and {nodeB} (K = {stiffness} Nm/rad)")

    def add_damper(self):
        """Adds a damper (edge) between two nodes"""
        if len(self.graph.nodes) < 2:
            QMessageBox.warning(self, "Error", "You need at least 2 inertias to add a damper.")
            return

        nodeA, ok1 = QInputDialog.getInt(self, "Connect Damper", "Enter first node number:", 1, 1, self.node_counter-1, 1)
        nodeB, ok2 = QInputDialog.getInt(self, "Connect Damper", "Enter second node number:", 1, 1, self.node_counter-1, 1)
        damping, ok3 = QInputDialog.getDouble(self, "Damping Coefficient", "Enter damping (Nm·s/rad):", 0.1, 0.01, 10.0, 2)

        if ok1 and ok2 and ok3 and nodeA != nodeB:
            self.graph.add_edge(nodeA, nodeB, damping=damping, type="damper")
            QMessageBox.information(self, "Added", f"Damper added between {nodeA} and {nodeB} (C = {damping} Nm·s/rad)")

    def visualize_system(self):
        """Plots the torsional system"""
        plt.figure(figsize=(8, 4))
        pos = nx.spring_layout(self.graph)  # Auto-layout

        # Draw Nodes (Inertias)
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", node_size=2000, edge_color="black", font_size=10)

        # Annotate springs and dampers
        edge_labels = {edge: f"{data['stiffness']} Nm/rad" for edge, data in self.graph.edges.items() if "stiffness" in data}
        edge_labels.update({edge: f"{data['damping']} Nm·s/rad" for edge, data in self.graph.edges.items() if "damping" in data})
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color="red")

        plt.title("Torsional System Visualization")
        plt.show()

    def export_to_python(self):
        """Generates a Python script based on the designed system"""
        code = "from system import TorsionalSystem\n\n"
        code += "system = TorsionalSystem()\n\n"

        for node, data in self.graph.nodes(data=True):
            code += f"n{node} = system.add_inertia({data['inertia']})\n"

        for edge in self.graph.edges(data=True):
            nodeA, nodeB, properties = edge
            if "stiffness" in properties:
                code += f"system.add_spring(n{nodeA}, n{nodeB}, {properties['stiffness']})\n"

        with open("generated_system.py", "w") as file:
            file.write(code)

        QMessageBox.information(self, "Exported", "System exported to generated_system.py")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TorsionalSystemGUI()
    window.show()
    sys.exit(app.exec())