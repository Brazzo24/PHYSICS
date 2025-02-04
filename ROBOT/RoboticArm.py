import numpy as np
import matplotlib.pyplot as plt

class RoboticArm:
    def __init__(self, link1_length, link2_length):
        self.link1_length = link1_length
        self.link2_length = link2_length

    def forward_kinematics(self, theta1, theta2):
        """
        Compute the (x, y) position of the end effector based on joint angles.
        :param theta1: Angle of the revolute joint to ground (in radians).
        :param theta2: Angle of the elbow joint (in radians).
        :return: Coordinates of the joints and end effector.
        """
        # First joint (base)
        x0, y0 = 0, 0

        # Second joint
        x1 = self.link1_length * np.cos(theta1)
        y1 = self.link1_length * np.sin(theta1)

        # End effector
        x2 = x1 + self.link2_length * np.cos(theta1 + theta2)
        y2 = y1 + self.link2_length * np.sin(theta1 + theta2)

        return [(x0, y0), (x1, y1), (x2, y2)]

    def visualize(self, angles, interval=0.5):
        """
        Visualize the robotic arm's motion.
        :param angles: List of (theta1, theta2) tuples.
        :param interval: Time interval between frames.
        """
        plt.figure()
        plt.axis('equal')
        plt.grid(True)
        plt.title("Robotic Arm Simulation")

        for theta1, theta2 in angles:
            # Compute positions
            joints = self.forward_kinematics(theta1, theta2)

            # Clear the plot
            plt.clf()
            plt.grid(True)
            plt.axis([-2, 2, -2, 2])  # Set limits based on arm length
            plt.title("Robotic Arm Simulation")

            # Plot the links
            plt.plot([joints[0][0], joints[1][0]], [joints[0][1], joints[1][1]], 'o-', linewidth=4)
            plt.plot([joints[1][0], joints[2][0]], [joints[1][1], joints[2][1]], 'o-', linewidth=4)

            # Show the joints
            plt.scatter([j[0] for j in joints], [j[1] for j in joints], color='red', zorder=5)

            # Pause for visualization
            plt.pause(interval)

        plt.show()


# Example usage with mock position data
link1_length = 1.0
link2_length = 1.0
robotic_arm = RoboticArm(link1_length, link2_length)

# Mock position data: list of (theta1, theta2) in radians
mock_angles = [
    (0, 0),
    (np.pi / 6, np.pi / 6),
    (np.pi / 4, np.pi / 4),
    (np.pi / 3, np.pi / 3),
    (np.pi / 2, np.pi / 2),
    (np.pi / 3, np.pi / 4),
    (np.pi / 6, np.pi / 3),
    (0, 0),
]

# Visualize the simulation
robotic_arm.visualize(mock_angles)