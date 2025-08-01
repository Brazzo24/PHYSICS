from sympy import symbols, sin, cos
from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame, Point, RigidBody
from sympy.physics.mechanics import KanesMethod, inertia, dot

# Time variable
t = symbols('t')

# Generalized coordinates and speeds
x, theta = dynamicsymbols('x theta')       # Coordinates
xd, thetad = dynamicsymbols('x theta', 1)  # Time derivatives

# Parameters
m_w, m_b = symbols('m_w m_b')              # Wheel mass, body mass
l = symbols('l')                           # Distance from axle to COM of body
g = symbols('g')                           # Gravity

# Reference frames
N = ReferenceFrame('N')        # Inertial frame
B = N.orientnew('B', 'Axis', [theta, N.z])  # Body frame (rotates w.r.t. N)

# Define points
O = Point('O')                 # Ground contact point of wheel
O.set_vel(N, 0)

P = O.locatenew('P', x * N.x)  # Wheel axle
P.set_vel(N, xd * N.x)

G = P.locatenew('G', -l * B.y)  # Center of mass of body
G.v2pt_theory(P, N, B)          # Automatically computes velocity of G in N

# Inertia and masses
I_b = inertia(B, 0, 0, m_b * l**2)  # Body inertia around wheel axis
body = RigidBody('Body', G, B, m_b, (I_b, G))

# External forces
forces = [
    (P, 0),                        # no net force at axle (ideal constraint)
    (G, m_b * g * N.y),            # gravity on the body
    (B, 0)                         # no torque yet
]

# Create list of bodies
bodies = [body]

# Kinematic differential equations
KM = KanesMethod(N, q_ind=coordinates, u_ind=speeds)
(fr, frstar) = KM.kanes_equations(forces, bodies)
# Setup Kane’s Method
coordinates = [x, theta]
speeds = [xd, thetad]

KM = KanesMethod(N, q_ind=coordinates, u_ind=speeds, kd_eqs=kin_diff)
(fr, frstar) = KM.kanes_equations(forces, bodies)