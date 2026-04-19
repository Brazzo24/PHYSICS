import sympy as sp
import sympy.physics.mechanics as me

me.init_vprinting()

# Time
t = sp.symbols('t')

# Generalized coordinates
x, y, theta = me.dynamicsymbols('x y theta')
xd, yd, thetad = me.dynamicsymbols('x y theta', 1)