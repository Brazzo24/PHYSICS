import numpy as np

def rotation_x(A, alpha):
    """
    Rotates a given vektor A around X-Axis by given angle alpha in degrees.
    
    """
   
    #transform angle from degrees to radians
    alpha = np.radians(alpha)

    c,s = np.cos(alpha), np.sin(alpha)

    #rotation matrix
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s , c]])

    # vector A after rotation around axis X by alpha degrees
    A_rot =Rx @ A  
    
    return A_rot

def rotation_y(A, alpha):
    """
    Rotates a given vektor A around X-Axis by given angle alpha in degrees.
    
    """
   
    #transform angle from degrees to radians
    alpha = np.radians(alpha)

    c,s = np.cos(alpha), np.sin(alpha)

    #rotation matrix
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0 , c]])

    # vector A after rotation around axis X by alpha degrees
    A_rot =Ry @ A  
    
    return A_rot

def rotation_z(A, alpha):
    """
    Rotates a given vektor A around X-Axis by given angle alpha in degrees.
    
    """
   
    #transform angle from degrees to radians
    alpha = np.radians(alpha)

    c,s = np.cos(alpha), np.sin(alpha)

    #rotation matrix
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0 , 1]])

    # vector A after rotation around axis X by alpha degrees
    A_rot = Rz @ A  
    
    return A_rot