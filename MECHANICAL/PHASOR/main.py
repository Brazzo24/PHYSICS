import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter: Spannung, Strom und Startwinkel
U = 230  # Spannung (V)
I = 10   # Strom (A)
phi_max = np.radians(90)  # Maximaler Phasenwinkel (90°)
frames = 100  # Anzahl der Animationsschritte

# Erstellen der Grafik
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2 * U, 1.2 * U)
ax.set_ylim(-1.2 * U, 1.2 * U)
ax.set_xlabel("Realteil")
ax.set_ylabel("Imaginärteil")
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_title("Animiertes Zeigerdiagramm")

# Pfeile für Spannung, Strom, P und Q
spg_arrow = ax.quiver(0, 0, U, 0, angles='xy', scale_units='xy', scale=1, color='blue', label="Spannung U")
strom_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='red', label="Strom I")
p_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='green', label="Wirkleistung P")
q_arrow = ax.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='purple', label="Blindleistung Q")

# Textfelder für P und Q
p_text = ax.text(10, -20, "", fontsize=12, color="green")
q_text = ax.text(10, 20, "", fontsize=12, color="purple")

# Animationsfunktion
def update(frame):
    phi = np.radians(frame * (90 / frames))  # Phasenwinkel von 0° bis 90°
    
    # Stromkomponenten berechnen
    I_x, I_y = I * np.cos(phi), I * np.sin(phi)
    
    # Wirkleistung P und Blindleistung Q berechnen
    P = U * I * np.cos(phi)
    Q = U * I * np.sin(phi)
    
    # Zeiger aktualisieren
    strom_arrow.set_UVC(I_x, I_y)
    p_arrow.set_UVC(P, 0)
    q_arrow.set_offsets((P, 0))
    q_arrow.set_UVC(0, Q)

    # Text aktualisieren
    p_text.set_text(f"P = {P:.1f} W")
    p_text.set_position((P / 2, -20))
    
    q_text.set_text(f"Q = {Q:.1f} var")
    q_text.set_position((P + 10, Q / 2))
    
    return strom_arrow, p_arrow, q_arrow, p_text, q_text

# Animation starten
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

# Legende & Diagramm anzeigen

ax.legend()
plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)
plt.grid()
plt.show()