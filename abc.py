import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================================
# AIRFOIL GENERATION
# ==========================================================

def get_airfoil(code, n=120):

    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    x = np.linspace(0.001, 1, n)   # avoid x=0 (division issue)

    yt = 5*t*(0.2969*np.sqrt(x) - 0.126*x - 0.3516*x**2 +
              0.2843*x**3 - 0.1015*x**4)

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i in range(n):
        if p != 0:
            if x[i] < p:
                yc[i] = (m/p**2)*(2*p*x[i] - x[i]**2)
                dyc_dx[i] = (2*m/p**2)*(p - x[i])
            else:
                yc[i] = (m/(1-p)**2)*((1-2*p)+2*p*x[i]-x[i]**2)
                dyc_dx[i] = (2*m/(1-p)**2)*(p - x[i])

    theta = np.arctan(dyc_dx)

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    x_full = np.concatenate([xu, xl[::-1]])
    y_full = np.concatenate([yu, yl[::-1]])

    return xu, yu, xl, yl, x, x_full, y_full


# ==========================================================
# GUI SETUP
# ==========================================================

root = Tk()
root.title("Aerodynamic Simulator (Improved Cp)")
root.geometry("1100x700")

control = Frame(root)
control.pack(pady=10)

Label(control, text="Angle of Attack (deg)").grid(row=0, column=0)

alpha_slider = Scale(control, from_=-10, to=15,
                     orient=HORIZONTAL, length=300)
alpha_slider.grid(row=0, column=1)

Label(control, text="Airfoil").grid(row=0, column=2)

airfoil_var = StringVar(value="2412")
OptionMenu(control, airfoil_var, "0012", "2412", "4412").grid(row=0, column=3)

cl_label = Label(control, text="Cl = 0.0", font=("Arial", 12))
cl_label.grid(row=0, column=4, padx=20)

# Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Colorbar (create once)
scatter = ax1.scatter([], [], c=[], cmap='jet')
cbar = fig.colorbar(scatter, ax=ax1)
cbar.set_label("Cp")


# ==========================================================
# UPDATE FUNCTION
# ==========================================================

def update(val):

    alpha_deg = alpha_slider.get()
    alpha = np.radians(alpha_deg)

    xu, yu, xl, yl, x, x_full, y_full = get_airfoil(airfoil_var.get())

    ax1.clear()
    ax2.clear()

    # ======================================================
    # FLOW FIELD
    # ======================================================

    X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 40),
                       np.linspace(-0.5, 0.5, 40))

    U = np.cos(alpha)
    V = np.sin(alpha)

    ax1.streamplot(X, Y,
                   U*np.ones_like(X),
                   V*np.ones_like(Y),
                   density=1.2)

    # ======================================================
    # THIN AIRFOIL THEORY (Lift)
    # ======================================================

    Cl = 2*np.pi*alpha

    # ======================================================
    # IMPROVED Cp MODEL
    # ======================================================

    Cp = -4 * alpha * np.sqrt((1 - x) / x)

    # Limit extreme values (for visualization stability)
    Cp = np.clip(Cp, -3, 1)

    # Map Cp to airfoil surface
    xc = np.concatenate([x, x[::-1]])
    yc = np.concatenate([yu, yl[::-1]])
    Cp_full = np.concatenate([Cp, Cp[::-1]])

    # Plot Cp color
    sc = ax1.scatter(xc, yc, c=Cp_full, cmap='jet', s=15)
    cbar.update_normal(sc)

    # ======================================================
    # AIRFOIL + VECTORS
    # ======================================================

    ax1.plot(xu, yu, 'k')
    ax1.plot(xl, yl, 'k')

    ax1.arrow(-0.4, 0,
              0.3*np.cos(alpha),
              0.3*np.sin(alpha),
              color='red', head_width=0.03)

    ax1.arrow(0.3, 0,
              -Cl*np.sin(alpha)*0.2,
               Cl*np.cos(alpha)*0.2,
              color='green', head_width=0.03)

    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title(f"Flow Field (NACA {airfoil_var.get()})")
    ax1.axis("equal")

    # ======================================================
    # Cp GRAPH
    # ======================================================

    ax2.plot(x, Cp)
    ax2.invert_yaxis()
    ax2.set_title("Cp Distribution (Improved)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Cp")
    ax2.grid()

    cl_label.config(text=f"Cl = {Cl:.3f}")

    canvas.draw_idle()


# Bind
alpha_slider.config(command=update)
airfoil_var.trace("w", lambda *args: update(0))

update(0)

root.mainloop()