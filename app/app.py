import tkinter as tk
from tkinter import PhotoImage, messagebox
import sympy as sp
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from methode_steepect_descent import *
from methode_newton import *
from methode_newton_modifie import *
from gradient_direction_conjugue import *

def optimize():
    try:
        func_str = func_entry.get().strip()
        vars_str = vars_entry.get().strip()
        x0_input = x0_entry.get().strip()
        x0 = [float(val.strip()) for val in x0_input.split(',') if val.strip()]
        tol = float(tol_entry.get().strip())
        method = method_var.get()

        vars = sp.symbols(vars_str)
        func = sp.sympify(func_str)
        grad = [sp.lambdify(vars, sp.diff(func, v)) for v in vars]
        hess = [[sp.lambdify(vars, sp.diff(sp.diff(func, v1), v2)) for v2 in vars] for v1 in vars]

        if method == "Steepest Descent":
            result, iterations, trajectory = steepest_descent(func, grad, x0, tol)
        elif method == "Newton":
            result, iterations, trajectory = newton_method(func, grad, hess, x0, tol)
        elif method == "Newton Modified":
            result, iterations, trajectory = newton_modified(func, grad, hess, x0, tol)
        elif method == "Conjugate Direction":
            result, iterations, trajectory = conjugate_direction(func, grad, hess, x0, tol)
        else:
            raise ValueError("Méthode non implémentée.")

        f_lambdified = sp.lambdify(vars, func)
        f_value = f_lambdified(*result)
        result_label.config(text=f" Le Point Optimal : {result}\nLa Fonction Minimale : {f_value}\nItérations : {iterations}")

        # Tracer le graphique
        plot_trajectory(func, vars, trajectory)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# Fonction pour tracer le graphique
def plot_trajectory(func, vars, trajectory):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_title("Trajectoire de l'optimisation")
    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])

    # Générer une grille de points
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    f_lambdified = sp.lambdify(vars, func)
    Z = f_lambdified(X, Y)
    ax.contour(X, Y, Z, levels=50, cmap="viridis")

    # Tracer la trajectoire
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Trajectoire')
    ax.legend()

    # Mettre à jour le graphique
    canvas.draw()

#################################################################
#################################################################


# Configuration principale
root = tk.Tk()
root.title("Optimisation Sans Contrainte")
root.geometry("900x600")
root.configure(bg="#cce56f")

# Configuration du cadre principal
frame_left = tk.Frame(root, bg="#75e053", width=500)
frame_left.grid(row=0, column=0, sticky="nsew")
frame_right = tk.Frame(root, bg="#a0f7ec", width=400)
frame_right.grid(row=0, column=1, sticky="nsew")

# Ajouter une icône
icon = PhotoImage(file="app\icons8-math-50.png")
root.iconphoto(True, icon)

# Configurer le côté gauche pour le graphique
fig = Figure(figsize=(5, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_left)
canvas.get_tk_widget().pack(fill="both", expand=True)

# Widgets pour la saisie à droite
tk.Label(frame_right, text="Fonction Objective:", bg="#f0f8ff", font=("Arial", 12)).pack(pady=10)
func_entry = tk.Entry(frame_right, font=("Arial", 12))
func_entry.pack(pady=10)

tk.Label(frame_right, text="Variables x, y:", bg="#f0f8ff", font=("Arial", 12)).pack(pady=10)
vars_entry = tk.Entry(frame_right, font=("Arial", 12))
vars_entry.pack(pady=10)

tk.Label(frame_right, text="Point Initial x0 (ex: 1, 1):", bg="#f0f8ff", font=("Arial", 12)).pack(pady=10)
x0_entry = tk.Entry(frame_right, font=("Arial", 12))
x0_entry.pack(pady=10)

tk.Label(frame_right, text="Tolérance:", bg="#f0f8ff", font=("Arial", 12)).pack(pady=10)
tol_entry = tk.Entry(frame_right, font=("Arial", 12))
tol_entry.pack(pady=10)

tk.Label(frame_right, text="Méthodes:", bg="#f0f8ff", font=("Arial", 12)).pack(pady=10)
method_var = tk.StringVar(value="Steepest Descent")
tk.OptionMenu(frame_right, method_var, "Steepest Descent", "Newton" ,"Newton Modified", "Conjugate Direction").pack(pady=10)

tk.Button(frame_right, text="Optimiser", command=optimize, bg="#4682b4", fg="white", font=("Arial", 12)).pack(pady=20)
result_label = tk.Label(frame_right, text="Résultats : ", bg="#f0f8ff", font=("Arial", 12, "italic"))
result_label.pack(pady=10)

# Configurer le redimensionnement
root.grid_columnconfigure(0, weight=3)
root.grid_columnconfigure(1, weight=2)
root.grid_rowconfigure(0, weight=1)

root.mainloop()
