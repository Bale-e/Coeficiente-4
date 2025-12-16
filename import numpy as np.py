import numpy as np
import matplotlib.pyplot as plt

print("=== MODELO DE CONSUMO ENERGÉTICO SIN sklearn ===\n")

# ================================
# 1. Ingreso de datos
# ================================

n = int(input("¿Cuántos datos vas a ingresar? "))

cpu_load = []
energy = []

print("\nIngresa los datos:\n")

for i in range(n):
    print(f"--- Dato {i+1} ---")
    carga = float(input("Carga CPU (%): "))
    consumo = float(input("Consumo energético (W): "))
    
    cpu_load.append(carga)
    energy.append(consumo)

cpu_load = np.array(cpu_load)
energy = np.array(energy)

# ================================
# 2. Ajuste polinómico manual
# ================================

# Matriz A: [x^2, x, 1]
A = np.vstack([cpu_load**2, cpu_load, np.ones(n)]).T

# Resolver mínimos cuadrados: theta = (A^T A)^(-1) A^T y
theta = np.linalg.lstsq(A, energy, rcond=None)[0]

a, b, c = theta  # coeficientes

print("\n=== MODELO AJUSTADO ===")
print(f"Función encontrada:\ny = {a:.4f}x² + {b:.4f}x + {c:.4f}")

# ================================
# 3. Graficar datos + modelo
# ================================

x_plot = np.linspace(0, 100, 300)
y_plot = a*x_plot**2 + b*x_plot + c

plt.scatter(cpu_load, energy, color='blue', label="Datos")
plt.plot(x_plot, y_plot, color='red', label="Modelo polinómico")
plt.xlabel("Carga CPU (%)")
plt.ylabel("Consumo (W)")
plt.title("Consumo energético según uso del CPU")
plt.grid(True)
plt.legend()
plt.show()

# ================================
# 4. Predicción
# ================================

nuevo = float(input("\nIngresa una carga para predecir consumo (0–100%): "))
pred = a*nuevo**2 + b*nuevo + c

print(f"\nConsumo estimado a {nuevo}% de carga: {pred:.2f} W\n")

