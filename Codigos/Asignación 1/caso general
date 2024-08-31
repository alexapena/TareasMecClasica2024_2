#caso general

import numpy as np
import matplotlib.pyplot as plt
from random import seed
from random import random

# Parametros
g = 9.81  # Aceleracion debida a la gravedad (m/s^2)
l1 = 1.0  # Longitud del pendulo 1 (m)
l2 = 1.0  # Longitud del pendulo 2 (m)
k = 20  # Constante del resorte (N/m)
m1 = 10.0  # Masa 1 (kg)
m2 = 40.0  # Masa 2 (kg)

# Definimos el sistema de ecuaciones diferenciales
def sistema_ecuaciones(t, y):
    theta, omega1, alpha, omega2 = y

    # Ecuaciones del sistema
    dtheta_dt = omega1
    domega1_dt = -(g / l1) * np.sin(theta) - (k * l2 / (m1 * l1)) * np.sin(theta - alpha)
    dalpha_dt = omega2
    domega2_dt = -(g / l2) * np.sin(alpha) + (k * l1 / (m2 * l2)) * np.sin(theta - alpha)

    return np.array([dtheta_dt, domega1_dt, dalpha_dt, domega2_dt])

# Implementacion del metodo de Runge-Kutta de cuarto orden
def runge_kutta_4(sistema_ecuaciones, y0, t0, tf, h):
    t = np.arange(t0, tf, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        k1 = h * sistema_ecuaciones(t[i-1], y[i-1])
        k2 = h * sistema_ecuaciones(t[i-1] + 0.5*h, y[i-1] + 0.5*k1)
        k3 = h * sistema_ecuaciones(t[i-1] + 0.5*h, y[i-1] + 0.5*k2)
        k4 = h * sistema_ecuaciones(t[i-1] + h, y[i-1] + k3)

        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y

# Condiciones iniciales
theta0 = np.pi/3 # angulo inicial de theta (radianes)
omega1_0 = 0.0  # velocidad angular inicial de theta (rad/s)
alpha0 = np.pi/5 # angulo inicial de alpha (radianes)
omega2_0 = 0.3  # velocidad angular inicial de alpha (rad/s)
y0 = [theta0, omega1_0, alpha0, omega2_0]

t0 = 0
tf = 40
h = 0.001

# Solucionamos el sistema
t, y = runge_kutta_4(sistema_ecuaciones, y0, t0, tf, h)

# Graficamos los resultados
plt.plot(t, y[:, 0], label='θ(t)')
plt.plot(t, y[:, 2], label='α(t)')
plt.xlabel('Tiempo t')
plt.ylabel('Angulos θ, α (rad)')
plt.legend()
plt.show()
