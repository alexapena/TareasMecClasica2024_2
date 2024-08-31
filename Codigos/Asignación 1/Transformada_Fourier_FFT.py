import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal

# Parámetros físicos del sistema
g = 9.81  # m/s^2
l1 = 2.0  # longitud del péndulo 1
l2 = 1.0  # longitud del péndulo 2
m1 = 1.0  # masa del péndulo 1
m2 = 10.0  # masa del péndulo 2
k = 20.0   # constante del resorte

# Condiciones iniciales en grados: [theta1, theta1_dot, theta2, theta2_dot]
theta1_inicial = 10  # grados
theta2_inicial = 20  # grados

# Conversión a radianes
theta1_inicial = np.radians(theta1_inicial)
theta2_inicial = np.radians(theta2_inicial)

cond_iniciales = [theta1_inicial, 0, theta2_inicial, 0]

# Definición de las ecuaciones diferenciales
def ecuaciones(t, y):
    theta1, omega1, theta2, omega2 = y

    # Ecuaciones diferenciales dadas
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    domega1_dt = -(g * np.sin(theta1)) / l1 - (k * l2 *np.sin(theta1-theta2)) / (m1 * l1)
    domega2_dt = -(g * np.sin(theta2)) / l2 + (k * l1 *np.sin(theta1-theta2)) / (m2 * l2)

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Intervalo de tiempo para la simulación
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

# Resolución del sistema de ecuaciones diferenciales
sol = solve_ivp(ecuaciones, t_span, cond_iniciales, t_eval=t_eval, method='RK45')

# Obtención de las soluciones theta1 y theta2 en función del tiempo
theta1 = sol.y[0]
theta2 = sol.y[2]
t = sol.t

# --- Transformada Rápida de Fourier (FFT) ---

# Calcula la FFT
theta1_fft = np.fft.fft(theta1)
theta2_fft = np.fft.fft(theta2)

# Calcula las frecuencias correspondientes
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# Calcula el espectro de potencias (magnitud al cuadrado)
theta1_fft_power = np.abs(theta1_fft) ** 2
theta2_fft_power = np.abs(theta2_fft) ** 2

# Grafica el espectro de potencias para theta1
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(freqs[:len(freqs) // 2], theta1_fft_power[:len(freqs) // 2])
plt.title('Espectro de Potencias de Fourier para $\\theta(t)$')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia')
plt.grid(True)

# Grafica el espectro de potencias para theta2
plt.subplot(2, 1, 2)
plt.plot(freqs[:len(freqs) // 2], theta2_fft_power[:len(freqs) // 2], color='orange')
plt.title('Espectro de Potencias de Fourier para $\\alpha(t)$')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia')
plt.grid(True)

plt.tight_layout()
plt.show()

