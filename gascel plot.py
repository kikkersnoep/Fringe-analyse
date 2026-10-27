import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constanten
druk_atm = 102300
golflengte = 532e-9
L = 0.0983

# Meetgegevens
M1 = np.array([24.5, 23.5, 24, 24.5, 23.5])
M2 = np.array([20.25, 20.5, 20, 20.5, 20.25])
M3 = np.array([15.15, 16.5, 15.7, 15.5, 16])
M4 = np.array([10.5, 10, 9.5, 9.7, 10])
M5 = np.array([4, 4, 3, 4, 4])

druk_M1 = np.array([74233, 75495, 76000, 74643, 76221])
druk_M2 = np.array([80230, 80861, 80640, 79977, 80198])
druk_M3 = np.array([84648, 84859, 85311, 85090, 84712])
druk_M4 = np.array([90266, 91024, 91718, 91213, 90519])
druk_M5 = np.array([97462, 97688, 97589, 97241, 97336])

franjes = np.concatenate([M1, M2, M3, M4, M5])
druk = np.concatenate([druk_M1, druk_M2, druk_M3, druk_M4, druk_M5])
delta_druk = np.abs(druk - druk_atm)

# Fitfunctie
def formule(delta_p, q):
    return (2 * L / golflengte) * q * delta_p / druk_atm

# Fit uitvoeren
parameters, covariantie = curve_fit(formule, delta_druk, franjes, p0=[0.0003])
q = parameters[0]
sigma_q = np.sqrt(covariantie[0, 0])

# Brekingsindex berekenen
n = 1 + q
sigma_n = sigma_q

print(f"q = {q:.8e} ± {sigma_q:.2e}")
print(f"n = {n:.8f} ± {sigma_n:.2e}")

# Fitlijn en onzekerheidsbanden
x_lijn = np.linspace(0, max(delta_druk) * 1.05, 200)
y_lijn = formule(x_lijn, q)
sigma_y = (2 * L / golflengte) * x_lijn / druk_atm * sigma_q

# Plot
plt.figure(figsize=(8, 5))
plt.fill_between(x_lijn, y_lijn - 3 * sigma_y, y_lijn + 3 * sigma_y, color="blue", alpha=0.15, label="3σ-band")
plt.fill_between(x_lijn, y_lijn - sigma_y, y_lijn + sigma_y, color="orange", alpha=0.30, label="1σ-band")
plt.scatter(delta_druk, franjes, color="black", marker="s", s=20, label="Meetgegevens")
plt.plot(x_lijn, y_lijn, color="red", label="Fit")
plt.plot([], [], " ", label=f"n = {n:.6f} ± {sigma_n:.2e}")

plt.xlabel("Drukverschil |Δp| (Pa)")
plt.ylabel("Aantal franjes N (-)")
plt.title("Brekingsindex van lucht")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()