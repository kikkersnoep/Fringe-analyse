
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. PARAMETERS EN CONSTANTEN
# ==========================================
# Vul hier je teamnummer in (bijv. "01" of "02")
TEAM_NUMMER = "6"  

# Bekende vaste parameters (pas deze aan naar jouw opstelling!)
LAMBDA_M = 532e-9  # Golflengte van de laser in meters 
DIKTE_M = 3.0e-2     # Dikte van de brekende laag d in meters 

# Bestandsnaam van de meetdata 
DATA_FILE = f"team_6_data.csv"

# ==========================================
# 2. INSTELLEN VAN HET THEORETISCH MODEL
# ==========================================
def franjes_model(theta_deg, n):
    
    theta = np.radians(theta_deg)
    N = (2 * DIKTE_M / LAMBDA_M) * (np.sqrt(n**2 - np.sin(theta)**2) - np.cos(theta) + (1 - n))
    
    return N

# ==========================================
# 3. DATA INLEZEN EN FITTEN
# ==========================================
def main():
    # Lees de meetdata in uit de CSV-file
    # Verwacht kolommen: hoek_graden, aantal_franjes, onzekerheid_N
    data = np.genfromtxt(DATA_FILE, delimiter=',', names=True, skip_header=0)
    
    theta_data = data['hoek_graden']
    N_data = data['aantal_franjes']
    sigma_N = data['onzekerheid_N']

    # Fitten van de parameter n met scipy.optimize.curve_fit
    # p0 is de initiële gok voor de brekingsindex n 
    # sigma geeft de foutbalken op N mee zodat meetpunten gewogen worden
    popt, pcov = curve_fit(
        franjes_model, 
        theta_data, 
        N_data, 
        p0=[1.0], 
        bounds=(1.0, 2.5),
        sigma=sigma_N, 
        absolute_sigma=True
    )

    # Optillen van het fitresultaat en de onzekerheid (standaardafwijking)
    n_fit = popt[0]
    n_err = np.sqrt(pcov[0][0])  # De wortel uit de variantie is de standaardfout op n

    # Print de resultaten in de terminal
    print(f"--- Fit Resultaten Team {TEAM_NUMMER} ---")
    print(f"Gefitte brekingsindex n = {n_fit:.5f} +/- {n_err:.5f}")

    # ==========================================
    # 4. GRAFIEK MAKEN EN OPSLAAN
    # ==========================================
    plt.figure(figsize=(8, 6))

    # Plot de meetpunten met foutbalken
    plt.errorbar(
        theta_data, 
        N_data, 
        yerr=sigma_N, 
        fmt='o', 
        color='navy', 
        ecolor='crimson', 
        capsize=3, 
        label='Meetdata'
    )

    # Genereer een gladde lijn voor de gefitte curve
    theta_fit = np.linspace(min(theta_data), max(theta_data), 500)
    N_fit = franjes_model(theta_fit, n_fit)

    # Plot de gefitte curve
    plt.plot(
        theta_fit, 
        N_fit, 
        color='darkorange', 
        linewidth=2, 
        label=f'Fit: n = {n_fit:.4f} ± {n_err:.4f}'
    )

    # Opmaak van de grafiek
    plt.xlabel('Invalshoek θ (°)')
    plt.ylabel('Aantal franjes N (-)')
    plt.title(f'Brekingsindex Fit - Team {TEAM_NUMMER}')
    plt.grid(True, linestyle='--')
    plt.legend()

    # Sla de grafiek op
    output_plot_name = f"team_{TEAM_NUMMER}_fit_brekingsindex_plot.png"
    plt.savefig(output_plot_name, dpi=300, bbox_inches='tight')
    print(f"Grafiek succesvol opgeslagen als '{output_plot_name}'")
    
    plt.show()

if __name__ == "__main__":
    main()