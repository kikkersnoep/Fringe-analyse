"""
Live brekingsindex bepalen tijdens het draaien
================================================

Gebaseerd op jouw eigen fitcode (Fresnel-formule + curve_fit), maar dan
in een loop: na elk nieuw meetpunt (hoek, aantal franjes) wordt de fit
opnieuw gedaan met ALLE data tot dan toe. Zo zie je n en de onzekerheid
live bijwerken, en wordt de onzekerheid kleiner naarmate je meer punten
(dus meer gedraaid) hebt.

Er zijn 4 manieren om aan nieuwe meetpunten te komen (kies via BRON
hieronder) — je hoeft alleen de manier te gebruiken die bij jouw opstelling
past:

  "TIJD"        Geen hoeksensor nodig: je draait zelf op de klok mee, bijv.
                elke TIJD_PER_STAP_S seconden GRADEN_PER_STAP graden verder,
                tot EIND_HOEK_GRADEN. Het script telt zelf continu de
                franjes op de oscilloscoop en neemt elke stap automatisch
                een nieuw punt (verwachte_hoek, cumulatieve N). Werkt alleen
                goed als je echt binnen elk tijdsblok precies die hoek draait.

  "CSV_LIVE"    Het script houdt hetzelfde CSV-bestand in de gaten
                (hoek, N, onzekerheid) en pikt nieuwe regels op zodra ze
                worden toegevoegd tijdens de meting. Handig als jullie
                nu al per stap een regel wegschrijven.

  "HANDMATIG"   Na elke draaistap typ je hoek + N direct in de terminal.
                Geen extra hardware nodig, werkt meteen.

  "AUTOMATISCH" Sjabloon om te koppelen aan een hoeksensor of motor,
                zodat draaien + tellen + fitten helemaal vanzelf gaat.
                Moet je op de plekken met "TODO" aanpassen aan jouw opstelling.

Benodigde packages:
    pip install numpy matplotlib scipy
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =====================================================================
# INSTELLINGEN
# =====================================================================

BRON = "TIJD"   # "TIJD", "CSV_LIVE", "HANDMATIG" of "AUTOMATISCH"

# --- Constantes van de opstelling (zelfde als in jouw code) ---
d = 0.03
lam = 532e-9

# --- TIJD instellingen (BRON = "TIJD") ---
GRADEN_PER_STAP = 2.0      # hoeveel graden je per stap draait
TIJD_PER_STAP_S = 5.0      # hoeveel seconden je daarvoor de tijd hebt
EIND_HOEK_GRADEN = 24.0    # tot welke totale hoek doorgaan
TIJD_KANAAL = 1            # oscilloscoopkanaal met het interferentiesignaal
TIJD_ONZEKERHEID_PER_PUNT = 1.0   # geschatte onzekerheid op N per stap (in franjes)

# --- CSV_LIVE instellingen ---
CSV_PAD = "team_6_fit_brekingsindex/team_6_data.csv"
POLL_INTERVAL_S = 1.0   # hoe vaak (in s) het bestand gecontroleerd wordt op nieuwe regels

# Als er al oude data in CSV_PAD staat: True = die meteen meenemen bij start
# (je ziet dan gelijk een n op basis van de bestaande data). False = alleen
# reageren op regels die NA het starten van dit script worden toegevoegd.
NEEM_BESTAANDE_CSV_DATA_MEE = False

# --- Fit instellingen ---
MIN_PUNTEN_VOOR_FIT = 2      # curve_fit heeft minstens 2 punten nodig voor een zinnige onzekerheid
START_SCHATTING_N = 1.5      # p0 voor curve_fit

# --- Output ---
OUT_MAP = "team_6_fit_brekingsindex/out"
OUT_PLOT = "team_6_fit_brekingsindex_live_plot.png"
OUT_CSV = "team_6_data_live.csv"


# =====================================================================
# De fitformule (ongewijzigd overgenomen uit jouw code)
# =====================================================================

def brekingshoek_formule(i, n):
    i = i / 180 * np.pi
    N = 2 * d / lam * (np.sqrt(n**2 - np.sin(i)**2) - np.cos(i) + (1 - n))
    return N


def fit_n(hoek_graden, aantal_fransjes, onzekerheid_N):
    """Doet de curve_fit met alle data tot nu toe. Geeft (n, onzekerheid_n)
    terug, of (None, None) als de fit (nog) niet lukt."""
    try:
        popt, pcov = curve_fit(
            brekingshoek_formule, hoek_graden, aantal_fransjes,
            p0=[START_SCHATTING_N], sigma=onzekerheid_N, absolute_sigma=True,
        )
        n = popt[0]
        onzekerheid_n = np.sqrt(pcov[0, 0])
        return n, onzekerheid_n
    except RuntimeError:
        return None, None


# =====================================================================
# Databronnen — kies er 1 via BRON
# =====================================================================

def nieuwe_punten_csv_live(reeds_gelezen):
    """Leest het CSV-bestand opnieuw in en geeft alleen de regels terug die
    nog niet eerder gezien zijn (op basis van het aantal reeds gelezen
    regels)."""
    if not os.path.exists(CSV_PAD):
        return [], reeds_gelezen

    data = np.loadtxt(CSV_PAD, delimiter=",", skiprows=1, ndmin=2)
    if len(data) <= reeds_gelezen:
        return [], reeds_gelezen

    nieuwe_rijen = data[reeds_gelezen:]
    punten = [(rij[0], rij[1], rij[2]) for rij in nieuwe_rijen]
    return punten, len(data)


def nieuw_punt_handmatig():
    """Vraagt interactief om een nieuw meetpunt. Typ 'stop' bij de hoek
    om te stoppen met meten."""
    invoer = input("\nHoek in graden (of 'stop' om te stoppen): ").strip()
    if invoer.lower() == "stop":
        return None
    hoek = float(invoer)
    N = float(input("Aantal franjes tot nu toe: "))
    onzeker = input("Onzekerheid op N (Enter = 0.5): ").strip()
    onzeker = float(onzeker) if onzeker else 0.5
    return (hoek, N, onzeker)


# --- TIJD: draaien op de klok, N continu automatisch tellen via de scoop ---

def _meet_een_stap(scope, duur_s):
    """Telt gedurende 'duur_s' seconden continu de franjes op de
    oscilloscoop (met een countdown in de terminal) en geeft het aantal
    in dat tijdsblok gevonden franjes terug."""
    from team_6_fit_brekingsindex.franjes_tellen import acquire_waveform, detect_peaks

    MIN_PIEK_AFSTAND_S = 0.005
    PIEK_PROMINENTIE_V = 0.02

    tijd_offset_totaal = 0.0
    aantal_deze_stap = 0
    start = time.time()
    laatste_getoonde_seconde = None

    while (time.time() - start) < duur_s:
        resterend = duur_s - (time.time() - start)
        seconde = int(np.ceil(resterend))
        if seconde != laatste_getoonde_seconde:
            print(f"  nog {seconde}s...", end="\r", flush=True)
            laatste_getoonde_seconde = seconde

        try:
            t, v = acquire_waveform(scope, channel=TIJD_KANAAL, mode="NORM")
        except RuntimeError:
            continue

        t_verschoven = t - t[0] + tijd_offset_totaal
        piek_idx = detect_peaks(t_verschoven, v, MIN_PIEK_AFSTAND_S,
                                 PIEK_PROMINENTIE_V, "max")
        aantal_deze_stap += len(piek_idx)
        if len(t_verschoven) > 1:
            tijd_offset_totaal = t_verschoven[-1] + (t_verschoven[1] - t_verschoven[0])

    print(" " * 20, end="\r")  # countdown-regel wissen
    return aantal_deze_stap


def tijd_gebaseerde_meetreeks():
    """Generator: jij draait zelf mee op de klok (elke TIJD_PER_STAP_S
    seconden GRADEN_PER_STAP graden verder), het script telt ondertussen
    continu de franjes op de oscilloscoop en levert na elke stap een nieuw
    (verwachte_hoek, cumulatieve_N, onzekerheid) punt."""
    try:
        from franjes_tellen import connect_scope
    except ImportError as e:
        raise RuntimeError(
            "Kon franjes_tellen.py niet importeren — zorg dat het in "
            "dezelfde map staat als dit script."
        ) from e

    scope = connect_scope()
    cumulatief_N = 0
    eind_stappen = int(round(EIND_HOEK_GRADEN / GRADEN_PER_STAP))

    print(f"\nStart over 3 seconden — {eind_stappen} stappen van "
          f"{GRADEN_PER_STAP:.1f} graden in {TIJD_PER_STAP_S:.0f}s per stap, "
          f"tot {EIND_HOEK_GRADEN:.1f} graden totaal.\n")
    time.sleep(3)

    try:
        for stap in range(1, eind_stappen + 1):
            verwachte_hoek = stap * GRADEN_PER_STAP
            print(f">>> NU DRAAIEN naar {verwachte_hoek:.1f} graden <<<")

            nieuwe_pieken = _meet_een_stap(scope, TIJD_PER_STAP_S)
            cumulatief_N += nieuwe_pieken
            print("\a", end="", flush=True)  # piep als teken dat de stap klaar is
            print(f"[stap {stap}/{eind_stappen}] hoek = {verwachte_hoek:.1f} graden, "
                  f"+{nieuwe_pieken} franjes deze stap, totaal N = {cumulatief_N}")

            yield (verwachte_hoek, float(cumulatief_N), TIJD_ONZEKERHEID_PER_PUNT)
    finally:
        scope.close()


# --- AUTOMATISCH: sjabloon om te koppelen aan de oscilloscoop + motor ---
# Dit blok probeert de functies uit franjes_tellen.py en
# brekingsindex_berekenen.py te hergebruiken (moeten in dezelfde map staan).
# Pas de TODO's aan naar jouw eigen opstelling: stapgrootte, eindhoek,
# kanaal, piekinstellingen, etc.

def automatische_meetreeks():
    """Generator die tijdens het draaien telkens een nieuw (hoek, N,
    onzekerheid) punt oplevert, door de motor aan te sturen en na elke
    stap het aantal nieuwe franjes op de oscilloscoop te tellen."""
    try:
        from team_6_fit_brekingsindex.franjes_tellen import connect_scope, acquire_waveform, detect_peaks
        from brekingsindex_berekenen import draai_en_lees_hoek
    except ImportError as e:
        raise RuntimeError(
            "Kon franjes_tellen.py / brekingsindex_berekenen.py niet "
            "importeren — zorg dat ze in dezelfde map staan, of gebruik "
            "BRON = 'CSV_LIVE' / 'HANDMATIG' in plaats van 'AUTOMATISCH'."
        ) from e

    # TODO: pas deze instellingen aan naar jouw opstelling
    ARDUINO_POORT = "COM5"
    ARDUINO_BAUDRATE = 9600
    STAP_GROOTTE_GRADEN = 0.5
    EIND_HOEK_GRADEN = 25.0
    KANAAL = 1
    MIN_PIEK_AFSTAND_S = 0.005
    PIEK_PROMINENTIE_V = 0.02
    ONZEKERHEID_PER_PUNT = 1.0   # bijv. 1 franje onzekerheid per telstap

    scope = connect_scope()
    cumulatief_N = 0

    try:
        while True:
            werkelijke_hoek = draai_en_lees_hoek(
                ARDUINO_POORT, ARDUINO_BAUDRATE, STAP_GROOTTE_GRADEN
            )

            t, v = acquire_waveform(scope, channel=KANAAL, mode="NORM")
            piek_idx = detect_peaks(t, v, MIN_PIEK_AFSTAND_S,
                                     PIEK_PROMINENTIE_V, "max")
            cumulatief_N += len(piek_idx)

            yield (werkelijke_hoek, cumulatief_N, ONZEKERHEID_PER_PUNT)

            if werkelijke_hoek >= EIND_HOEK_GRADEN:
                break
    finally:
        scope.close()


# =====================================================================
# Live plot
# =====================================================================

def maak_live_plot():
    plt.ion()
    fig, (ax_data, ax_n) = plt.subplots(1, 2, figsize=(13, 5))

    ax_data.set_xlabel("invalshoek (graden)")
    ax_data.set_ylabel("aantal franjes")
    ax_data.grid()

    ax_n.set_xlabel("aantal meetpunten")
    ax_n.set_ylabel("geschatte n")
    ax_n.grid()
    ax_n.set_title("Nauwkeurigheid van n tijdens het draaien")

    return fig, ax_data, ax_n


def update_live_plot(fig, ax_data, ax_n,
                      hoek_graden, aantal_fransjes, onzekerheid_N,
                      n_geschiedenis, onzekerheid_geschiedenis):
    ax_data.cla()
    ax_data.set_xlabel("invalshoek (graden)")
    ax_data.set_ylabel("aantal franjes")
    ax_data.grid()
    ax_data.errorbar(hoek_graden, aantal_fransjes, yerr=onzekerheid_N,
                      fmt="o", label="metingen")

    if n_geschiedenis:
        n_nu = n_geschiedenis[-1]
        onz_nu = onzekerheid_geschiedenis[-1]
        hoek_theorie = np.linspace(0, max(hoek_graden) * 1.05, 200)
        N_theorie = brekingshoek_formule(hoek_theorie, n_nu)
        ax_data.plot(hoek_theorie, N_theorie,
                     label=f"fit, n = {n_nu:.4f} ± {onz_nu:.4f}")
    ax_data.legend()

    ax_n.cla()
    ax_n.set_xlabel("aantal meetpunten")
    ax_n.set_ylabel("geschatte n")
    ax_n.grid()
    ax_n.set_title("Nauwkeurigheid van n tijdens het draaien")
    if n_geschiedenis:
        x = np.arange(1, len(n_geschiedenis) + 1)
        ax_n.errorbar(x, n_geschiedenis, yerr=onzekerheid_geschiedenis,
                       fmt="-o", markersize=4)

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


# =====================================================================
# Hoofdprogramma
# =====================================================================

def main():
    hoek_graden, aantal_fransjes, onzekerheid_N = [], [], []
    n_geschiedenis, onzekerheid_geschiedenis = [], []

    fig, ax_data, ax_n = maak_live_plot()

    def verwerk_nieuw_punt(hoek, N, onzeker):
        hoek_graden.append(hoek)
        aantal_fransjes.append(N)
        onzekerheid_N.append(onzeker)

        if len(hoek_graden) >= MIN_PUNTEN_VOOR_FIT:
            n, onz = fit_n(np.array(hoek_graden), np.array(aantal_fransjes),
                            np.array(onzekerheid_N))
            if n is not None:
                n_geschiedenis.append(n)
                onzekerheid_geschiedenis.append(onz)
                print(f"[{len(hoek_graden)} punten] "
                      f"hoek = {hoek:.2f} graden, N = {N:.1f}  ->  "
                      f"n = {n:.5f} ± {onz:.5f}")
            else:
                print(f"[{len(hoek_graden)} punten] fit lukte nog niet, "
                      "meer/betere data nodig.")
        else:
            print(f"[{len(hoek_graden)} punt(en)] nog te weinig voor een fit "
                  f"(minimaal {MIN_PUNTEN_VOOR_FIT} nodig).")

        update_live_plot(fig, ax_data, ax_n, hoek_graden, aantal_fransjes,
                          onzekerheid_N, n_geschiedenis,
                          onzekerheid_geschiedenis)

    try:
        if BRON == "TIJD":
            for hoek, N, onzeker in tijd_gebaseerde_meetreeks():
                verwerk_nieuw_punt(hoek, N, onzeker)

        elif BRON == "CSV_LIVE":
            if NEEM_BESTAANDE_CSV_DATA_MEE or not os.path.exists(CSV_PAD):
                reeds_gelezen = 0
            else:
                bestaande_data = np.loadtxt(CSV_PAD, delimiter=",",
                                             skiprows=1, ndmin=2)
                reeds_gelezen = len(bestaande_data)
                print(f"{reeds_gelezen} bestaande regel(s) overgeslagen — "
                      "alleen nieuwe regels vanaf nu tellen mee.")
            print(f"Houd '{CSV_PAD}' in de gaten voor nieuwe regels... "
                  "(Ctrl+C om te stoppen)")
            while True:
                nieuwe_punten, reeds_gelezen = nieuwe_punten_csv_live(reeds_gelezen)
                for hoek, N, onzeker in nieuwe_punten:
                    verwerk_nieuw_punt(hoek, N, onzeker)
                plt.pause(POLL_INTERVAL_S)

        elif BRON == "HANDMATIG":
            while True:
                punt = nieuw_punt_handmatig()
                if punt is None:
                    break
                verwerk_nieuw_punt(*punt)

        elif BRON == "AUTOMATISCH":
            for hoek, N, onzeker in automatische_meetreeks():
                verwerk_nieuw_punt(hoek, N, onzeker)

        else:
            raise ValueError("BRON moet 'CSV_LIVE', 'HANDMATIG' of 'AUTOMATISCH' zijn.")

    except KeyboardInterrupt:
        print("\nMeting gestopt door gebruiker.")

    # --- Eindresultaat opslaan, net als in jouw oorspronkelijke code ---
    if n_geschiedenis:
        print(f"\n--- EINDRESULTAAT ---")
        print(f"n = {n_geschiedenis[-1]:.4f} ± {onzekerheid_geschiedenis[-1]:.4f} "
              f"(gebaseerd op {len(hoek_graden)} punten)")

        os.makedirs(OUT_MAP, exist_ok=True)
        fig.savefig(os.path.join(OUT_MAP, OUT_PLOT))
        print(f"Plot opgeslagen in: {os.path.join(OUT_MAP, OUT_PLOT)}")

        eind_csv = np.column_stack([hoek_graden, aantal_fransjes, onzekerheid_N])
        np.savetxt(
            os.path.join(OUT_MAP, OUT_CSV), eind_csv, delimiter=",",
            header="hoek_graden,aantal_fransjes,onzekerheid_N", comments="",
        )
        print(f"Meetdata opgeslagen in: {os.path.join(OUT_MAP, OUT_CSV)}")
    else:
        print("Geen bruikbare fit gelukt — te weinig of geen data ontvangen.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()