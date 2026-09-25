"""
Franjes tellen met een Rigol DS2000-serie oscilloscoop
(bijv. DS2102A)

Deze versie:
1. Maakt automatisch verbinding met de oscilloscoop via USB (VISA).
2. Leest de instellingen van de scoop uit.
3. Haalt herhaaldelijk het signaal van kanaal 1 op.
4. Rekent de ruwe data om naar volt en seconden.
5. Smooth het volledige interferentiesignaal.
6. Telt iedere lokale maximale piek.
7. Markeert de gevonden pieken in een grafiek.
8. Slaat alle meetdata en het resultaat op in CSV-bestanden.

Benodigde packages:
    pip install pyvisa pyvisa-py numpy matplotlib scipy
"""

import sys
import time
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter

try:
    import pyvisa
except ImportError:
    sys.exit(
        "pyvisa is niet geinstalleerd. "
        "Run: pip install pyvisa pyvisa-py"
    )


# =====================================================================
# INSTELLINGEN
# =====================================================================

# VISA-resource string van de scoop.
# None = automatisch zoeken.
VISA_RESOURCE = None

# Welk kanaal moet worden uitgelezen
CHANNEL = 1

# Meetmodus:
# "HANDMATIG" = Enter -> draaien -> Ctrl+C
# "TIJD"      = automatisch stoppen na MEETDUUR_S
MEETMODUS = "HANDMATIG"

# Alleen gebruikt bij MEETMODUS = "TIJD"
MEETDUUR_S = 10.0

# Golfvorm:
# "NORM" = wat op het scherm staat
# "RAW"  = volledige geheugendiepte
WAV_MODE = "NORM"


# =====================================================================
# PIEKHERKENNING
# =====================================================================

# We willen iedere lokale piek tellen.
# Daarom geen minimale tijd tussen pieken.
MIN_PIEK_AFSTAND_S = 0


# =====================================================================
# SMOOTHINSTELLINGEN
# =====================================================================

# Hoeveel meetpunten worden gebruikt voor smoothing.
#
# 7 = weinig smoothing
# 21 = meer smoothing
# 31 = nog meer smoothing
#
# Begin met 7 zodat zoveel mogelijk pieken behouden blijven.
SMOOTH_WINDOW = 7

SMOOTH_POLYORDER = 2


# =====================================================================
# PIEKMODUS
# =====================================================================

# "max"  = alleen maxima tellen
# "min"  = alleen minima tellen
# "both" = maxima + minima tellen
PIEK_MODUS = "max"


# =====================================================================
# OUTPUTBESTANDEN
# =====================================================================

CSV_DATA_BESTAND = "franjes_meetdata.csv"
CSV_RESULTAAT_BESTAND = "franjes_resultaat.csv"


# =====================================================================
# HULPKLASSEN
# =====================================================================

@dataclass
class ScoopInstellingen:
    tijdschaal_s_div: float
    tijd_offset_s: float
    spanningschaal_v_div: float
    spanning_offset_v: float


# =====================================================================
# VERBINDING MAKEN MET OSCILLOSCOOP
# =====================================================================

def connect_scope(resource=None, timeout_ms=5000):

    rm = pyvisa.ResourceManager()

    if resource is None:

        beschikbare = rm.list_resources()

        if not beschikbare:
            raise ConnectionError(
                "Geen VISA-apparaten gevonden. "
                "Is de oscilloscoop aangesloten en aan staan, "
                "en zijn de VISA-drivers geinstalleerd?"
            )

        # Zoek naar Rigol
        rigol_kandidaten = [
            r for r in beschikbare
            if "0x1AB1" in r or "1AB1" in r
        ]

        if rigol_kandidaten:
            resource = rigol_kandidaten[0]
        else:
            resource = beschikbare[0]

        print(
            f"Automatisch gekozen VISA-resource: {resource}"
        )

    try:

        scope = rm.open_resource(resource)

        scope.timeout = timeout_ms

        idn = scope.query("*IDN?").strip()

    except Exception as e:

        raise ConnectionError(
            f"Kon geen verbinding maken met '{resource}'.\n"
            f"Controleer of de oscilloscoop aan staat "
            f"en via USB is aangesloten.\n"
            f"Details: {e}"
        )

    if "RIGOL" not in idn.upper():

        print(
            "Waarschuwing: apparaat reageert, "
            f"maar lijkt geen Rigol te zijn: {idn}"
        )

    else:

        print(f"Verbonden met: {idn}")

    return scope


# =====================================================================
# INSTELLINGEN VAN DE OSCILLOSCOOP UITLEZEN
# =====================================================================

def read_scope_settings(scope, channel=1):

    tijdschaal = float(
        scope.query(":TIM:SCAL?")
    )

    tijd_offset = float(
        scope.query(":TIM:OFFS?")
    )

    spanningschaal = float(
        scope.query(f":CHAN{channel}:SCAL?")
    )

    spanning_offset = float(
        scope.query(f":CHAN{channel}:OFFS?")
    )

    instellingen = ScoopInstellingen(
        tijdschaal_s_div=tijdschaal,
        tijd_offset_s=tijd_offset,
        spanningschaal_v_div=spanningschaal,
        spanning_offset_v=spanning_offset,
    )

    print(
        f"Tijdschaal: "
        f"{instellingen.tijdschaal_s_div:.3e} s/div"
    )

    print(
        f"Tijd-offset: "
        f"{instellingen.tijd_offset_s:.3e} s"
    )

    print(
        f"Spanningsschaal: "
        f"{instellingen.spanningschaal_v_div:.3f} V/div"
    )

    print(
        f"Spanning-offset: "
        f"{instellingen.spanning_offset_v:.3f} V"
    )

    return instellingen


# =====================================================================
# GOLFVORM OPHALEN
# =====================================================================

def acquire_waveform(scope, channel=1, mode="NORM"):

    scope.write(
        f":WAV:SOUR CHAN{channel}"
    )

    scope.write(
        ":WAV:FORM BYTE"
    )

    scope.write(
        f":WAV:MODE {mode}"
    )

    raw = scope.query_binary_values(
        ":WAV:DATA?",
        datatype="B",
        container=np.array
    )

    if raw.size == 0:

        raise RuntimeError(
            "Geen data ontvangen van de oscilloscoop."
        )

    # Preamble uitlezen
    preamble = (
        scope.query(":WAV:PRE?")
        .strip()
        .split(",")
    )

    x_inc = float(preamble[4])
    x_origin = float(preamble[5])
    x_ref = float(preamble[6])

    y_inc = float(preamble[7])
    y_origin = float(preamble[8])
    y_ref = float(preamble[9])

    # Tijdas
    idx = np.arange(raw.size)

    tijd = (
        (idx - x_ref) * x_inc
        + x_origin
    )

    # Spanningsas
    spanning = (
        raw.astype(np.float64)
        - y_origin
        - y_ref
    ) * y_inc

    return tijd, spanning


# =====================================================================
# FRANJES DETECTEREN
# =====================================================================

def detect_peaks(
    tijd,
    spanning,
    min_afstand_s,
    modus="max"
):

    """
    Smootht het volledige signaal en zoekt daarna
    iedere lokale maximale of minimale piek.

    Er wordt GEEN prominence-filter gebruikt.
    """

    if len(tijd) < 5:

        return np.array([], dtype=int), spanning

    # -------------------------------------------------------------
    # 1. Smooth-window bepalen
    # -------------------------------------------------------------

    window = min(
        SMOOTH_WINDOW,
        len(spanning)
    )

    # Window moet oneven zijn
    if window % 2 == 0:
        window -= 1

    # Window moet groter zijn dan polyorder
    if window <= SMOOTH_POLYORDER:
        window = SMOOTH_POLYORDER + 3

    # Nogmaals controleren
    if window % 2 == 0:
        window += 1

    # -------------------------------------------------------------
    # 2. Signaal smoothen
    # -------------------------------------------------------------

    spanning_smooth = savgol_filter(
        spanning,
        window_length=window,
        polyorder=SMOOTH_POLYORDER
    )

    # -------------------------------------------------------------
    # 3. Tijdstap bepalen
    # -------------------------------------------------------------

    dt = np.mean(
        np.diff(tijd)
    )

    if dt <= 0:

        raise ValueError(
            "Ongeldige tijdstap in de meetdata."
        )

    # -------------------------------------------------------------
    # 4. Minimale afstand
    # -------------------------------------------------------------

    if min_afstand_s <= 0:

        min_afstand_samples = 1

    else:

        min_afstand_samples = max(
            1,
            int(
                round(
                    min_afstand_s / dt
                )
            )
        )

    # -------------------------------------------------------------
    # 5. ALLE PIEKEN ZOEKEN
    # -------------------------------------------------------------

    if modus == "max":

        piek_idx, eigenschappen = find_peaks(
            spanning_smooth,
            distance=min_afstand_samples
        )

    elif modus == "min":

        piek_idx, eigenschappen = find_peaks(
            -spanning_smooth,
            distance=min_afstand_samples
        )

    elif modus == "both":

        max_idx, _ = find_peaks(
            spanning_smooth,
            distance=min_afstand_samples
        )

        min_idx, _ = find_peaks(
            -spanning_smooth,
            distance=min_afstand_samples
        )

        piek_idx = np.sort(
            np.concatenate(
                [max_idx, min_idx]
            )
        )

    else:

        raise ValueError(
            "PIEK_MODUS moet 'max', 'min' of 'both' zijn."
        )

    # -------------------------------------------------------------
    # 6. Resultaat printen
    # -------------------------------------------------------------

    print()
    print("--- PIEKDETECTIE ---")

    print(
        f"Smooth-window: "
        f"{window} meetpunten"
    )

    print(
        f"Minimale afstand: "
        f"{min_afstand_samples} samples"
    )

    print(
        f"Alle lokale pieken geteld: "
        f"{len(piek_idx)}"
    )

    return piek_idx, spanning_smooth


# =====================================================================
# CONTINUE METING
# =====================================================================

def continue_meting(
    scope,
    channel,
    duur_s,
    mode
):

    alle_tijd = []
    alle_spanning = []

    start_wallclock = time.time()

    tijd_offset_totaal = 0.0

    n_acquisities = 0

    vorige_v = None

    if duur_s is None:

        print()
        print(
            "Meting gestart — draai nu je hoek."
        )

        print(
            "Druk Ctrl+C zodra je klaar bent."
        )

    try:

        while (
            duur_s is None
            or
            (
                time.time() - start_wallclock
                < duur_s
            )
        ):

            try:

                t, v = acquire_waveform(
                    scope,
                    channel=channel,
                    mode=mode
                )

            except RuntimeError as e:

                print(
                    f"Waarschuwing: acquisitie "
                    f"overgeslagen ({e})"
                )

                continue

            # -----------------------------------------------------
            # Controleer of dezelfde data opnieuw binnenkomt
            # -----------------------------------------------------

            if (
                vorige_v is not None
                and np.array_equal(v, vorige_v)
            ):

                continue

            vorige_v = v.copy()

            # -----------------------------------------------------
            # Tijd achter elkaar zetten
            # -----------------------------------------------------

            t_verschoven = (
                t
                - t[0]
                + tijd_offset_totaal
            )

            # -----------------------------------------------------
            # Data bewaren
            # -----------------------------------------------------

            alle_tijd.append(
                t_verschoven
            )

            alle_spanning.append(
                v
            )

            # -----------------------------------------------------
            # Tijd-offset volgende acquisitie
            # -----------------------------------------------------

            dt = (
                t_verschoven[1]
                - t_verschoven[0]
            )

            tijd_offset_totaal = (
                t_verschoven[-1]
                + dt
            )

            n_acquisities += 1

    except KeyboardInterrupt:

        print()
        print(
            "Meting gestopt door gebruiker."
        )

    if n_acquisities == 0:

        raise RuntimeError(
            "Geen enkele acquisitie is gelukt "
            "tijdens de meting."
        )

    # -------------------------------------------------------------
    # Alle acquisities samenvoegen
    # -------------------------------------------------------------

    tijd_totaal = np.concatenate(
        alle_tijd
    )

    spanning_totaal = np.concatenate(
        alle_spanning
    )

    # -------------------------------------------------------------
    # Franjes zoeken in volledige signaal
    # -------------------------------------------------------------

    piek_idx, spanning_smooth = detect_peaks(
        tijd_totaal,
        spanning_totaal,
        MIN_PIEK_AFSTAND_S,
        modus=PIEK_MODUS
    )

    # Tijdstippen van gevonden franjes
    piek_tijden_totaal = (
        tijd_totaal[piek_idx]
    )

    werkelijke_duur = (
        time.time()
        - start_wallclock
    )

    print()
    print(
        f"{n_acquisities} acquisities "
        f"uitgevoerd in "
        f"{werkelijke_duur:.1f} s."
    )

    return (
        tijd_totaal,
        spanning_totaal,
        piek_tijden_totaal,
        spanning_smooth
    )


# =====================================================================
# GRAFIEK
# =====================================================================

def plot_resultaat(
    tijd,
    spanning,
    spanning_smooth,
    piek_tijden,
    piek_spanningen,
    modus
):

    plt.figure(
        figsize=(11, 5)
    )

    # Ruwe meetdata
    plt.plot(
        tijd,
        spanning,
        linewidth=0.5,
        alpha=0.35,
        label="Ruw signaal CH1"
    )

    # Gesmoothd signaal
    plt.plot(
        tijd,
        spanning_smooth,
        linewidth=1.2,
        label="Gesmoothd signaal"
    )

    # Gevonden franjes
    plt.plot(
        piek_tijden,
        piek_spanningen,
        "rx",
        markersize=7,
        label=(
            f"Gedetecteerde franjes "
            f"({modus})"
        )
    )

    plt.xlabel(
        "Tijd (s)"
    )

    plt.ylabel(
        "Spanning (V)"
    )

    plt.title(
        f"Interferentiesignaal — "
        f"{len(piek_tijden)} franjes geteld"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================================
# DATA OPSLAAN
# =====================================================================

def opslaan_csv(
    tijd,
    spanning,
    bestandsnaam
):

    with open(
        bestandsnaam,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "tijd_s",
                "spanning_V"
            ]
        )

        writer.writerows(
            zip(
                tijd,
                spanning
            )
        )

    print(
        f"Meetdata opgeslagen in: "
        f"{bestandsnaam}"
    )


# =====================================================================
# RESULTAAT OPSLAAN
# =====================================================================

def opslaan_resultaat(
    bestandsnaam,
    aantal_franjes,
    duur_s,
    gem_tijd_tussen_franjes,
    instellingen
):

    with open(
        bestandsnaam,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "parameter",
                "waarde"
            ]
        )

        writer.writerow(
            [
                "aantal_franjes",
                aantal_franjes
            ]
        )

        writer.writerow(
            [
                "meetduur_s",
                duur_s
            ]
        )

        writer.writerow(
            [
                "gem_tijd_tussen_franjes_s",
                gem_tijd_tussen_franjes
            ]
        )

        writer.writerow(
            [
                "tijdschaal_s_div",
                instellingen.tijdschaal_s_div
            ]
        )

        writer.writerow(
            [
                "spanningschaal_v_div",
                instellingen.spanningschaal_v_div
            ]
        )

        writer.writerow(
            [
                "kanaal",
                CHANNEL
            ]
        )

        writer.writerow(
            [
                "piek_modus",
                PIEK_MODUS
            ]
        )

        writer.writerow(
            [
                "min_piek_afstand_s",
                MIN_PIEK_AFSTAND_S
            ]
        )

        writer.writerow(
            [
                "smooth_window",
                SMOOTH_WINDOW
            ]
        )

        writer.writerow(
            [
                "smooth_polyorder",
                SMOOTH_POLYORDER
            ]
        )

    print(
        f"Resultaatoverzicht opgeslagen in: "
        f"{bestandsnaam}"
    )


# =====================================================================
# HOOFDPROGRAMMA
# =====================================================================

def main():

    # -------------------------------------------------------------
    # Verbinding maken
    # -------------------------------------------------------------

    try:

        scope = connect_scope(
            VISA_RESOURCE
        )

    except ConnectionError as e:

        sys.exit(
            f"FOUT: {e}"
        )

    try:

        # ---------------------------------------------------------
        # Instellingen uitlezen
        # ---------------------------------------------------------

        instellingen = read_scope_settings(
            scope,
            channel=CHANNEL
        )

        # ---------------------------------------------------------
        # Meetduur bepalen
        # ---------------------------------------------------------

        if MEETMODUS == "HANDMATIG":

            input(
                "Druk Enter om de meting te starten, "
                "draai daarna je hoek en druk "
                "Ctrl+C zodra je klaar bent..."
            )

            gekozen_duur = None

        elif MEETMODUS == "TIJD":

            gekozen_duur = MEETDUUR_S

        else:

            sys.exit(
                "FOUT: MEETMODUS moet "
                "'HANDMATIG' of 'TIJD' zijn."
            )

        # ---------------------------------------------------------
        # Meting uitvoeren
        # ---------------------------------------------------------

        (
            tijd,
            spanning,
            piek_tijden,
            spanning_smooth
        ) = continue_meting(
            scope,
            channel=CHANNEL,
            duur_s=gekozen_duur,
            mode=WAV_MODE
        )

    finally:

        scope.close()

    # -------------------------------------------------------------
    # Aantal franjes
    # -------------------------------------------------------------

    aantal_franjes = len(
        piek_tijden
    )

    # -------------------------------------------------------------
    # Gemiddelde tijd tussen franjes
    # -------------------------------------------------------------

    if aantal_franjes > 1:

        gem_tijd_tussen_franjes = float(
            np.mean(
                np.diff(
                    piek_tijden
                )
            )
        )

    else:

        gem_tijd_tussen_franjes = float(
            "nan"
        )

    # -------------------------------------------------------------
    # Spanningswaarden van de pieken
    # -------------------------------------------------------------

    piek_spanningen = np.interp(
        piek_tijden,
        tijd,
        spanning_smooth
    )

    # -------------------------------------------------------------
    # Resultaat printen
    # -------------------------------------------------------------

    print()
    print(
        "=============================="
    )

    print(
        "RESULTAAT"
    )

    print(
        "=============================="
    )

    print(
        f"Totale meetduur: "
        f"{tijd[-1] - tijd[0]:.4f} s"
    )

    print(
        f"Aantal getelde franjes: "
        f"{aantal_franjes}"
    )

    print(
        f"Gem. tijd tussen franjes: "
        f"{gem_tijd_tussen_franjes:.4e} s"
    )

    print(
        "=============================="
    )

    # -------------------------------------------------------------
    # CSV opslaan
    # -------------------------------------------------------------

    opslaan_csv(
        tijd,
        spanning,
        CSV_DATA_BESTAND
    )

    opslaan_resultaat(
        CSV_RESULTAAT_BESTAND,
        aantal_franjes,
        tijd[-1] - tijd[0],
        gem_tijd_tussen_franjes,
        instellingen
    )

    # -------------------------------------------------------------
    # Grafiek
    # -------------------------------------------------------------

    plot_resultaat(
        tijd,
        spanning,
        spanning_smooth,
        piek_tijden,
        piek_spanningen,
        PIEK_MODUS
    )


# =====================================================================
# START
# =====================================================================

if __name__ == "__main__":
    main()