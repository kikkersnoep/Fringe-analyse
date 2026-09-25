"""
Franjes tellen met een Rigol DS2000-serie oscilloscoop (bijv. DS2102A)
=======================================================================

Dit script:
  1. Maakt automatisch verbinding met de oscilloscoop via USB (VISA).
  2. Leest de instellingen van de scoop uit (tijdschaal, spanningsschaal, posities).
  3. Haalt herhaaldelijk het signaal van kanaal 1 op, gedurende een ingestelde tijd.
  4. Rekent de ruwe data om naar echte volt/seconden met de preamble van de scoop.
  5. Telt automatisch de franjes (pieken) in het interferentiesignaal.
  6. Markeert de gevonden pieken in een grafiek.
  7. Slaat alle meetdata en het resultaat op in CSV-bestanden.

Benodigde packages:
    pip install pyvisa pyvisa-py numpy matplotlib scipy

Auteur: gegenereerd door Claude, pas gerust aan naar wens.
"""

import sys
import time
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

try:
    import pyvisa
except ImportError:
    sys.exit("pyvisa is niet geinstalleerd. Run: pip install pyvisa pyvisa-py")


# =====================================================================
# INSTELLINGEN — dit is het enige stuk dat je meestal hoeft aan te passen
# =====================================================================

# VISA-resource string van de scoop. Laat op None om automatisch te zoeken
# naar een Rigol-apparaat (aanbevolen). Vul anders het exacte adres in,
# bijv. "USB0::0x1AB1::0x04B0::DS2D232801993::INSTR"
VISA_RESOURCE = None

# Welk kanaal moet uitgelezen worden
CHANNEL = 1

# Hoe lang moet er continu gemeten worden (in seconden)
MEETDUUR_S = 10.0

# Golfvorm-mode: "NORM" = wat er op het scherm staat (snel, ~1200 punten),
# "RAW" = volledige geheugendiepte (trager, meer punten per acquisitie)
WAV_MODE = "NORM"

# --- Piekherkenning ---
# Minimale tijd tussen twee opeenvolgende pieken (in seconden). Alles wat
# dichter bij elkaar ligt wordt als 1 piek/franje gezien. Zet dit iets
# lager dan de kortst verwachte tijd tussen twee franjes.
MIN_PIEK_AFSTAND_S = 0.005

# Prominentie = hoe duidelijk een piek boven de "ruis" uit moet steken,
# in volt. Hoger = strenger (minder gevoelig voor ruis), lager = gevoeliger.
PIEK_PROMINENTIE_V = 0.02

# "max" = alleen toppen tellen, "min" = alleen dalen, "both" = beide
# (bij "both" wordt het aantal pieken + dalen bij elkaar opgeteld)
PIEK_MODUS = "max"

# Bestandsnamen voor de output
CSV_DATA_BESTAND = "franjes_meetdata.csv"
CSV_RESULTAAT_BESTAND = "franjes_resultaat.csv"


# =====================================================================
# Hulpfuncties
# =====================================================================

@dataclass
class ScoopInstellingen:
    tijdschaal_s_div: float
    tijd_offset_s: float
    spanningschaal_v_div: float
    spanning_offset_v: float


def connect_scope(resource=None, timeout_ms=5000):
    """Maakt verbinding met de oscilloscoop. Zoekt automatisch een Rigol
    apparaat als er geen resource-string is opgegeven. Controleert of het
    apparaat daadwerkelijk reageert op *IDN?."""
    rm = pyvisa.ResourceManager()

    if resource is None:
        beschikbare = rm.list_resources()
        if not beschikbare:
            raise ConnectionError(
                "Geen VISA-apparaten gevonden. Is de oscilloscoop aangesloten "
                "en aan staan, en zijn de VISA-drivers geinstalleerd?"
            )
        # Zoek naar iets dat op een Rigol lijkt (vendor ID 0x1AB1)
        rigol_kandidaten = [r for r in beschikbare if "0x1AB1" in r or "1AB1" in r]
        resource = rigol_kandidaten[0] if rigol_kandidaten else beschikbare[0]
        print(f"Automatisch gekozen VISA-resource: {resource}")

    try:
        scope = rm.open_resource(resource)
        scope.timeout = timeout_ms
        idn = scope.query("*IDN?").strip()
    except Exception as e:
        raise ConnectionError(
            f"Kon geen verbinding maken met '{resource}'. Controleer of de "
            f"oscilloscoop aan staat en via USB is aangesloten.\nDetails: {e}"
        )

    if "RIGOL" not in idn.upper():
        print(f"Waarschuwing: apparaat reageert, maar lijkt geen Rigol te zijn: {idn}")
    else:
        print(f"Verbonden met: {idn}")

    return scope


def read_scope_settings(scope, channel=1) -> ScoopInstellingen:
    """Leest tijdbasis- en spanningsinstellingen van de scoop uit."""
    tijdschaal = float(scope.query(":TIM:SCAL?"))
    tijd_offset = float(scope.query(":TIM:OFFS?"))
    spanningschaal = float(scope.query(f":CHAN{channel}:SCAL?"))
    spanning_offset = float(scope.query(f":CHAN{channel}:OFFS?"))

    instellingen = ScoopInstellingen(
        tijdschaal_s_div=tijdschaal,
        tijd_offset_s=tijd_offset,
        spanningschaal_v_div=spanningschaal,
        spanning_offset_v=spanning_offset,
    )

    print(
        f"Tijdschaal: {instellingen.tijdschaal_s_div:.3e} s/div | "
        f"Tijd-offset: {instellingen.tijd_offset_s:.3e} s | "
        f"Spanningschaal: {instellingen.spanningschaal_v_div:.3f} V/div | "
        f"Spanning-offset: {instellingen.spanning_offset_v:.3f} V"
    )
    return instellingen


def acquire_waveform(scope, channel=1, mode="NORM"):
    """Haalt een golfvorm op van de gekozen kanaal en zet de ruwe bytes om
    naar echte tijd (s) en spanning (V), met de preamble van de scoop."""
    scope.write(f":WAV:SOUR CHAN{channel}")
    scope.write(":WAV:FORM BYTE")
    scope.write(f":WAV:MODE {mode}")

    raw = scope.query_binary_values(":WAV:DATA?", datatype="B", container=np.array)

    if raw.size == 0:
        raise RuntimeError(
            "Geen data ontvangen van de oscilloscoop. Controleer of er een "
            "signaal aanwezig is en of het kanaal is ingeschakeld."
        )

    # Preamble: format,type,points,count,xincrement,xorigin,xreference,
    #           yincrement,yorigin,yreference
    preamble = scope.query(":WAV:PRE?").strip().split(",")
    x_inc = float(preamble[4])
    x_origin = float(preamble[5])
    x_ref = float(preamble[6])
    y_inc = float(preamble[7])
    y_origin = float(preamble[8])
    y_ref = float(preamble[9])

    idx = np.arange(raw.size)
    tijd = (idx - x_ref) * x_inc + x_origin
    spanning = (raw.astype(np.float64) - y_origin - y_ref) * y_inc

    return tijd, spanning


def detect_peaks(tijd, spanning, min_afstand_s, prominentie_v, modus="max"):
    """Zoekt pieken (franjes) in het signaal. Retourneert de indices van de
    gevonden pieken. 'min_afstand_s' wordt omgerekend naar samples aan de
    hand van de gemiddelde tijdstap tussen meetpunten."""
    if len(tijd) < 2:
        return np.array([], dtype=int)

    dt = np.mean(np.diff(tijd))
    if dt <= 0:
        raise ValueError("Ongeldige tijdstap in de meetdata (dt <= 0).")

    min_afstand_samples = max(1, int(round(min_afstand_s / dt)))

    if modus == "max":
        piek_idx, _ = find_peaks(
            spanning, distance=min_afstand_samples, prominence=prominentie_v
        )
    elif modus == "min":
        piek_idx, _ = find_peaks(
            -spanning, distance=min_afstand_samples, prominence=prominentie_v
        )
    elif modus == "both":
        top_idx, _ = find_peaks(
            spanning, distance=min_afstand_samples, prominence=prominentie_v
        )
        dal_idx, _ = find_peaks(
            -spanning, distance=min_afstand_samples, prominence=prominentie_v
        )
        piek_idx = np.sort(np.concatenate([top_idx, dal_idx]))
    else:
        raise ValueError("PIEK_MODUS moet 'max', 'min' of 'both' zijn.")

    return piek_idx


def continue_meting(scope, channel, duur_s, mode,
                     min_afstand_s, prominentie_v, modus):
    """Voert herhaalde acquisities uit gedurende 'duur_s' seconden, telt
    steeds de pieken per stuk signaal en plakt alle data achter elkaar.
    Elk stukje krijgt een tijd-offset zodat de tijd-as over de hele
    meting doorloopt."""
    alle_tijd = []
    alle_spanning = []
    alle_piek_tijden = []

    start_wallclock = time.time()
    tijd_offset_totaal = 0.0
    n_acquisities = 0

    while (time.time() - start_wallclock) < duur_s:
        try:
            t, v = acquire_waveform(scope, channel=channel, mode=mode)
        except RuntimeError as e:
            print(f"Waarschuwing: acquisitie overgeslagen ({e})")
            continue

        # Zorg dat elk stukje mooi aansluit op het vorige stukje in tijd
        t_verschoven = t - t[0] + tijd_offset_totaal

        piek_idx = detect_peaks(t_verschoven, v, min_afstand_s,
                                 prominentie_v, modus)

        alle_tijd.append(t_verschoven)
        alle_spanning.append(v)
        alle_piek_tijden.append(t_verschoven[piek_idx])

        tijd_offset_totaal = t_verschoven[-1] + (t_verschoven[1] - t_verschoven[0])
        n_acquisities += 1

    if n_acquisities == 0:
        raise RuntimeError("Geen enkele acquisitie is gelukt binnen de meetduur.")

    tijd_totaal = np.concatenate(alle_tijd)
    spanning_totaal = np.concatenate(alle_spanning)
    piek_tijden_totaal = np.concatenate(alle_piek_tijden)

    print(f"{n_acquisities} acquisities uitgevoerd in {duur_s:.1f} s.")

    return tijd_totaal, spanning_totaal, piek_tijden_totaal


def plot_resultaat(tijd, spanning, piek_tijden, piek_spanningen, modus):
    plt.figure(figsize=(11, 5))
    plt.plot(tijd, spanning, linewidth=0.8, label="Signaal CH1")
    plt.plot(piek_tijden, piek_spanningen, "rx", markersize=7,
              label=f"Gedetecteerde franjes ({modus})")
    plt.xlabel("Tijd (s)")
    plt.ylabel("Spanning (V)")
    plt.title(f"Interferentiesignaal — {len(piek_tijden)} franjes geteld")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def opslaan_csv(tijd, spanning, bestandsnaam):
    with open(bestandsnaam, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tijd_s", "spanning_V"])
        writer.writerows(zip(tijd, spanning))
    print(f"Meetdata opgeslagen in: {bestandsnaam}")


def opslaan_resultaat(bestandsnaam, aantal_franjes, duur_s,
                       gem_tijd_tussen_franjes, instellingen: ScoopInstellingen):
    with open(bestandsnaam, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "waarde"])
        writer.writerow(["aantal_franjes", aantal_franjes])
        writer.writerow(["meetduur_s", duur_s])
        writer.writerow(["gem_tijd_tussen_franjes_s", gem_tijd_tussen_franjes])
        writer.writerow(["tijdschaal_s_div", instellingen.tijdschaal_s_div])
        writer.writerow(["spanningschaal_v_div", instellingen.spanningschaal_v_div])
        writer.writerow(["kanaal", CHANNEL])
        writer.writerow(["piek_modus", PIEK_MODUS])
        writer.writerow(["min_piek_afstand_s", MIN_PIEK_AFSTAND_S])
        writer.writerow(["piek_prominentie_v", PIEK_PROMINENTIE_V])
    print(f"Resultaatoverzicht opgeslagen in: {bestandsnaam}")


# =====================================================================
# Hoofdprogramma
# =====================================================================

def main():
    try:
        scope = connect_scope(VISA_RESOURCE)
    except ConnectionError as e:
        sys.exit(f"FOUT: {e}")

    try:
        instellingen = read_scope_settings(scope, channel=CHANNEL)

        tijd, spanning, piek_tijden = continue_meting(
            scope,
            channel=CHANNEL,
            duur_s=MEETDUUR_S,
            mode=WAV_MODE,
            min_afstand_s=MIN_PIEK_AFSTAND_S,
            prominentie_v=PIEK_PROMINENTIE_V,
            modus=PIEK_MODUS,
        )
    finally:
        scope.close()

    aantal_franjes = len(piek_tijden)

    if aantal_franjes == 0:
        print("Waarschuwing: er zijn geen franjes gedetecteerd. Controleer "
              "de signaalsterkte, PIEK_PROMINENTIE_V en MIN_PIEK_AFSTAND_S.")
    elif aantal_franjes < 3:
        print("Waarschuwing: er zijn heel weinig franjes gevonden — "
              "de piekherkenning is mogelijk onbetrouwbaar. Controleer de "
              "instellingen en het signaal.")

    if aantal_franjes > 1:
        gem_tijd_tussen_franjes = float(np.mean(np.diff(piek_tijden)))
    else:
        gem_tijd_tussen_franjes = float("nan")

    # Voor de plot hebben we ook de spanningswaarden op de piekmomenten nodig
    piek_spanningen = np.interp(piek_tijden, tijd, spanning)

    print("\n--- RESULTAAT ---")
    print(f"Totale meetduur:            {tijd[-1] - tijd[0]:.4f} s")
    print(f"Aantal getelde franjes:     {aantal_franjes}")
    print(f"Gem. tijd tussen franjes:   {gem_tijd_tussen_franjes:.4e} s")

    opslaan_csv(tijd, spanning, CSV_DATA_BESTAND)
    opslaan_resultaat(CSV_RESULTAAT_BESTAND, aantal_franjes,
                       tijd[-1] - tijd[0], gem_tijd_tussen_franjes,
                       instellingen)

    plot_resultaat(tijd, spanning, piek_tijden, piek_spanningen, PIEK_MODUS)


if __name__ == "__main__":
    main()