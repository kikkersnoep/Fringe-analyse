
import sys
import time
import csv
import numpy as np

try:
    import pyvisa
except ImportError:
    sys.exit(
        "pyvisa is niet geinstalleerd. "
        "Run: pip install pyvisa pyvisa-py"
    )


# ============================================================
# INSTELLINGEN
# ============================================================

VISA_RESOURCE = None
CHANNEL = 1


CSV_BESTAND = "gem_fringe.csv"

# Maximale hoeveelheid punten die we per keer ophalen.
# Als de oscilloscoop meer punten heeft, halen we ze in blokken op.
BLOK_GROOTTE = 100000
GEMIDDELDE_GROOTTE = 50

# ============================================================
# VERBINDING
# ============================================================

def connect_scope(resource=None, timeout_ms=10000):

    rm = pyvisa.ResourceManager()

    if resource is None:

        beschikbare = rm.list_resources()

        if not beschikbare:
            raise ConnectionError(
                "Geen VISA-apparaten gevonden."
            )

        rigol_kandidaten = [
            r for r in beschikbare
            if "0x1AB1" in r or "1AB1" in r
        ]

        if rigol_kandidaten:
            resource = rigol_kandidaten[0]
        else:
            resource = beschikbare[0]

        print(f"Automatisch gekozen VISA-resource: {resource}")

    try:

        scope = rm.open_resource(resource)

        scope.timeout = timeout_ms

        idn = scope.query("*IDN?").strip()

    except Exception as e:

        raise ConnectionError(
            f"Kon geen verbinding maken met '{resource}'.\n"
            f"Details: {e}"
        )

    print(f"Verbonden met: {idn}")

    return scope


# ============================================================
# WACHTEN TOT SINGLE METING KLAAR IS
# ============================================================

def wacht_tot_stop(scope):

    print()
    print("Wachten tot de Single-meting klaar is...")

    while True:

        status = scope.query(":TRIG:STAT?").strip().upper()

        if status == "STOP":
            break

        time.sleep(0.05)

    print("Meting klaar.")


# ============================================================
# RAW WAVEFORM UITLEZEN
# ============================================================

def acquire_raw_waveform(scope, channel=1):

    # Zorg dat de oscilloscoop gestopt is.
    scope.write(":STOP")

    # Kanaal kiezen
    scope.write(f":WAV:SOUR CHAN{channel}")

    # RAW = interne geheugenbuffer
    scope.write(":WAV:MODE RAW")

    # BYTE = 1 byte per meetpunt
    scope.write(":WAV:FORM BYTE")

    # Preambule uitlezen
    preamble = scope.query(":WAV:PRE?").strip().split(",")

    x_inc = float(preamble[4])
    x_origin = float(preamble[5])
    x_ref = float(preamble[6])

    y_inc = float(preamble[7])
    y_origin = float(preamble[8])
    y_ref = float(preamble[9])

    aantal_punten = int(float(preamble[2]))

    print()
    print(f"Aantal RAW-punten beschikbaar: {aantal_punten}")

    if aantal_punten <= 0:
        raise RuntimeError(
            "De oscilloscoop geeft geen RAW-meetpunten terug."
        )

    # ========================================================
    # DATA IN BLOKKEN OPHALEN
    # ========================================================

    alle_raw = []

    start = 1

    while start <= aantal_punten:

        stop = min(
            start + BLOK_GROOTTE - 1,
            aantal_punten
        )

        print(
            f"Meetpunten {start} t/m {stop} ophalen..."
        )

        scope.write(f":WAV:STAR {start}")
        scope.write(f":WAV:STOP {stop}")

        raw = scope.query_binary_values(
            ":WAV:DATA?",
            datatype="B",
            container=np.array
        )

        alle_raw.append(raw)

        start = stop + 1

    raw = np.concatenate(alle_raw)

    print()
    print(f"Daadwerkelijk ontvangen punten: {len(raw)}")

       # ========================================================
    # TIJD EN SPANNING OMREKENEN
    # ========================================================

    index = np.arange(len(raw))

    tijd = (
        (index - x_ref) * x_inc
        + x_origin
    )

    spanning = (
        raw.astype(np.float64) - y_ref
    ) * y_inc + y_origin

    # ========================================================
    # GEMIDDELDE PER BLOK
    # ========================================================

    aantal_blokken = len(raw) // GEMIDDELDE_GROOTTE
    bruikbare_punten = aantal_blokken * GEMIDDELDE_GROOTTE

    tijd = tijd[:bruikbare_punten].reshape(
        aantal_blokken, GEMIDDELDE_GROOTTE
    ).mean(axis=1)

    spanning = spanning[:bruikbare_punten].reshape(
        aantal_blokken, GEMIDDELDE_GROOTTE
    ).mean(axis=1)

    print(f"Meetpunten na middelen: {len(tijd)}")

    return tijd, spanning


# ============================================================
# CSV OPSLAAN
# ============================================================

def opslaan_csv(tijd, spanning, bestandsnaam):

    with open(
        bestandsnaam,
        "w",
        newline=""
    ) as bestand:

        writer = csv.writer(bestand)

        writer.writerow([
            "tijd_s",
            "spanning_V"
        ])

        writer.writerows(
            zip(tijd, spanning)
        )

    print()
    print(
        f"CSV opgeslagen als: {bestandsnaam}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        scope = connect_scope(
            VISA_RESOURCE
        )

    except ConnectionError as e:

        sys.exit(f"FOUT: {e}")

    try:

        print()
        print("==============================")
        print("SINGLE-METING")
        print("==============================")

        print()
        print(
            "Zet de oscilloscoop op SINGLE."
        )

        input(
            "Druk daarna op Enter en druk op FORCE op de oscilloscoop..."
        )

        # Wacht totdat de Single-acquisitie klaar is
        wacht_tot_stop(scope)

        # RAW-data uit geheugen ophalen
        tijd, spanning = acquire_raw_waveform(
            scope,
            channel=CHANNEL
        )

    finally:

        scope.close()

    # CSV opslaan
    opslaan_csv(
        tijd,
        spanning,
        CSV_BESTAND
    )

    print()
    print("==============================")
    print("KLAAR")
    print("==============================")

    print(
        f"Aantal meetpunten: {len(tijd)}"
    )

    print(
        f"Meetduur: {tijd[-1] - tijd[0]:.6f} s"
    )

    print(
        f"Bestand: {CSV_BESTAND}"

    )


if __name__ == "__main__":
    main()
