#!/usr/bin/env python3
"""
fringe_analyzer.py
===================

Bepaalt of een interferentiepatroon op een vast pixelpunt in een video
"groen" (fringe/constructieve interferentie) of "wit" (achtergrond/
destructieve interferentie) is, voor elk frame van de video.

Alle instellingen (videopad, meetpunt, etc.) staan onderaan in main() -
pas ze daar aan en run het script gewoon met:

    python fringe_analyzer.py

Vereisten:
    - ffmpeg / ffprobe moeten in je PATH staan (bv. via
      `conda install -c conda-forge ffmpeg`)
    - pip install numpy matplotlib
    - fringe_status_template.html moet in dezelfde map staan als dit
      script (bevat de opmaak/JavaScript van de interactieve grafiek)

Output (in de door jou gekozen outdir):
    fringe_data.csv               frame, tijd (s), status, meetwaarde
    interferometer_status.png     overzichtsgrafiek (tijd vs status)
    interferometer_status.html    interactieve grafiek (zoom/pan)
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


# ==========================================================================
# STAP 1: video-informatie opvragen
# ==========================================================================
# Waarom: we hebben de fps nodig om frame-nummers om te rekenen naar
# seconden, en de (echte, na rotatie) breedte/hoogte om in de HTML-header
# te tonen. Telefoonvideo's bevatten vaak een rotatie-tag (bv. -90 graden)
# waardoor de "ruwe" afmetingen die ffprobe teruggeeft (bv. 1024x576)
# afwijken van wat je daadwerkelijk ziet als je de video afspeelt
# (576x1024). Omdat ffmpeg bij het uitlezen van frames automatisch
# roteert, en jouw meetpunt (x, y) dus op het GEROTEERDE beeld slaat,
# wisselen we breedte/hoogte hier om als er een 90/-90 graden rotatie is.
# ==========================================================================

def get_video_info(path: str) -> dict:
    """Geeft {'width', 'height', 'fps', 'nb_frames'} terug van de video."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)["streams"][0]

    # r_frame_rate komt als breuk terug, bv. "30/1" -> 30.0
    num, den = info["r_frame_rate"].split("/")
    fps = float(num) / float(den)

    width, height = int(info["width"]), int(info["height"])

    # Aparte ffprobe-aanroep voor de rotatie-metadata (staat in "side_data",
    # niet in de gewone stream-info hierboven).
    rot_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "side_data=rotation",
        "-of", "json", path,
    ]
    rot_result = subprocess.run(rot_cmd, capture_output=True, text=True, check=True)
    rotation = 0
    match = re.search(r'"rotation":\s*(-?\d+)', rot_result.stdout)
    if match:
        rotation = int(match.group(1))
    if abs(rotation) == 90:
        width, height = height, width  # portret/landschap omwisselen

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "nb_frames": int(info.get("nb_frames", 0)) or None,
    }


# ==========================================================================
# STAP 2: pixeldata uit de video extraheren
# ==========================================================================
# Waarom: we willen niet elk frame volledig decoderen in Python (traag,
# veel geheugen voor een video van 224 seconden). In plaats daarvan laten
# we ffmpeg zelf, tijdens het decoderen, meteen een klein gebiedje
# (patch x patch pixels) rond het meetpunt uitknippen met het "crop"
# filter, en de rest weggooien. Het resultaat schrijven we weg als
# "rawvideo": rauwe RGB-bytes zonder compressie/containerformaat, zodat we
# het direct met numpy kunnen inlezen.
# ==========================================================================

def extract_pixel_patch(path: str, x: int, y: int, patch: int) -> np.ndarray:
    """
    Croppt met ffmpeg een patch x patch pixelgebied rond (x, y) uit elk
    frame (na auto-rotatie) en geeft een array terug met vorm
    (n_frames, patch, patch, 3) in uint8 RGB.
    """
    half = patch // 2
    crop_x = max(0, x - half)
    crop_y = max(0, y - half)

    raw_path = Path("_pixel_data.rgb")  # tijdelijk bestand, wordt na afloop verwijderd
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vf", f"crop={patch}:{patch}:{crop_x}:{crop_y},format=rgb24",
        "-f", "rawvideo", str(raw_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    data = np.fromfile(raw_path, dtype=np.uint8)
    raw_path.unlink(missing_ok=True)

    # De ruwe bytes zijn één lange rij; we vouwen ze terug naar
    # (n_frames, patch, patch, 3) zodat elk frame zijn eigen mini-plaatje is.
    frame_bytes = patch * patch * 3
    n_frames = len(data) // frame_bytes
    data = data[: n_frames * frame_bytes].reshape(n_frames, patch, patch, 3)
    return data


# ==========================================================================
# STAP 3: signaalverwerking - ruis onderdrukken
# ==========================================================================
# Waarom: video-compressie (h.264) introduceert kleine, willekeurige
# schommelingen in pixelwaarden van frame tot frame, ook als de werkelijke
# helderheid niet verandert. Een simpel voortschrijdend gemiddelde ("moving
# average") strijkt die schommelingen glad zonder de echte, langzamere
# fringe-overgangen te vervagen (mits het venster klein genoeg blijft
# t.o.v. de snelste fringe-beweging in de video).
# ==========================================================================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ==========================================================================
# STAP 4: automatische drempelbepaling (Otsu)
# ==========================================================================
# Waarom: we willen een grenswaarde tussen "groen" en "wit" die niet
# handmatig hoeft te worden ingesteld voor elke nieuwe video/belichting.
# Otsu's methode zoekt de drempel die de histogram van signaalwaarden in
# twee groepen splitst met een zo groot mogelijk verschil tussen de
# groepsgemiddelden (en een zo klein mogelijke spreiding binnen elke
# groep) - precies wat we nodig hebben voor een bimodaal groen/wit-signaal.
# ==========================================================================

def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Automatische drempel die de twee klassen (groen/wit) het best scheidt."""
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    sum_all = np.sum(hist * centers)

    best_thr, best_var = centers[0], -1.0
    sum_below, weight_below = 0.0, 0.0
    for h, c in zip(hist, centers):
        weight_below += h
        if weight_below == 0:
            continue
        weight_above = total - weight_below
        if weight_above == 0:
            break
        sum_below += h * c
        mean_below = sum_below / weight_below
        mean_above = (sum_all - sum_below) / weight_above
        between_var = weight_below * weight_above * (mean_below - mean_above) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thr = c
    return float(best_thr)


# ==========================================================================
# STAP 5: classificatie met hysteresis (Schmitt-trigger)
# ==========================================================================
# Waarom: met één harde drempel zou het signaal, wanneer het rond die
# drempel dobbert (bv. tijdens een langzame overgang), telkens heen en
# weer "flikkeren" tussen groen/wit terwijl er in werkelijkheid maar één
# overgang plaatsvindt. Door twee drempels te gebruiken (drempel + band en
# drempel - band) moet het signaal duidelijk naar de andere kant
# doorschieten voordat de status omslaat - dit is dezelfde truc als een
# Schmitt-trigger in de elektronica.
# ==========================================================================

def classify_with_hysteresis(signal: np.ndarray, threshold: float, band: float) -> np.ndarray:
    """
    Status blijft 'groen' (1) totdat het signaal onder (threshold - band)
    zakt, en blijft 'wit' (0) totdat het signaal boven (threshold + band)
    komt.
    """
    hi, lo = threshold + band, threshold - band
    state = np.zeros(len(signal), dtype=int)
    current = 1 if signal[0] > threshold else 0
    for i, v in enumerate(signal):
        if current == 1 and v < lo:
            current = 0
        elif current == 0 and v > hi:
            current = 1
        state[i] = current
    return state


def count_cycles(state: np.ndarray) -> int:
    """Eén volledige cyclus = groen->wit->groen = 2 statusovergangen."""
    transitions = int(np.sum(np.abs(np.diff(state))))
    return transitions // 2


# ==========================================================================
# STAP 6: output wegschrijven
# ==========================================================================
# Waarom: drie formaten voor drie doelen -
#   - CSV: ruwe data om zelf verder te verwerken (bv. in Excel/Python) en
#     om de golflengte te berekenen zodra je de spiegelsnelheid kent.
#   - PNG: een vast overzichtsplaatje van de hele meting, handig om te
#     delen of in een verslag te plakken.
#   - HTML: een interactieve versie waarin je kunt in-/uitzoomen om
#     fringes precies te tellen, zonder dat er extra software nodig is
#     om te openen (werkt in elke browser).
# ==========================================================================

def write_csv(path: Path, t: np.ndarray, state: np.ndarray, metric: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "status", "green_metric"])
        for i in range(len(t)):
            w.writerow([i, f"{t[i]:.4f}", "groen" if state[i] else "wit", f"{metric[i]:.2f}"])


def write_overview_png(path: Path, t: np.ndarray, state: np.ndarray, n_cycles: int) -> None:
    import matplotlib
    matplotlib.use("Agg")  # geen grafisch scherm nodig/beschikbaar; direct naar bestand
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.step(t, state, where="post", color="green", linewidth=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wit", "groen"])
    ax.set_xlabel("Tijd (s)")
    ax.set_ylabel("Status")
    ax.set_title(f"Interferentiepatroon status — {n_cycles} volledige fringe-cycli gedetecteerd")
    ax.set_xlim(0, t[-1])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def write_interactive_html(
    template_path: Path, output_path: Path,
    t: np.ndarray, state: np.ndarray, metric: np.ndarray,
    video_name: str, info: dict, x: int, y: int,
) -> None:
    """
    Leest het losse HTML/JS-template in en vervangt de __PLACEHOLDER__
    tokens door de echte meetdata en video-info. Zo blijft alle opmaak/
    JavaScript gescheiden van de Python-logica, en is het template zelf
    ook los te openen/aan te passen als gewone HTML.
    """
    if not template_path.exists():
        sys.exit(
            f"Kan template niet vinden: {template_path}\n"
            "Zorg dat fringe_status_template.html in dezelfde map staat als dit script."
        )

    # Signaal normaliseren naar 0-1 zodat het netjes samen met de status
    # (0 of 1) in dezelfde grafiek te tonen is als referentielijn.
    m_norm = (metric - metric.min()) / (metric.max() - metric.min())
    payload = {
        "t": np.round(t, 3).tolist(),
        "state": state.astype(int).tolist(),
        "metric": np.round(m_norm, 4).tolist(),
    }

    html = template_path.read_text(encoding="utf-8")
    replacements = {
        "__X__": str(x),
        "__Y__": str(y),
        "__VIDEO_NAME__": video_name,
        "__WIDTH__": str(info["width"]),
        "__HEIGHT__": str(info["height"]),
        "__FPS__": f"{info['fps']:.3f}",
        "__N_FRAMES__": str(len(t)),
        "__DURATION__": f"{t[-1]:.2f}",
        "__DATA_JSON__": json.dumps(payload),
    }
    for token, value in replacements.items():
        html = html.replace(token, value)

    output_path.write_text(html, encoding="utf-8")


# ==========================================================================
# MAIN: alle instellingen staan hier - pas ze hier aan naar wens
# ==========================================================================

def main():
    # ---- instellingen (voorheen command-line argumenten) ----------------
    VIDEO_PATH = "interferometer1.mp4"      # pad naar de videofile
    X = 278                                 # x-coördinaat van het meetpunt
    Y = 780                                 # y-coördinaat van het meetpunt
    PATCH_SIZE = 3                          # grootte (in pixels) van het middelingsgebied rond het meetpunt, oneven getal
    SMOOTH_WINDOW = 9                       # aantal frames voor het gladstrijken van het signaal (ruisonderdrukking)
    HYSTERESIS_BAND = 4.0                   # breedte van de hysteresis-band rond de drempel (voorkomt flikkeren)
    THRESHOLD = "auto"                      # "auto" voor automatische Otsu-drempel, of een vast getal
    OUTPUT_DIR = "out"                      # map waarin de resultaten worden weggeschreven
    TEMPLATE_PATH = "fringe_status_template.html"  # HTML/JS-sjabloon voor de interactieve grafiek
    # -----------------------------------------------------------------------

    video_path = Path(VIDEO_PATH)
    if not video_path.exists():
        sys.exit(f"Video niet gevonden: {video_path}")

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Stap 1: video-info ---
    print(f"[1/5] Video-info opvragen ({video_path.name}) ...")
    info = get_video_info(str(video_path))
    print(f"      {info['width']}x{info['height']} @ {info['fps']:.4f} fps")

    # --- Stap 2: pixeldata extraheren ---
    print(f"[2/5] Pixelpatch ({PATCH_SIZE}x{PATCH_SIZE}) rond (x={X}, y={Y}) extraheren ...")
    patch_data = extract_pixel_patch(str(video_path), X, Y, PATCH_SIZE).astype(float)
    n_frames = patch_data.shape[0]
    fps = info["fps"]
    # frame-index -> tijd in seconden, nodig voor de x-as van de grafiek
    t = np.arange(n_frames) / fps
    print(f"      {n_frames} frames uitgelezen ({t[-1]:.2f} s)")

    # --- Stap 3: signaal berekenen ---
    print("[3/5] Signaal berekenen en classificeren ...")
    # Gemiddelde kleur over het patch-gebied (i.p.v. één enkele pixel) om
    # de invloed van pixel-op-pixel ruis verder te verkleinen.
    patch_mean = patch_data.mean(axis=(1, 2))  # -> (n_frames, 3) voor R, G, B
    R, G, B = patch_mean[:, 0], patch_mean[:, 1], patch_mean[:, 2]
    # "Groenheid": hoog wanneer groen (G) sterk domineert over rood+blauw
    # (heldere fringe), laag wanneer de kleur richting grijs/wit gaat
    # (destructieve interferentie, achtergrond zichtbaar).
    metric_raw = G - (R + B) / 2.0
    metric = moving_average(metric_raw, SMOOTH_WINDOW)

    threshold = otsu_threshold(metric) if THRESHOLD == "auto" else float(THRESHOLD)
    print(f"      drempel = {threshold:.2f} (hysteresis ±{HYSTERESIS_BAND})")

    state = classify_with_hysteresis(metric, threshold, HYSTERESIS_BAND)
    n_cycles = count_cycles(state)
    print(f"      {n_cycles} volledige fringe-cycli gedetecteerd")

    # --- Stap 4: output wegschrijven ---
    print("[4/5] Bestanden wegschrijven ...")
    write_csv(outdir / "fringe_data.csv", t, state, metric)
    write_overview_png(outdir / "interferometer_status.png", t, state, n_cycles)
    write_interactive_html(
        Path(TEMPLATE_PATH), outdir / "interferometer_status.html",
        t, state, metric, video_path.name, info, X, Y,
    )

    print("[5/5] Klaar. Resultaten in:", outdir.resolve())
    print(f"      - {outdir/'fringe_data.csv'}")
    print(f"      - {outdir/'interferometer_status.png'}")
    print(f"      - {outdir/'interferometer_status.html'}")


if __name__ == "__main__":
    main()
