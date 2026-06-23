"""
homologar_dataset.py
Fusiona el dataset de Joel (JPEG, cmd int) con el de Alberto (PNG en ZIPs, cmd string).
Salida: data/dataset_merged.csv + imágenes unificadas en data/images/
"""

import os, csv, zipfile, io
import numpy as np
import cv2

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(BASE, "data")
IMGS  = os.path.join(DATA, "images")
OUT   = os.path.join(DATA, "dataset_merged.csv")

JOEL_CSV    = os.path.join(DATA, "dataset.csv")
ALBERTO_CSV = "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_9/ActFinal/ProyectoFinal_alberto/Capturas/dataset_mundo1.csv"
ALBERTO_ZIPS = [
    "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_9/ActFinal/ProyectoFinal_alberto/Capturas/capturas_parte_1.zip",
    "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_9/ActFinal/ProyectoFinal_alberto/Capturas/capturas_parte_2.zip",
    "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_9/ActFinal/ProyectoFinal_alberto/Capturas/capturas_parte_3.zip",
    "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_9/ActFinal/ProyectoFinal_alberto/Capturas/capturas_parte_4.zip",
]

# comando de Alberto → int (mismo esquema que Joel)
# "straight" = seguir ruta = CMD_CONTINUE (0); no hay distinción cruce/recto en su dataset
ALBERTO_CMD = {"straight": 0, "left": 2, "right": 3}

os.makedirs(IMGS, exist_ok=True)

rows_out = []

# ── 1. Dataset de Joel ────────────────────────────────────────────────────────
print("[1/2] Leyendo dataset de Joel...")
with open(JOEL_CSV) as f:
    for row in csv.DictReader(f):
        img_rel = row["image_path"]                      # data/images/img_XXXXXX.jpg
        img_abs = os.path.join(BASE, img_rel)
        if not os.path.exists(img_abs):
            continue
        rows_out.append({
            "image_path":    img_rel,
            "steering_angle": float(row["steering_angle"]),
            "speed_kmh":      float(row["speed_kmh"]),
            "nav_command":    int(row["nav_command"]),
            "source":         "joel",
        })

print(f"   Joel: {len(rows_out)} filas válidas")

# ── 2. Dataset de Alberto (extrae de ZIPs al vuelo) ───────────────────────────
print("[2/2] Extrayendo imágenes de Alberto desde ZIPs...")

# índice: nombre_archivo → ruta relativa en CSV
alberto_rows = {}
with open(ALBERTO_CSV) as f:
    for row in csv.DictReader(f):
        fname = os.path.basename(row["image_path"].replace("\\", "/"))
        alberto_rows[fname] = row

alberto_count = 0
next_idx = max(
    (int(f.split("_")[1].split(".")[0]) for f in os.listdir(IMGS) if f.startswith("img_")),
    default=-1
) + 1

for zip_path in ALBERTO_ZIPS:
    print(f"   Procesando {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name not in alberto_rows:
                continue
            row = alberto_rows[name]
            cmd_str = row["command"].strip().lower()
            if cmd_str not in ALBERTO_CMD:
                continue

            # leer PNG y guardar como JPEG
            data = zf.read(name)
            arr  = np.frombuffer(data, np.uint8)
            img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            out_name = f"img_{next_idx:06d}.jpg"
            out_path = os.path.join(IMGS, out_name)
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            rows_out.append({
                "image_path":    f"data/images/{out_name}",
                "steering_angle": float(row["steering_angle"]),
                "speed_kmh":      float(row["speed"]),
                "nav_command":    ALBERTO_CMD[cmd_str],
                "source":         "alberto",
            })
            next_idx      += 1
            alberto_count += 1

    print(f"   {alberto_count} imágenes de Alberto procesadas hasta ahora")

# ── 3. Escribir CSV unificado ─────────────────────────────────────────────────
print(f"\nEscribiendo {OUT}...")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image_path","steering_angle","speed_kmh","nav_command","source"])
    w.writeheader()
    w.writerows(rows_out)

# ── 4. Resumen ────────────────────────────────────────────────────────────────
from collections import Counter
cmds = Counter(r["nav_command"] for r in rows_out)
srcs = Counter(r["source"] for r in rows_out)
total = len(rows_out)
labels = {0:"CONTINUE", 1:"RECTO", 2:"IZQUIERDA", 3:"DERECHA"}

print("\n════════════════════════════════")
print(f"  TOTAL: {total:,} imágenes")
print(f"  Joel:    {srcs['joel']:,}")
print(f"  Alberto: {srcs['alberto']:,}")
print("────────────────────────────────")
for c in range(4):
    n = cmds[c]
    print(f"  {labels[c]:10s}: {n:5,} ({n/total*100:.1f}%)")
print("════════════════════════════════")
print(f"\nDataset guardado en: {OUT}")
