"""
clean_dataset.py
================
Limpia y balancea el dataset_merged.csv en dos pasos:

Paso 1 — Filtrado de ruido en CMD_IZQUIERDA / CMD_DERECHA:
  Si el conductor presionó 'a'/'d' antes de llegar al cruce, los frames
  tienen steering≈0 pero cmd=IZQ/DER. Esos frames confunden al modelo
  (aprende "al dar vuelta, el volante va recto"). Se eliminan.

Paso 2 — Reducción de CMD_CONTINUE con steer≈0:
  El 68% de CONTINUE tiene steering≈0 (carretera recta). Se sub-muestrea
  para que el modelo no colapse prediciendo siempre 0.

Resultado: dataset_clean.csv
"""

import csv, random, os
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN   = os.path.join(BASE, "data", "dataset_merged.csv")
OUT  = os.path.join(BASE, "data", "dataset_clean.csv")

# ── Parámetros ────────────────────────────────────────────────────────────────
# Paso 1: umbral de steering para IZQ/DER
IZQ_MIN  = -0.04   # IZQUIERDA debe tener steer < IZQ_MIN  (al menos 0.04 rad a la izq)
DER_MIN  =  0.04   # DERECHA    debe tener steer > DER_MIN  (al menos 0.04 rad a la der)

# Paso 2: sub-muestreo de CONTINUE con steer≈0
CONT_ZERO_THRESH = 0.01   # |steer| < esto = "recto"
CONT_ZERO_KEEP   = 0.28   # conservar 28% de rectos en CONTINUE

random.seed(42)

# ── Lectura ───────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(IN)))
print(f"Dataset entrada: {len(rows):,} filas")

labels = {0:"CONTINUE", 1:"RECTO", 2:"IZQUIERDA", 3:"DERECHA"}

# ── Paso 1: filtrar IZQ / DER con steering incorrecto ────────────────────────
cleaned = []
removed_izq = 0
removed_der = 0

for r in rows:
    cmd   = int(r["nav_command"])
    steer = float(r["steering_angle"])

    if cmd == 2 and steer >= IZQ_MIN:   # IZQUIERDA pero no gira a la izquierda
        removed_izq += 1
        continue
    if cmd == 3 and steer <= DER_MIN:   # DERECHA pero no gira a la derecha
        removed_der += 1
        continue
    cleaned.append(r)

print(f"\nPaso 1 — Filtro de ruido IZQ/DER (umbral ±{DER_MIN:.2f} rad):")
print(f"  IZQUIERDA eliminados : {removed_izq:,}")
print(f"  DERECHA eliminados   : {removed_der:,}")

# ── Paso 2: sub-muestrear rectos en CONTINUE ─────────────────────────────────
balanced = []
removed_cont = 0

for r in cleaned:
    cmd   = int(r["nav_command"])
    steer = abs(float(r["steering_angle"]))

    if cmd == 0 and steer < CONT_ZERO_THRESH:
        if random.random() < CONT_ZERO_KEEP:
            balanced.append(r)
        else:
            removed_cont += 1
    else:
        balanced.append(r)

print(f"\nPaso 2 — Sub-muestreo CONTINUE recto (mantener {CONT_ZERO_KEEP*100:.0f}%):")
print(f"  CONTINUE eliminados  : {removed_cont:,}")

# ── Resultado ─────────────────────────────────────────────────────────────────
random.shuffle(balanced)

cmds  = Counter(int(r["nav_command"]) for r in balanced)
total = len(balanced)
zeros = sum(1 for r in balanced if abs(float(r["steering_angle"])) < 0.01)

print(f"\n=== dataset_clean.csv ===")
print(f"Total: {total:,} filas  (eliminados {len(rows)-total:,})")
for c in range(4):
    n   = cmds[c]
    bar = "█" * int(n / total * 40)
    print(f"  {labels[c]:10s}: {n:5,} ({n/total*100:5.1f}%) {bar}")
print(f"  steer≈0 global : {zeros:,} ({zeros/total*100:.1f}%)")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(balanced)

print(f"\nGuardado: {OUT}")
