"""
balance_dataset.py
Genera dataset_balanced.csv reduciendo el dominio de steering≈0 en CMD_CONTINUE.
"""
import csv, random, os
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN   = os.path.join(BASE, "data", "dataset_merged.csv")
OUT  = os.path.join(BASE, "data", "dataset_balanced.csv")

ZERO_THRESH      = 0.01   # |steering| < ZERO_THRESH → "ejemplo recto"
ZERO_KEEP_RATIO  = 0.25   # conservar solo el 25% de rectos en CMD_CONTINUE

random.seed(42)
rows = list(csv.DictReader(open(IN)))
print(f"Dataset original: {len(rows):,} filas")

balanced = []
cmd_zero_kept = 0
cmd_zero_drop = 0

for r in rows:
    steer = abs(float(r["steering_angle"]))
    cmd   = int(r["nav_command"])

    # Solo filtrar CMD_CONTINUE (0) con steering≈0
    if cmd == 0 and steer < ZERO_THRESH:
        if random.random() < ZERO_KEEP_RATIO:
            balanced.append(r)
            cmd_zero_kept += 1
        else:
            cmd_zero_drop += 1
    else:
        balanced.append(r)

random.shuffle(balanced)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(balanced)

labels = {0:"CONTINUE", 1:"RECTO", 2:"IZQUIERDA", 3:"DERECHA"}
cmds   = Counter(int(r["nav_command"]) for r in balanced)
total  = len(balanced)

print(f"\nDataset balanceado: {total:,} filas  (eliminados {cmd_zero_drop:,} rectos de CONTINUE)")
for c in range(4):
    n = cmds[c]
    print(f"  {labels[c]:10s}: {n:6,} ({n/total*100:.1f}%)")

z = sum(1 for r in balanced if abs(float(r["steering_angle"])) < ZERO_THRESH)
print(f"\nSteering≈0 total: {z:,} ({z/total*100:.1f}%)  (antes era 42%)")
print(f"Guardado en: {OUT}")
