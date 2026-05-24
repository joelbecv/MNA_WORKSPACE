"""
Supervisor de recolección automática de datos de entrenamiento.
Abre collect_data.wbt en Webots y corre este controlador como "collector".

Guarda:
  training_data/positive/  — crops 64x128 del peatón Webots
  training_data/negative/  — patches 64x128 de fondo sin peatón
"""
from controller import Supervisor
import numpy as np
import cv2
import os

robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

W = camera.getWidth()
H = camera.getHeight()

ped       = robot.getFromDef("PED")
ped_trans = ped.getField("translation")
ped_rot   = ped.getField("rotation")

CTRL_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "..", "training_data"))
OUT_POS  = os.path.join(OUT_ROOT, "positive")
OUT_NEG  = os.path.join(OUT_ROOT, "negative")
os.makedirs(OUT_POS, exist_ok=True)
os.makedirs(OUT_NEG, exist_ok=True)

DISTANCES = [3.5, 4.5, 5.5, 6.5, 8.0, 10.0, 12.0]
LATERALS  = [-1.5, -0.75, 0.0, 0.75, 1.5]
ROT_RADS  = [i * 0.785 for i in range(8)]   # 0°, 45°, 90°… 315°

def settle(n=6):
    for _ in range(n):
        robot.step(timestep)

def capture_bgr():
    settle()
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((H, W, 4))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def hide_ped():
    ped_trans.setSFVec3f([200.0, 0.0, 200.0])
    settle(8)

# -----------------------------------------------------------------
# Paso 1: capturar frame de referencia sin peatón (fondo limpio)
# -----------------------------------------------------------------
hide_ped()
ref_bgr = capture_bgr()
ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

# Negativos del fondo de referencia
rng = np.random.default_rng(42)
count_neg = 0
for _ in range(200):
    rx = int(rng.integers(0, W - 64))
    ry = int(rng.integers(0, H - 128))
    patch = ref_bgr[ry:ry+128, rx:rx+64]
    cv2.imwrite(os.path.join(OUT_NEG, f"neg_{count_neg:04d}.png"), patch)
    count_neg += 1

print(f"[collect] {count_neg} negativos de referencia guardados")

# -----------------------------------------------------------------
# Paso 2: posicionar peatón y extraer crop por diferencia de imagen
# -----------------------------------------------------------------
count_pos = 0
total = len(DISTANCES) * len(LATERALS) * len(ROT_RADS)
print(f"[collect] Iniciando {total} posiciones positivas…")

for dist in DISTANCES:
    for lat in LATERALS:
        for rot in ROT_RADS:
            ped_trans.setSFVec3f([lat, 0.0, -dist])
            ped_rot.setSFRotation([0, 1, 0, rot])
            frame = capture_bgr()

            # Diferencia con fondo para localizar el peatón
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame_gray, ref_gray)
            _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)

            if cw < 8 or ch < 20:
                continue

            # Normalizar a proporción 1:2 (64×128)
            target_h = max(ch + 10, 40)
            target_w = target_h // 2

            cx_center = cx + cw // 2
            cy_center = cy + ch // 2

            x1 = max(cx_center - target_w // 2, 0)
            x2 = min(cx_center + target_w // 2, W)
            y1 = max(cy_center - target_h // 2, 0)
            y2 = min(cy_center + target_h // 2, H)

            if (x2 - x1) < 8 or (y2 - y1) < 20:
                continue

            crop = cv2.resize(frame[y1:y2, x1:x2], (64, 128))
            cv2.imwrite(os.path.join(OUT_POS, f"pos_{count_pos:04d}.png"), crop)
            count_pos += 1

            # Negativo del mismo frame (zona opuesta al peatón)
            neg_x = W - cx_center
            neg_x1 = max(neg_x - 32, 0)
            neg_x2 = min(neg_x + 32, W)
            if (neg_x2 - neg_x1) == 64:
                patch = cv2.resize(frame[y1:y2, neg_x1:neg_x2], (64, 128))
                cv2.imwrite(os.path.join(OUT_NEG, f"neg_{count_neg:04d}.png"), patch)
                count_neg += 1

print(f"[collect] {count_pos} positivos  |  {count_neg} negativos")
print(f"[collect] Guardado en {OUT_ROOT}")
print("[collect] Ahora ejecuta retrain_svm.py")

robot.simulationQuit(0)
