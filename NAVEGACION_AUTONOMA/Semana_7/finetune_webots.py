"""
Fine-tuning del modelo GTSRB con imágenes capturadas en Webots.

FLUJO:
1. En Webots, acércate a cada señal y presiona 'A' para capturar fotos.
2. Mueve cada foto a la carpeta correcta en webots_signs/:
     14_stop/         → StopSign (octágono rojo)
     13_yield/        → YieldSign (triángulo)
     18_precaucion/   → CautionSign genérico (diamante amarillo)
     19_curva_izq/    → CautionSign turn_left
     20_curva_der/    → CautionSign turn_right
     22_bache/        → CautionSign bump
     17_prohibido/    → OrderSign no_right_turn
     15_sin_vehiculos/→ OrderSign no_pedestrian
     3_lim60/         → SpeedLimitSign 55 mph
     4_lim70/         → SpeedLimitSign 65 mph
     34_girar_izq/    → one_way_sign_left
     35_adelante/     → HighwaySign
3. Ejecuta este script: python finetune_webots.py
4. El modelo actualizado se guarda como modelo_gtsrb_webots.keras
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
SIGNS_DIR   = os.path.join(BASE_DIR, "webots_signs")
MODEL_IN    = os.path.join(BASE_DIR, "modelo_gtsrb.keras")
MODEL_OUT   = os.path.join(BASE_DIR, "modelo_gtsrb_webots.keras")
IMG_SIZE    = 32
EPOCHS      = 10
LR          = 0.0001   # lr bajo para fine-tuning (no destruir pesos ya aprendidos)

# Mapeo carpeta → ClassId GTSRB
FOLDER_TO_CLASS = {
    "14_stop":          14,
    "13_yield":         13,
    "18_precaucion":    18,
    "19_curva_izq":     19,
    "20_curva_der":     20,
    "22_bache":         22,
    "17_prohibido":     17,
    "15_sin_vehiculos": 15,
    "3_lim60":           3,
    "4_lim70":           4,
    "34_girar_izq":     34,
    "35_adelante":      35,
}

# ── Cargar imágenes de Webots ──────────────────────────────────────────────────
def cargar_webots(signs_dir):
    X, y = [], []
    for carpeta, clase in FOLDER_TO_CLASS.items():
        ruta = os.path.join(signs_dir, carpeta)
        if not os.path.exists(ruta):
            continue
        archivos = [f for f in os.listdir(ruta)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"  [{clase:2d}] {carpeta}: {len(archivos)} imágenes")
        for nombre in archivos:
            img = cv2.imread(os.path.join(ruta, nombre))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.astype(np.float32) / 255.0)
            y.append(clase)
    return np.array(X), np.array(y, dtype=np.int64)

print("Cargando imágenes de Webots...")
X_wb, y_wb = cargar_webots(SIGNS_DIR)

if len(X_wb) == 0:
    print("\n⚠ No hay imágenes en webots_signs/. Captura señales en Webots con la tecla 'A'")
    print("  y coloca cada foto en la carpeta correspondiente.")
    exit()

print(f"\nTotal imágenes Webots: {len(X_wb)}")
print(f"Clases presentes: {sorted(set(y_wb))}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_wb, y_wb, test_size=0.20, random_state=42,
    stratify=y_wb if len(set(y_wb)) > 1 else None
)

# ── Cargar modelo base y congelar capas convolucionales ───────────────────────
print("\nCargando modelo GTSRB base...")
model = load_model(MODEL_IN)
model.summary()

# Congelar todo excepto las dos últimas capas Dense
# (las Conv ya aprendieron a detectar bordes/formas — las reutilizamos)
n_trainable = 0
for layer in model.layers:
    if layer.name in ("dense", "dense_1", "batch_normalization_6",
                      "dropout_4", "flatten"):
        layer.trainable = True
        n_trainable += 1
    elif "dense" in layer.name.lower() or "flatten" in layer.name.lower():
        layer.trainable = True
        n_trainable += 1
    else:
        layer.trainable = False

print(f"\nCapas entrenables: {n_trainable}")
trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
total     = sum(np.prod(v.shape) for v in model.variables)
print(f"Parámetros entrenables: {trainable:,} / {total:,}")

# ── Fine-tuning ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n─── Fine-tuning ───────────────────────────────────────────────────")
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=max(8, len(X_tr) // 4),   # batch pequeño para datasets chicos
    verbose=1
)

# ── Guardar modelo actualizado ─────────────────────────────────────────────────
model.save(MODEL_OUT)
print(f"\n✓ Modelo guardado: {MODEL_OUT}")
print("\nPara usar en Webots, cambia MODEL_PATH en simple_controller_H2_CNN.py:")
print(f'  MODEL_PATH = r"{MODEL_OUT}"')
