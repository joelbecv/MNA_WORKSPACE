"""
Fine-tuning con crops reales de Webots.
Genera 80 variantes de cada crop real (vs 200 sintéticos por clase)
para que el modelo aprenda la apariencia exacta de las señales en simulación.
"""
import os, random, glob, json, numpy as np, cv2
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

BASE     = "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_7"
AUG_DIR  = os.path.join(BASE, "webots_signs_aug")
REAL_DIR = os.path.join(BASE, "webots_crops_real")
MODEL_IN = os.path.join(BASE, "modelo_us_webots.keras")
MODEL_OUT= os.path.join(BASE, "modelo_us_webots.keras")
MAP_PATH = os.path.join(BASE, "us_class_map.json")
IMG_SIZE = 32

SIGN_MAP = {
    14: "STOP",        13: "Ceder_paso",   19: "Curva_izq",
    20: "Curva_der",   22: "Bache",        11: "Cruce",
    17: "No_girar_der",15: "No_peatones",   3: "Lim_60",
     4: "Lim_70",      34: "Un_sentido",
}
IDX_TO_CLASS = {0:3,1:4,2:11,3:13,4:14,5:15,6:17,7:19,8:20,9:22,10:34}
CLASS_TO_IDX = {v:k for k,v in IDX_TO_CLASS.items()}

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))

def augment_real(img_bgr, n=80):
    """Genera n variantes de un crop real: CLAHE, brillo, ruido, flip."""
    results = []
    for _ in range(n):
        img = img_bgr.copy()
        # Brillo aleatorio
        alpha = random.uniform(0.5, 1.4)
        beta  = random.uniform(-30, 30)
        img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
        # Ruido
        noise = np.random.normal(0, random.uniform(0, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # CLAHE siempre (los crops reales son oscuros)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Flip horizontal ocasional
        if random.random() < 0.3:
            img = cv2.flip(img, 1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
    return results

# ── Carga dataset sintético ──────────────────────────────────────────────
X_all, y_all = [], []
for clase, nombre in SIGN_MAP.items():
    out_dir = os.path.join(AUG_DIR, f"{clase}_{nombre}")
    if not os.path.exists(out_dir): continue
    for p in glob.glob(os.path.join(out_dir, "*.png")):
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.cvtColor(cv2.resize(img,(IMG_SIZE,IMG_SIZE)), cv2.COLOR_BGR2RGB)
        X_all.append(img.astype(np.float32)/255.0)
        y_all.append(clase)

synt_count = len(X_all)
print(f"Sintético: {synt_count} imágenes")

# ── Carga crops reales (80 variantes cada uno) ───────────────────────────
real_added = {}
for clase, nombre in SIGN_MAP.items():
    real_folder = os.path.join(REAL_DIR, f"{clase}_{nombre}")
    if not os.path.exists(real_folder): continue
    crops = glob.glob(os.path.join(real_folder, "*.png")) + \
            glob.glob(os.path.join(real_folder, "*.jpg"))
    if not crops: continue
    n_added = 0
    for p in crops:
        img = cv2.imread(p)
        if img is None: continue
        variantes = augment_real(img, n=80)
        X_all.extend(variantes)
        y_all.extend([clase]*len(variantes))
        n_added += len(variantes)
    real_added[clase] = n_added

print("Reales añadidos:")
for c, n in real_added.items():
    print(f"  [{c:2d}] {SIGN_MAP[c]:20s}: {n} variantes")

X_all = np.array(X_all, dtype=np.float32)
y_idx = np.array([CLASS_TO_IDX[c] for c in y_all], dtype=np.int64)

print(f"\nDataset total: {X_all.shape}")
for idx, clase in IDX_TO_CLASS.items():
    n = (y_idx==idx).sum()
    print(f"  [{clase:2d}] {SIGN_MAP[clase]:20s}: {n}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_idx, test_size=0.15, random_state=SEED, stratify=y_idx)
print(f"\nTrain: {X_tr.shape}  Val: {X_val.shape}")

# ── Carga modelo ─────────────────────────────────────────────────────────
print("\nCargando modelo base...")
model = load_model(MODEL_IN)

# Descongelar solo las capas del head
for layer in model.layers:
    layer.trainable = layer.name.startswith("head_")

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                   restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=4, min_lr=1e-7, verbose=1),
]

print("Entrenando...")
model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
          epochs=60, batch_size=32, callbacks=callbacks, verbose=1)

vl, va = model.evaluate(X_val, y_val, verbose=0)
print(f"\nVal Loss: {vl:.4f}  |  Val Acc: {va:.4f}")

# Verificación en los crops problemáticos
print("\n=== Verificación en crops reales ===")
for clase, nombre in [(19,"Curva_izq"), (3,"Lim_60")]:
    folder = os.path.join(REAL_DIR, f"{clase}_{nombre}")
    crops  = sorted(glob.glob(os.path.join(folder,"*.png")))[:5]
    print(f"\n[{clase}] {nombre}:")
    for p in crops:
        img = cv2.imread(p)
        if img is None: continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        n = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),(32,32)).astype(np.float32)/255.0
        probs = model.predict(np.expand_dims(n,0), verbose=0)[0]
        top2 = sorted([(IDX_TO_CLASS[i], float(probs[i])) for i in range(len(probs))], key=lambda x:-x[1])[:2]
        print(f"  {os.path.basename(p):30s} → g{top2[0][0]}({top2[0][1]:.2f})  g{top2[1][0]}({top2[1][1]:.2f})")

model.save(MODEL_OUT)
print(f"\nModelo guardado: {MODEL_OUT}")
