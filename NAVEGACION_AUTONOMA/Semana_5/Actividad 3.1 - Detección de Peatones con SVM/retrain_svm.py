"""
Reentrenamiento del SVM mezclando dataset INRIA original + capturas Webots.

Uso:
    python retrain_svm.py

Requiere haber ejecutado collect_training_data.py en Webots primero
para generar la carpeta training_data/.
"""
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import cv2
import joblib
import os
import glob

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ORIG_NEG_DIR = os.path.join(SCRIPT_DIR, "human detection dataset", "0")
ORIG_POS_DIR = os.path.join(SCRIPT_DIR, "human detection dataset", "1")
NEW_POS_DIR  = os.path.join(SCRIPT_DIR, "training_data", "positive")
NEW_NEG_DIR  = os.path.join(SCRIPT_DIR, "training_data", "negative")
OUT_MODEL    = os.path.join(SCRIPT_DIR, "pedestrian_svm.joblib")

HOG_PARAMS = dict(
    orientations=11,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    transform_sqrt=False,
    feature_vector=True,
)


def load_folder(folder, label):
    paths = glob.glob(os.path.join(folder, "*.png")) + \
            glob.glob(os.path.join(folder, "*.jpg"))
    imgs, labels = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 128))
        imgs.append(img)
        labels.append(label)
    return imgs, labels


def extract_hog(imgs):
    return np.array([hog(img, **HOG_PARAMS) for img in imgs])


print("Cargando dataset INRIA original…")
pos_imgs, pos_labels = load_folder(ORIG_POS_DIR, 1)
neg_imgs, neg_labels = load_folder(ORIG_NEG_DIR, 0)
print(f"  INRIA  — positivos: {len(pos_imgs)}  negativos: {len(neg_imgs)}")

print("Cargando capturas Webots…")
w_pos_imgs, w_pos_labels = load_folder(NEW_POS_DIR, 1) if os.path.isdir(NEW_POS_DIR) else ([], [])
w_neg_imgs, w_neg_labels = load_folder(NEW_NEG_DIR, 0) if os.path.isdir(NEW_NEG_DIR) else ([], [])
print(f"  Webots — positivos: {len(w_pos_imgs)}  negativos: {len(w_neg_imgs)}")

if len(w_pos_imgs) == 0:
    print("\n[AVISO] No se encontraron capturas Webots.")
    print("  Abre collect_data.wbt en Webots y ejecuta el controlador collect_training_data.")
    print("  Luego vuelve a correr este script.\n")

all_pos = pos_imgs + w_pos_imgs
all_neg = neg_imgs + w_neg_imgs
print(f"\nTotal — positivos: {len(all_pos)}  negativos: {len(all_neg)}")

print("Extrayendo HOG…")
X_pos = extract_hog(all_pos)
X_neg = extract_hog(all_neg)
X     = np.vstack([X_pos, X_neg])
y     = np.array([1] * len(all_pos) + [0] * len(all_neg))
print(f"  Shape: {X.shape}")

print("Entrenando SVM (rbf)…")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=500)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
print(f"  CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

model.fit(X, y)
joblib.dump(model, OUT_MODEL)
print(f"\n[OK] Modelo guardado en {OUT_MODEL}")
print("Reinicia Webots con city_2025a_v2.wbt para probar.")
