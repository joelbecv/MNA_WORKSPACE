"""
Benchmark del controlador — mide cuánto tarda cada operación por frame.
Corre sin Webots, usa imágenes sintéticas.
"""
import time
import numpy as np
import cv2
import joblib
import os
from skimage.feature import hog

# ── Parámetros (igual que el controlador) ─────────────────────────────────────
HOG_WIN_W  = 64
HOG_WIN_H  = 128
SLIDE_STEP = 32
IMG_W, IMG_H = 256, 128
N_RUNS = 50

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'pedestrian_svm.joblib')

# ── Imagen sintética ──────────────────────────────────────────────────────────
fake_bgr  = np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
fake_gray = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2GRAY)

def time_ms(fn, runs=N_RUNS):
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    elapsed = (time.perf_counter() - start) / runs * 1000
    return elapsed

print(f"Benchmark — {N_RUNS} iteraciones cada operación\n")

# 1. OpenCV pipeline
def op_cv():
    hsv  = cv2.cvtColor(fake_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15,80,80]), np.array([35,255,255]))
    c1   = cv2.Canny(fake_gray, 50, 150)
    c2   = cv2.Canny(mask, 50, 150)
    cv2.bitwise_or(c1, c2)

t = time_ms(op_cv)
print(f"OpenCV (HSV + Canny x2 + bitwise_or): {t:.2f} ms/frame")

# 2. Hough
edges = cv2.Canny(fake_gray, 50, 150)
def op_hough():
    cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=15)

t = time_ms(op_hough)
print(f"HoughLinesP:                           {t:.2f} ms/frame")

# 3. HOG una ventana
window = fake_gray[:HOG_WIN_H, :HOG_WIN_W]
def op_hog():
    hog(window, orientations=11, pixels_per_cell=(16,16),
        cells_per_block=(2,2), transform_sqrt=False, feature_vector=True)

t = time_ms(op_hog)
print(f"HOG (1 ventana 64x128):                {t:.2f} ms/frame")

# 4. HOG todas las ventanas (con ROI x)
x_start = int(IMG_W * 0.20)
x_end   = int(IMG_W * 0.80)
x_positions = list(range(x_start, min(x_end, IMG_W - HOG_WIN_W + 1), SLIDE_STEP))
n_windows = len(x_positions)

def op_hog_all():
    for x in x_positions:
        win = fake_bgr[0:HOG_WIN_H, x:x+HOG_WIN_W]
        g   = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
        hog(g, orientations=11, pixels_per_cell=(16,16),
            cells_per_block=(2,2), transform_sqrt=False, feature_vector=True)

t = time_ms(op_hog_all)
print(f"HOG todas ventanas ({n_windows} ventanas):          {t:.2f} ms/frame")

# 5. SVM predict
if os.path.exists(os.path.normpath(MODEL_PATH)):
    svm = joblib.load(os.path.normpath(MODEL_PATH))
    feat = hog(window, orientations=11, pixels_per_cell=(16,16),
               cells_per_block=(2,2), transform_sqrt=False, feature_vector=True)
    def op_svm():
        svm.predict([feat])
    t = time_ms(op_svm)
    print(f"SVM predict (1 ventana):               {t:.2f} ms/frame")

    def op_svm_all():
        for x in x_positions:
            win = fake_bgr[0:HOG_WIN_H, x:x+HOG_WIN_W]
            g   = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
            f   = hog(g, orientations=11, pixels_per_cell=(16,16),
                      cells_per_block=(2,2), transform_sqrt=False, feature_vector=True)
            svm.predict([f])
    t = time_ms(op_svm_all)
    print(f"HOG+SVM todas ventanas ({n_windows} ventanas):    {t:.2f} ms/frame")
else:
    print("Modelo SVM no encontrado — omitiendo test SVM")

# 6. display_gray equivalente (numpy)
def op_display():
    np.dstack((fake_gray, fake_gray, fake_gray)).tobytes()

t = time_ms(op_display)
print(f"Preparar buffer display (tobytes):     {t:.2f} ms/frame")

print(f"\nTimestep de la simulación:             10.00 ms/frame")
print(f"Budget total del controller:           ~30 ms (cada 3 frames)")
