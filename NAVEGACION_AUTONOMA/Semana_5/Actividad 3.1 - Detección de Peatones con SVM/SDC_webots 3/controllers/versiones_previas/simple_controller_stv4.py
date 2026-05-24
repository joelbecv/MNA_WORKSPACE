# =============================================================================
# simple_controller_stv4.py
# Actividad 3.1 — Sistema de Detección de Peatones para Vehículo Autónomo
# Navegación Autónoma — Maestría en Inteligencia Artificial
# =============================================================================
#
# NARRATIVA DEL SISTEMA
# ─────────────────────
# Este controlador implementa un sistema de seguridad activa para un vehículo
# autónomo BMW circulando en entorno urbano. El auto enfrenta el reto central
# de la conducción autónoma: moverse eficientemente sin comprometer la seguridad
# de los peatones.
#
# El sistema resuelve esto con TRES capas de inteligencia trabajando en paralelo:
#
#   CAPA 1 — NAVEGACIÓN (PID)
#   El auto sigue la línea amarilla del carril usando visión por computadora.
#   Detecta los bordes de la línea, calcula su posición y corrige el volante
#   en tiempo real con un controlador PID. El auto circula a 30 km/h.
#
#   CAPA 2 — DETECCIÓN DE PERSONAS (SVM + HOG)
#   Una ventana deslizante recorre la imagen de la cámara buscando siluetas
#   humanas. Cada ventana se convierte en un descriptor HOG (924 valores que
#   capturan los gradientes de la imagen) y se clasifica con una SVM entrenada
#   con el dataset INRIA de personas reales. Si 2 scans consecutivos confirman
#   un peatón → freno de emergencia.
#
#   CAPA 3 — DETECCIÓN DE OBSTÁCULOS (LiDAR)
#   Un sensor LiDAR Sick LMS 291 escanea 180° frente al auto con rayos láser.
#   Medimos el cono central (±30°) para detectar cualquier objeto físico a
#   menos de 8m. El LiDAR actúa más rápido que el SVM — no necesita "ver"
#   al peatón, solo detectar que algo está en el camino.
#
# LÓGICA DE SEGURIDAD (por capas)
# ────────────────────────────────
#   < 5m  → LiDAR fuerza freno SIEMPRE (override total)
#   < 8m  → LiDAR frena si SVM no ha actuado aún
#   SVM x2 → Confirmación visual de peatón → freno de emergencia
#
# RESULTADO
# ─────────
# El LiDAR detiene el auto rápidamente ante cualquier obstáculo físico.
# El SVM identifica específicamente si ese obstáculo es una persona.
# Juntos reducen tanto los falsos positivos (frenadas innecesarias)
# como los falsos negativos (no detectar un peatón real).
#
# REFERENCIAS
#   - INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
#   - Dalal & Triggs, "HOG for Human Detection", CVPR 2005
#   - Webots Driver API: https://cyberbotics.com/doc/automobile/driver-library
# =============================================================================

from controller import Display, Keyboard
from vehicle import Car, Driver
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import math
import os
import time
import threading
import queue
from datetime import datetime

import subprocess as _sp

_music_proc  = None
_music_on    = False
_MUSIC_FILE  = '/Users/joelbecerril/Downloads/Let It Ride - Anno Domini Beats.mp3'
_MUSIC_VOL   = '0.5'

def _start_music():
    global _music_proc, _music_on
    def _loop():
        global _music_proc, _music_on
        while _music_on:
            _music_proc = _sp.Popen(
                ['afplay', '-v', _MUSIC_VOL, _MUSIC_FILE],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL
            )
            _music_proc.wait()
    threading.Thread(target=_loop, daemon=True).start()

def toggle_music():
    global _music_on, _music_proc
    try:
        if _music_on:
            _music_on = False
            try:
                if _music_proc and _music_proc.poll() is None:
                    _music_proc.terminate()
            except Exception:
                pass
            print("[Audio] Música pausada — presiona B para reanudar")
        else:
            _music_on = True
            _start_music()
            print("[Audio] Música reanudada — presiona B para pausar")
    except Exception as e:
        print(f"[Audio] Error toggle: {e}")

_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))


# =============================================================================
# PARÁMETROS DEL SISTEMA
# =============================================================================

# ── Navegación PID ───────────────────────────────────────────────────────────
CRUISE_SPEED   = 30       # velocidad de crucero en km/h
MAX_ANGLE      = 0.5      # ángulo máximo de giro del volante (radianes)
MAX_STEER_RATE = 0.03     # suavizado: máximo cambio de ángulo por frame
DEBOUNCE_TIME  = 0.1      # tiempo mínimo entre teclas (segundos)

# Ganancias PID: Kp corrige el error actual, Ki acumula errores persistentes,
# Kd amortigua oscilaciones. Calibradas empíricamente para este mundo.
Kp, Ki, Kd     = 0.28, 0.01, 0.01

# Líneas con pendiente < 0.6 son casi horizontales (franjas de cruce peatonal)
# y se descartan — solo interesan las líneas diagonales del carril.
MIN_ABS_SLOPE  = 0.4

# Rango HSV del color amarillo de las líneas del carril
YELLOW_LOW     = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH    = np.array([35, 255, 255], dtype=np.uint8)

# ── LiDAR Sick LMS 291 — Detección dual zona ─────────────────────────────────
# ZONA FRONTAL: cono estrecho (61°) a distancia larga (8m) — detecta lo que
#   viene de frente con tiempo suficiente para frenar.
#   Alcance lateral a 8m: 8 × tan(30.5°) = 4.7m
#
# ZONA LATERAL: cono amplio (120°) a distancia corta (5m) — detecta peatones
#   que cruzan perpendicularmente o regresan desde los costados.
#   Alcance lateral a 5m: 5 × tan(60°) = 8.7m
#   La distancia corta evita falsas activaciones por edificios y señales lejanas.
LIDAR_CONE_FRONT  = 61.0  # cono frontal: ángulo total (°)
LIDAR_CONE_SIDE   = 120.0 # cono lateral: ángulo total (°)
LIDAR_MAX_M       = 6.0   # distancia de alerta zona frontal (metros)
LIDAR_SIDE_M      = 1.5   # distancia de alerta zona lateral (metros)
LIDAR_EMERGENCY_M = 6.0   # distancia de freno inmediato sin confirmación
LIDAR_OVERRIDE_M  = 4.0   # distancia de override: frena aunque SVM ya actuó
LIDAR_EVERY       = 1     # leer LiDAR cada frame = cada ~10ms
LIDAR_CONFIRM     = 1     # basta 1 lectura para activar alerta

# ── Detector SVM + HOG ───────────────────────────────────────────────────────
# La ventana HOG estándar de Dalal & Triggs es 64×128 píxeles.
# Como el ROI de la cámara es pequeño (~34px de alto), escalamos 4×
# para alcanzar el mínimo requerido: 34 × 4 = 136px > 128px.
HOG_WIN_W     = 64        # ancho de ventana HOG (píxeles)
HOG_WIN_H     = 128       # alto de ventana HOG (píxeles)
SLIDE_STEP    = 32        # paso del sliding window (píxeles)
DETECT_EVERY  = 10        # ejecutar SVM cada 10 frames = cada ~100ms
DISPLAY_EVERY = 3         # actualizar display cada 3 frames = cada ~30ms
CONFIRM_N     = 2         # scans positivos consecutivos para confirmar peatón
RELEASE_N     = 2         # scans negativos consecutivos para liberar el freno
HOLD_FRAMES   = 30        # frames mínimos de freno garantizado (~0.3 segundos)
MIN_HITS      = 1         # ventanas positivas mínimas por pasada
SVM_THRESHOLD = 0.25      # umbral de decision_function(): calibrado para Webots
                          # (INRIA da scores >1.0; Webots da 0.06-0.70 por domain gap)


# =============================================================================
# MÓDULO 1 — NAVEGACIÓN: SEGUIMIENTO DE CARRIL
# =============================================================================
# Filosofía: el auto solo "ve" el color amarillo de las líneas.
# Ignorar el resto de la imagen evita que peatones, cruces o sombras
# interfieran con la dirección del volante.

def get_image(camera):
    """Convierte la imagen BGRA de Webots a array NumPy."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_color(display, bgr_img):
    """Envía una imagen BGR al display de Webots (convierte a RGB internamente)."""
    rgb = bgr_img[:, :, ::-1].copy()
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def apply_roi(edges, h, w):
    """
    Aplica una máscara trapezoidal sobre los bordes detectados.
    Solo conserva la zona donde aparecen las líneas del carril:
    desde el frente del capó hasta el horizonte próximo.
    La forma trapezoidal imita la perspectiva real de la carretera.
    """
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),                   # esquina inf-izq
        (int(w * 0.35), int(h * 0.60)),       # esquina sup-izq
        (int(w * 0.65), int(h * 0.60)),       # esquina sup-der
        (int(w * 0.90), h),                   # esquina inf-der
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines(lines):
    """
    Descarta líneas casi horizontales (franjas de cruce peatonal).
    Solo acepta líneas con pendiente ≥ MIN_ABS_SLOPE (0.6),
    que corresponden a las líneas diagonales reales del carril.
    """
    if lines is None:
        return None
    ok = [l for l in lines
          if l[0][2] != l[0][0]
          and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(ok) if ok else None

def compute_center(lines):
    """
    Calcula el centro del carril a partir de las líneas Hough detectadas.
    Separa líneas por pendiente: negativa=izquierda, positiva=derecha.
    Si hay líneas de ambos lados, el centro es el promedio de sus medios.
    Si solo hay un lado, usa el promedio de todas las líneas disponibles.
    """
    if lines is None:
        return None
    lx, rx, ax = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid   = (x1 + x2) / 2
        ax.append(mid)
        (lx if slope < 0 else rx).append(mid)
    if lx and rx:
        return (np.mean(lx) + np.mean(rx)) / 2.0
    return np.mean(ax) if ax else None


# =============================================================================
# MÓDULO 2A — DETECCIÓN DE PERSONAS: SVM + HOG
# =============================================================================
# Filosofía: la cámara ve una franja horizontal de la escena (59%-85% del alto).
# Esa franja se amplía 4× para que las figuras humanas tengan el tamaño mínimo
# requerido por HOG. Una ventana de 64×128 recorre la imagen; cada posición
# genera 924 valores numéricos que describen la distribución de bordes
# (gradientes). La SVM decide si ese patrón se parece a una persona.

def svm_detect(bgr, model):
    """
    Ejecuta el sliding window HOG sobre el ROI de la cámara.

    Retorna: (detected, hits, max_score, windows, best_box)
      - detected:   True si al menos MIN_HITS ventanas superaron SVM_THRESHOLD
      - hits:       número de ventanas positivas en esta pasada
      - max_score:  score máximo registrado (útil para calibración)
      - windows:    total de ventanas evaluadas
      - best_box:   coordenadas (x1,y1,x2,y2) de la ventana con score máximo
    """
    h, w      = bgr.shape[:2]

    # ── Paso 1: recortar el ROI vertical ─────────────────────────────────────
    # Solo analizamos la banda donde aparecen peatones a distancia media.
    # Por encima (0%-59%): cielo y edificios → no hay peatones reales en el carril.
    # Por debajo (85%-100%): capó del auto → zona ciega de la cámara.
    y_off = int(h * 0.59)
    roi   = bgr[y_off : int(h * 0.85), :]

    total     = 0
    max_score = -999.0
    windows   = 0
    best_box  = None

    for scale in (4.0,):
        # ── Paso 2: escalar el ROI ────────────────────────────────────────────
        # El ROI tiene ~34px de alto. HOG necesita mínimo 128px.
        # Escala 4× → 34 × 4 = 136px. Suficiente para una ventana HOG completa.
        scaled = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)))
        sh, sw = scaled.shape[:2]
        if sh < HOG_WIN_H or sw < HOG_WIN_W:
            continue

        # ── Paso 3: limitar el barrido horizontal ─────────────────────────────
        # Solo recorremos el 30%-70% horizontal para evitar las banquetas.
        # Los peatones peligrosos están en el carril, no en las orillas.
        x0    = int(sw * 0.30)
        x1    = int(sw * 0.70)
        ystep = max(16, HOG_WIN_H // 4)

        for y in range(0, sh - HOG_WIN_H + 1, ystep):
            for x in range(x0, min(x1, sw - HOG_WIN_W + 1), SLIDE_STEP):

                # ── Paso 4: pre-filtro de color amarillo ──────────────────────
                # Si la ventana contiene >15% de píxeles amarillos, es una línea
                # del carril o un cruce peatonal — no puede ser una persona.
                # Saltamos sin correr el costoso HOG.
                win_bgr = scaled[y:y + HOG_WIN_H, x:x + HOG_WIN_W]
                win_hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
                if cv2.inRange(win_hsv, YELLOW_LOW, YELLOW_HIGH).mean() > 15:
                    continue

                # ── Paso 5: calcular descriptor HOG ──────────────────────────
                # HOG divide la ventana en celdas de 16×16px y calcula
                # histogramas de orientación de gradientes en 11 direcciones.
                # Con bloques 2×2 celdas → 924 valores que describen la forma.
                win  = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)
                feat = hog(win, orientations=11, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), transform_sqrt=False,
                           feature_vector=True)

                # ── Paso 6: clasificar con SVM ────────────────────────────────
                # decision_function() retorna la distancia al hiperplano de
                # separación. Positivo = lado "persona", negativo = lado "fondo".
                # Umbral calibrado en 0.25 (el modelo INRIA da scores más bajos
                # en Webots por diferencia de dominio visual).
                score   = model.decision_function([feat])[0]
                windows += 1
                if score > max_score:
                    max_score = score
                    bx1 = max(0, int(x / scale))
                    by1 = max(0, int(y / scale) + y_off)
                    bx2 = min(w, int((x + HOG_WIN_W) / scale))
                    by2 = min(h, int((y + HOG_WIN_H) / scale) + y_off)
                    best_box = (bx1, by1, bx2, by2)
                if score >= SVM_THRESHOLD:
                    total += 1

    detected = total >= MIN_HITS
    return detected, total, max_score, windows, best_box


# =============================================================================
# MÓDULO 2B — DETECCIÓN DE OBSTÁCULOS: LIDAR (dual zona)
# =============================================================================
# Filosofía: el LiDAR no identifica QUÉ está en el camino, solo DÓNDE.
# Dos zonas complementarias cubren tanto lo que viene de frente como
# peatones que cruzan perpendicularmente o regresan desde los costados.

def _cone_min(ranges, center, n_rays, fov_rad, cone_deg):
    """Retorna la distancia mínima válida dentro de un cono centrado."""
    half = max(1, int(n_rays * (math.radians(cone_deg) / fov_rad) / 2))
    vals = [r for r in ranges[center - half : center + half]
            if not (math.isnan(r) or math.isinf(r))]
    return min(vals) if vals else None

def lidar_read(lidar, fov_rad, n_rays):
    """
    Evalúa dos zonas de detección y retorna la amenaza más cercana.

    ZONA FRONTAL (61°/8m): detecta objetos que vienen de frente con
    tiempo suficiente para frenar a 30 km/h.

    ZONA LATERAL (120°/5m): detecta peatones perpendiculares o que
    regresan desde los costados del cruce. Distancia corta para evitar
    falsas activaciones con edificios y señales de las banquetas.

    Retorna: (alert, dist_m, dist_str)
      - alert:    True si alguna zona detecta algo dentro de su umbral
      - dist_m:   distancia mínima global detectada
      - dist_str: texto para display ("X.Xm [F]" / "X.Xm [L]" / "---")
    """
    ranges = lidar.getRangeImage()
    if not ranges:
        return False, None, '---'

    center = n_rays // 2

    d_front = _cone_min(ranges, center, n_rays, fov_rad, LIDAR_CONE_FRONT)
    d_side  = _cone_min(ranges, center, n_rays, fov_rad, LIDAR_CONE_SIDE)

    alert_front = d_front is not None and d_front < LIDAR_MAX_M
    alert_side  = d_side  is not None and d_side  < LIDAR_SIDE_M

    if alert_front and alert_side:
        d = min(d_front, d_side)
        label = 'F+L'
    elif alert_front:
        d, label = d_front, 'F'
    elif alert_side:
        d, label = d_side, 'L'
    else:
        # Sin alerta — reportar la distancia mínima global para el display
        candidates = [v for v in [d_front, d_side] if v is not None]
        d = min(candidates) if candidates else None
        return False, d, (f"{d:.1f}m" if d else '---')

    return True, d, f"{d:.1f}m[{label}]"


# =============================================================================
# MÓDULO 3 — LOOP PRINCIPAL
# =============================================================================

def main():

    # ── Cargar modelo SVM ─────────────────────────────────────────────────────
    # El modelo es un Pipeline de sklearn: StandardScaler + SVC(kernel='rbf').
    # Entrenado con INRIA Person Dataset (2,752 imágenes, 924 features HOG).
    model_path = os.path.normpath(MODEL_PATH)
    svm_model  = joblib.load(model_path) if os.path.exists(model_path) else None
    print("[OK] Modelo SVM cargado" if svm_model else "[AVISO] Sin modelo — solo PID")

    # ── Inicializar Webots ────────────────────────────────────────────────────
    # Driver() es la única instancia permitida — combina Car + Robot en una API.
    # El timestep sin multiplicador es crítico; multiplicarlo causa freeze en macOS.
    driver   = Driver()
    timestep = int(driver.getBasicTimeStep())   # 10ms por step

    camera  = driver.getDevice("camera")
    camera.enable(timestep)

    display  = driver.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── Inicializar LiDAR ─────────────────────────────────────────────────────
    # REQUISITO macOS Apple Silicon: abrir Webots con Rosetta 2.
    # (Finder → Webots.app → Cmd+I → "Abrir con Rosetta")
    # Sin Rosetta, lidar.enable() genera segmentation fault en la capa C++.
    lidar   = driver.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    _fov    = lidar.getFov()                    # 3.14 rad = 180°
    _n_rays = lidar.getHorizontalResolution()   # 180 rayos (el PROTO ignora valores menores)
    lidar.enablePointCloud()                    # activa visualización de rayos en Webots 3D

    # Calcular ángulos reales de ambas zonas para el display
    _half_rays_f = max(1, int(_n_rays * (math.radians(LIDAR_CONE_FRONT) / _fov) / 2))
    _half_deg_f  = round(_half_rays_f * math.degrees(_fov / _n_rays), 1)
    _half_rays_s = max(1, int(_n_rays * (math.radians(LIDAR_CONE_SIDE)  / _fov) / 2))
    _half_deg_s  = round(_half_rays_s * math.degrees(_fov / _n_rays), 1)
    _half_deg    = _half_deg_f   # para el triángulo del display usamos el cono frontal
    print(f"[LiDAR] FOV={_fov:.2f} rad  rayos={_n_rays}")
    print(f"[LiDAR] Zona frontal: ±{_half_deg_f}° / {LIDAR_MAX_M}m  |  Zona lateral: ±{_half_deg_s}° / {LIDAR_SIDE_M}m")

    # ── Variables de estado ───────────────────────────────────────────────────
    dw, dh   = display.getWidth(), display.getHeight()
    setpoint = dw / 2.0     # el auto debe mantenerse centrado en el display

    integral, prev_err, prev_t = 0.0, 0.0, time.time()
    steering, no_line_frames   = 0.0, 0
    frame_cnt  = 0

    # Contadores SVM: pos_streak sube con detecciones, baja sin ellas
    pos_streak = 0
    neg_streak = 0
    brake_hold = 0          # frames restantes de freno garantizado
    threat     = 'none'     # 'none' | 'pedestrian' | 'objeto'

    last_press = {}
    last_hits  = 0
    last_score = 0.0
    last_wins  = 0
    last_box   = None

    lidar_alert    = False
    lidar_dist_m   = None
    lidar_dist_str = '---'
    lidar_streak   = 0

    driver.setCruisingSpeed(CRUISE_SPEED)
    toggle_music()   # arranca música al iniciar — presiona M para pausar/reanudar
    print("Controlador listo — PID + SVM + LiDAR  |  9 = toggle música  A = screenshot")

    # =========================================================================
    # LOOP PRINCIPAL — se ejecuta cada 10ms (timestep de Webots)
    # =========================================================================
    while driver.step() != -1:
        t  = time.time()
        dt = max(t - prev_t, 1e-3)
        frame_cnt += 1

        # ── Captura de imagen ─────────────────────────────────────────────────
        # Cada frame obtenemos una imagen BGRA de la cámara frontal del BMW.
        # La convertimos a BGR y redimensionamos al tamaño del display (200×150).
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))

        # ── Lectura LiDAR ─────────────────────────────────────────────────────
        # Leemos el sensor cada 3 frames (~30ms). Más frecuente no es posible
        # sin causar inestabilidad en macOS Apple Silicon vía Rosetta.
        # lidar_streak cuenta lecturas consecutivas de alerta para evitar picos.
        if lidar and frame_cnt % LIDAR_EVERY == 0:
            lidar_alert, lidar_dist_m, lidar_dist_str = lidar_read(lidar, _fov, _n_rays)
            lidar_streak = (lidar_streak + 1) if lidar_alert else 0
            print(f"[LIDAR] f={frame_cnt:05d} dist={lidar_dist_str} alert={'SI' if lidar_alert else 'no'} streak={lidar_streak}/{LIDAR_CONFIRM} threat={threat}")

        # ── Detección SVM ─────────────────────────────────────────────────────
        # Corremos el sliding window HOG cada 10 frames (~100ms).
        # Es la operación más costosa del sistema: ~8 ventanas × HOG + SVM.
        # pos_streak: sube +1 con cada detección positiva (máx CONFIRM_N+3)
        # neg_streak: sube +1 con cada frame sin detección
        if svm_model and frame_cnt % DETECT_EVERY == 0:
            detected, last_hits, last_score, last_wins, last_box = svm_detect(bgr, svm_model)
            if detected:
                pos_streak = min(pos_streak + 1, CONFIRM_N + 3)
                neg_streak = 0
            else:
                neg_streak = min(neg_streak + 1, RELEASE_N + 3)
                pos_streak = max(pos_streak - 1, 0)
            print(f"[SVM] f={frame_cnt:05d} wins={last_wins} "
                  f"hits={last_hits}/{MIN_HITS} score={last_score:.3f} "
                  f"thresh={SVM_THRESHOLD} pos={pos_streak}/{CONFIRM_N} "
                  f"neg={neg_streak}/{RELEASE_N} "
                  f"lidar={'ALERTA' if lidar_alert else 'ok'}({lidar_dist_str}) threat={threat}")

        # ── Lógica de amenaza ─────────────────────────────────────────────────
        # NIVEL 1 — SVM confirma peatón (2 scans positivos consecutivos)
        # El auto frena y el freno dura mínimo HOLD_FRAMES (1 segundo).
        # neg_streak debe llegar a 4 DESPUÉS de que expire el hold para liberar.
        if pos_streak >= CONFIRM_N:
            threat     = 'pedestrian'
            brake_hold = HOLD_FRAMES
            neg_streak = 0
        elif neg_streak >= RELEASE_N and brake_hold <= 0:
            threat     = 'none'
            pos_streak = 0

        # Cuenta regresiva del hold — cuando llega a 0 sin confirmación, libera
        if brake_hold > 0:
            brake_hold -= 1
            if brake_hold == 0 and pos_streak < CONFIRM_N:
                threat = 'none'

        # NIVEL 2 — LiDAR override total a <5m
        # Si algo está a menos de 5m, el auto frena SIEMPRE, sin importar si
        # el SVM ya detectó o no. Solo se reinicia brake_hold al transicionar.
        if lidar and lidar_dist_m is not None and lidar_dist_m < LIDAR_OVERRIDE_M:
            if threat != 'objeto':
                brake_hold = HOLD_FRAMES
            threat = 'objeto'

        # NIVEL 3 — LiDAR alerta a <8m (solo si SVM no ha actuado aún)
        if lidar and lidar_dist_m is not None and lidar_dist_m < LIDAR_EMERGENCY_M and threat == 'none':
            threat     = 'objeto'
            brake_hold = HOLD_FRAMES

        # NIVEL 4 — LiDAR alerta normal con confirmación
        if lidar and lidar_streak >= LIDAR_CONFIRM and threat == 'none':
            threat     = 'objeto'
            brake_hold = HOLD_FRAMES

        # ── PID de seguimiento de carril ──────────────────────────────────────
        # 1. Convertir a HSV y extraer solo los píxeles amarillos
        # 2. Detectar bordes de la máscara amarilla con Canny
        # 3. Aplicar ROI trapezoidal — solo la zona del carril
        # 4. Detectar segmentos de línea con HoughLinesP
        # 5. Calcular el centro del carril entre líneas izq/der
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        edges = cv2.Canny(ymask, 50, 150)
        roi   = apply_roi(edges, dh, dw)
        lines = filter_lines(cv2.HoughLinesP(roi, 1, np.pi/180, 20,
                                             minLineLength=20, maxLineGap=15))
        center = compute_center(lines)

        # ── Display de diagnóstico ────────────────────────────────────────────
        # Se actualiza cada 3 frames para reducir carga en el sistema.
        #
        # GUÍA VISUAL DE ELEMENTOS EN PANTALLA:
        #   Rectángulo cian       → ROI del SVM (zona 59%–85% del alto)
        #   Líneas naranjas       → Límites horizontales del barrido (30%–70%)
        #   Triángulo verde/rojo  → Cono activo del LiDAR (±30.5°)
        #   Barra lateral         → Verde=LiDAR libre | Rojo=LiDAR alerta
        #   Rectángulo rojo       → Mejor ventana SVM sobre umbral
        #   Líneas amarillas      → Segmentos Hough del carril (PID)
        #   Texto superior izq:
        #     PID OK/PEATON/OBJETO → estado del sistema
        #     V:30 St:X.XX         → velocidad y ángulo de dirección
        #     SVM:X.XXX(>0.25)     → score SVM y umbral
        #     pos:X/2              → racha positiva hacia confirmación
        #     LiDAR:X.Xm ±30.5°   → distancia y ángulo del cono
        #     hold:X               → frames restantes de freno
        if frame_cnt % DISPLAY_EVERY == 0:
            viz = frame.copy()

            # ROI del SVM — rectángulo cian
            roi_y0 = int(dh * 0.59)
            roi_y1 = int(dh * 0.85)
            cv2.rectangle(viz, (0, roi_y0), (dw - 1, roi_y1), (200, 200, 0), 1)

            # Límites horizontales del barrido SVM — líneas naranjas
            hx0 = int(dw * 0.30);  hx1 = int(dw * 0.70)
            cv2.line(viz, (hx0, roi_y0), (hx0, roi_y1), (0, 140, 255), 1)
            cv2.line(viz, (hx1, roi_y0), (hx1, roi_y1), (0, 140, 255), 1)

            # Barra lateral — verde=libre, rojo=alerta LiDAR
            lidar_bar_col = (0, 0, 255) if lidar_alert else (0, 200, 0)
            cv2.rectangle(viz, (0, 0), (3, dh - 1), lidar_bar_col, -1)

            # Triángulo del cono LiDAR — proyectado desde el centro inferior
            cam_fov_deg = math.degrees(1.0)     # FOV de la cámara = 1 rad ≈ 57°
            cone_px     = int(_half_deg * (dw / cam_fov_deg))
            cx_lidar    = dw // 2
            cv2.line(viz, (cx_lidar, dh - 1), (cx_lidar - cone_px, 0), lidar_bar_col, 2)
            cv2.line(viz, (cx_lidar, dh - 1), (cx_lidar + cone_px, 0), lidar_bar_col, 2)

            # Cuadro rojo sobre la ventana SVM con mayor score (solo si supera umbral)
            if last_box is not None and last_score >= SVM_THRESHOLD:
                bx1, by1, bx2, by2 = last_box
                ih, iw = bgr.shape[:2]
                fx1 = int(bx1 * dw / iw);  fy1 = int(by1 * dh / ih)
                fx2 = int(bx2 * dw / iw);  fy2 = int(by2 * dh / ih)
                cv2.rectangle(viz, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

            # Segmentos Hough del PID — líneas amarillas del carril detectadas
            if lines is not None:
                for l in lines:
                    cv2.line(viz, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 255), 1)

            display_color(display, viz)

            # Texto de estado — color refleja la urgencia
            if threat == 'objeto':
                display.setColor(0xFF6600)
                display.drawText("OBJETO", 2, 2)
            elif threat == 'pedestrian':
                display.setColor(0xFF0000)
                display.drawText("PEATON", 2, 2)
            else:
                display.setColor(0x00FF00)
                display.drawText("PID OK", 2, 2)

            display.setColor(0xFFFFFF)
            display.drawText(f"V:{CRUISE_SPEED} St:{steering:.2f}", 2, 12)

            score_color = 0xFF4444 if last_score >= SVM_THRESHOLD else (0xFFAA00 if last_score >= 0.0 else 0xCCCCCC)
            display.setColor(score_color)
            display.drawText(f"SVM:{last_score:.3f}(>{SVM_THRESHOLD})", 2, 22)

            pos_color = 0xFF2222 if pos_streak >= CONFIRM_N else 0xFF8800
            display.setColor(pos_color)
            display.drawText(f"pos:{pos_streak}/{CONFIRM_N}", 2, 32)

            lidar_color = 0xFF2222 if lidar_alert else 0x44FF44
            display.setColor(lidar_color)
            display.drawText(f"LiDAR:{lidar_dist_str} ±{_half_deg}°", 2, 42)

            display.setColor(0xAAAAAA)
            display.drawText(f"hold:{brake_hold}", 2, 52)

        # ── Acción de control ─────────────────────────────────────────────────
        # Si hay amenaza activa: frenar completamente.
        # Los intermitentes se activan solo para 'objeto' (LiDAR) — indica
        # obstáculo genérico. 'pedestrian' (SVM) frena sin intermitentes.
        if threat != 'none':
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            driver.setHazardFlashers(threat == 'objeto')
            steering, integral = 0.0, 0.0
            prev_t = t
            continue

        # Sin amenaza: circular y corregir dirección con PID
        driver.setHazardFlashers(False)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        if center is not None:
            no_line_frames = 0
            # Error = desviación del centro del carril respecto al centro del display
            # Normalizado a [-1, 1]: error=0 → auto centrado, error=1 → derecha
            error    = (center - setpoint) / setpoint
            integral = max(-0.5, min(0.5, integral + error * dt))
            raw_s    = Kp * error + Ki * integral + Kd * (error - prev_err) / dt
            raw_s    = max(-MAX_ANGLE, min(MAX_ANGLE, raw_s))
            # Suavizado de dirección: MAX_STEER_RATE limita cambios bruscos del volante
            steering = max(steering - MAX_STEER_RATE,
                           min(steering + MAX_STEER_RATE, raw_s))
            prev_err = error
        else:
            # Sin líneas detectadas: reducir integradores y centrar gradualmente
            no_line_frames += 1
            integral       *= 0.6
            prev_err        = 0.0
            if no_line_frames > 10:
                steering *= 0.95   # volver al centro si perdemos el carril

        driver.setSteeringAngle(steering)
        prev_t = t

        # Teclas: A=screenshot  9=toggle música
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and t - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = t
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"[A] {ts}.png guardada")
                elif key == ord('9'):
                    try:
                        toggle_music()
                    except Exception as e:
                        print(f"[Audio] {e}")
                    print(f"[A] {ts}.png guardada")

if __name__ == "__main__":
    main()
