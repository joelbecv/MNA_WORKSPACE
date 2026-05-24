# =============================================================================
# simple_controller_pedestrian_v2_stable2.py
# Actividad 3.1 — Detección de Peatones con SVM
# Navegación Autónoma — Maestría en Inteligencia Artificial
# =============================================================================
#
# ARQUITECTURA DEL SISTEMA (3 módulos principales):
#
#   1. PID — Seguimiento de carril amarillo
#      Cámara → HSV (filtro amarillo) → Canny (bordes) → HoughLinesP (líneas)
#      → PID calcula el ángulo de dirección para mantener el auto centrado.
#
#   2. SVM — Detección de peatones por visión
#      Cámara → ROI (recorte de zona relevante) → Escala ×5 → Ventana deslizante
#      → HOG (descriptor de gradientes) → SVM.decision_function() → score
#      Si score ≥ 0.30 en 2 scans consecutivos → PEATÓN CONFIRMADO.
#
#   3. Lógica de control dual:
#      · Peatón detectado (SVM) → freno de emergencia, SIN intermitentes
#      · Barril detectado (LiDAR, cuando habilitado) → freno + intermitentes
#      · Sin amenaza → PID retoma el control y el auto avanza a 30 km/h
#
# NOTA SOBRE LIDAR:
#   El sensor Sick LMS 291 está montado en el BMW dentro del mundo Webots.
#   En macOS con Webots R2025a, llamar a lidar.enable() congela la simulación
#   (beachball) independientemente del intervalo — esto se diagnosticó probando
#   valores desde 10ms hasta 500ms, todos con el mismo resultado.
#   Por eso el LiDAR se deja deshabilitado en runtime (lidar = None) y la
#   detección se realiza íntegramente por cámara + SVM.
#   En Linux/Windows el bloque marcado "Sección 2-B" puede activarse sin cambios.
#
# DOMAIN GAP (limitación conocida del modelo):
#   El modelo SVM fue entrenado con el INRIA Person Dataset (fotos reales).
#   Los modelos 3D de Webots tienen texturas sintéticas sin sombras ni texturas
#   de ropa reales, por lo que los scores son más bajos de lo esperado:
#     - Fondo/edificios:   score ≈ 0.06 – 0.19
#     - Peatones Webots:   score ≈ 0.30 – 0.70
#     - Esperado INRIA:    score > 1.0  (nunca alcanzado en Webots)
#   El umbral 0.30 fue calibrado empíricamente para separar ambas distribuciones.
#
# BIBLIOGRAFÍA:
#   - INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
#   - Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005
#   - Webots Driver API: https://cyberbotics.com/doc/automobile/driver-library
# =============================================================================

# --- Librerías de Webots ---
# Display y Keyboard son periféricos del robot accesibles desde el controlador.
# Driver (de vehicle) es la interfaz de alto nivel para vehículos con motor,
# dirección y frenos — reemplaza a Robot() para este tipo de modelo.
from controller import Display, Keyboard
from vehicle import Car, Driver

# --- Visión computacional y ML ---
# hog: descriptor Histogram of Oriented Gradients (skimage)
# cv2: OpenCV — procesamiento de imagen (filtros, Canny, Hough, etc.)
# joblib: carga/guarda el modelo SVM entrenado desde archivo .joblib
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import math
import os
import time
from datetime import datetime

# Ruta al modelo SVM — está dos niveles arriba del controlador junto al .wbt
_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))


# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================

# --- Control de velocidad y dirección (PID) ---
CRUISE_SPEED   = 30      # velocidad de crucero en km/h
MAX_ANGLE      = 0.5     # máximo ángulo de dirección en radianes (~28°)
MAX_STEER_RATE = 0.03    # máximo cambio de ángulo por frame — suaviza las curvas
DEBOUNCE_TIME  = 0.1     # tiempo mínimo entre pulsaciones de teclado (segundos)

# Ganancias del controlador PID:
#   Kp (proporcional) — reacción inmediata al error de posición
#   Ki (integral)     — corrige el error acumulado (desvíos sostenidos)
#   Kd (derivativo)   — amortigua oscilaciones (frena si el error crece rápido)
Kp, Ki, Kd     = 0.28, 0.01, 0.01

# Filtro de líneas — descarta líneas casi horizontales (son ruido, no carril)
MIN_ABS_SLOPE  = 0.4

# Rango de color amarillo en espacio HSV
# HSV es mejor que RGB para segmentar colores bajo distintas iluminaciones
# H (Hue): 15–35 captura amarillo sin confundir con verde o naranja
# S y V ≥ 80: excluye grises/blancos que caerían en ese rango de H
YELLOW_LOW     = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH    = np.array([35, 255, 255], dtype=np.uint8)

# --- LiDAR (parámetros documentados — deshabilitado en macOS, ver Sección 2-B) ---
LIDAR_CONE_DEG = 25      # semángulo del cono frontal a monitorear (±12.5°)
LIDAR_MAX_M    = 20.0    # distancia máxima de alerta en metros

# --- SVM — Ventana deslizante ---
HOG_WIN_W    = 64        # ancho de la ventana HOG en píxeles (estándar para personas)
HOG_WIN_H    = 128       # alto de la ventana HOG — aspecto 1:2 vertical para humanos
SLIDE_STEP   = 32        # paso horizontal entre ventanas (50% de solapamiento)

# --- SVM — Control de temporización y confirmación ---
DETECT_EVERY  = 5        # ejecutar SVM cada 5 frames (~50 ms con timestep=10 ms)
                         # reducir cómputo sin perder reactividad

CONFIRM_N     = 2        # cuántos scans positivos CONSECUTIVOS se necesitan para confirmar
                         # peatón — evita que un único falso positivo active el freno

RELEASE_N     = 4        # cuántos scans negativos consecutivos para liberar el freno
                         # más alto que CONFIRM_N para evitar que el auto arranque demasiado pronto

HOLD_FRAMES   = 80       # frames mínimos de freno activo tras confirmación (~0.8 s)
                         # garantiza que el auto se detenga completamente

MIN_HITS      = 1        # mínimo de ventanas con score ≥ umbral por scan para contar como positivo

# Umbral de decisión del SVM:
# Se calibró observando los scores en consola con el mundo real de Webots:
#   - Fondo, edificios, postes: score máximo observado ≈ 0.19
#   - Peatones Webots 3D:       score mínimo observado ≈ 0.30
# Un umbral de 0.30 separa las dos distribuciones sin falsos positivos de fondo.
SVM_THRESHOLD = 0.30


# =============================================================================
# SECCIÓN 1 — VISIÓN: SEGUIMIENTO DE CARRIL (funciones auxiliares)
# =============================================================================

def get_image(camera):
    """Convierte el buffer BGRA de la cámara Webots a un array NumPy (H×W×4)."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_color(display, bgr_img):
    """
    Muestra una imagen BGR en el Display de Webots.
    Webots espera formato RGB, por eso se invierte el orden de canales (::-1).
    """
    rgb = bgr_img[:, :, ::-1].copy()
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)   # libera memoria del Display

def apply_roi(edges, h, w):
    """
    Aplica una máscara trapezoidal sobre la imagen de bordes Canny.

    El trapecio cubre la parte inferior del frame (donde está el carril visible)
    y excluye el cielo, edificios y horizontes lejanos que generan ruido.

    Vértices del trapecio (en fracción del ancho/alto):
      Inferior izq: (10%, 100%)  ─── Inferior der: (90%, 100%)
      Superior izq: (35%,  60%)  ─── Superior der: (65%,  60%)
    """
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.60)),
        (int(w * 0.65), int(h * 0.60)),
        (int(w * 0.90), h),
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines(lines):
    """
    Descarta líneas casi horizontales detectadas por Hough.
    Una línea con pendiente < MIN_ABS_SLOPE es probablemente ruido horizontal
    (borde de acera, sombra) y no corresponde al carril del auto.
    """
    if lines is None:
        return None
    ok = [l for l in lines
          if l[0][2] != l[0][0]                                       # no verticales puras
          and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(ok) if ok else None

def compute_center(lines):
    """
    Estima la posición lateral del carril a partir de las líneas detectadas.

    Estrategia:
    - Líneas con pendiente negativa → están a la izquierda del auto (lx)
    - Líneas con pendiente positiva → están a la derecha del auto (rx)
    - Si hay ambas: el centro del carril = promedio de lx y rx
    - Si solo hay un lado: se usa el promedio de todas las líneas (ax)

    El valor devuelto es la coordenada X del centro estimado del carril.
    """
    if lines is None:
        return None
    lx, rx, ax = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2-y1)/(x2-x1)
        mid   = (x1+x2)/2
        ax.append(mid)
        (lx if slope < 0 else rx).append(mid)
    if lx and rx:
        return (np.mean(lx)+np.mean(rx))/2.0
    return np.mean(ax) if ax else None


# =============================================================================
# SECCIÓN 2-A — SVM: DETECCIÓN DE PEATONES (ventana deslizante HOG)
# =============================================================================

def svm_detect(bgr, model):
    """
    Implementa la búsqueda por ventana deslizante + HOG + SVM para detectar peatones.

    Flujo completo:
      1. Recortar ROI vertical (51%–71% de la imagen) — zona donde aparecen
         peatones en el carril, evitando cielo y zona frontal muy cercana.
      2. Escalar el ROI ×5 — el ROI tiene ~26px de alto; el HOG necesita 128px
         mínimo. Escalar ×5 da 130px. Equivale a "acercar" la imagen.
      3. Deslizar la ventana 64×128px por la zona horizontal 30%–70%,
         saltando el centro 42%–58% donde viven las líneas amarillas del carril
         (evita falsos positivos de la pintura del camino).
      4. Por cada ventana: calcular HOG (924 features) → SVM.decision_function()
         → obtener score. Un score ≥ 0.30 cuenta como ventana positiva.
      5. Devolver si hay ≥ MIN_HITS ventanas positivas, el score máximo,
         y las coordenadas de la ventana con mayor score (para visualización).

    Retorna: (detected, hits, max_score, windows_evaluadas, best_box_coords)
    """
    h, w      = bgr.shape[:2]

    # --- Paso 1: Recorte vertical de la ROI ---
    # Solo la franja donde los peatones aparecen a distancia media
    # (51%–71% del alto de la imagen desde arriba)
    y_off     = int(h * 0.51)
    roi       = bgr[y_off : int(h * 0.71), :]   # altura ≈ 26px

    total     = 0        # contador de ventanas que superan el umbral
    max_score = -999.0   # score más alto visto en este scan
    windows   = 0        # total de ventanas evaluadas (para diagnóstico)
    best_box  = None     # coordenadas de la ventana con mayor score

    for scale in (5.0,):
        # --- Paso 2: Escalar el ROI ---
        # ROI tiene ~26px de alto. HOG necesita 128px → escala mínima = 128/26 ≈ 4.9
        # Se usa 5.0 para garantizar margen (26×5 = 130px ≥ 128px requeridos).
        scaled = cv2.resize(roi, (int(w*scale), int(roi.shape[0]*scale)))
        sh, sw = scaled.shape[:2]
        if sh < HOG_WIN_H or sw < HOG_WIN_W:
            continue   # ROI demasiado pequeño, saltar esta escala

        # --- Paso 3: Definir zona horizontal de búsqueda ---
        x0 = int(sw * 0.30)   # inicio del barrido horizontal (30%)
        x1 = int(sw * 0.70)   # fin del barrido horizontal (70%)

        # Hueco central: las líneas amarillas del carril están en el centro de la imagen
        # Sus gradientes HOG generaban falsos positivos. Se excluye 42%–58%.
        cx_skip_lo = int(sw * 0.42)
        cx_skip_hi = int(sw * 0.58)

        # Paso vertical: HOG_WIN_H // 4 = 32px (25% de solapamiento vertical)
        ystep = max(16, HOG_WIN_H // 4)

        for y in range(0, sh - HOG_WIN_H + 1, ystep):
            for x in range(x0, min(x1, sw - HOG_WIN_W + 1), SLIDE_STEP):

                # Saltar zona central donde viven las líneas amarillas
                if cx_skip_lo <= x < cx_skip_hi:
                    continue

                # --- Paso 4: HOG + SVM ---
                # Convertir a escala de grises — HOG trabaja sobre gradientes de intensidad
                win  = cv2.cvtColor(
                    scaled[y:y+HOG_WIN_H, x:x+HOG_WIN_W], cv2.COLOR_BGR2GRAY)

                # Extraer descriptor HOG (924 features):
                #   orientations=11  → 11 orientaciones de gradiente (vs 9 standard)
                #   pixels_per_cell=(16,16) → celdas de 16×16 px
                #   cells_per_block=(2,2)   → normalización en bloques de 2×2 celdas
                feat = hog(win, orientations=11, pixels_per_cell=(16,16),
                           cells_per_block=(2,2), transform_sqrt=False,
                           feature_vector=True)

                # decision_function devuelve la distancia al hiperplano de separación.
                # Positivo → lado "peatón", negativo → lado "fondo".
                # Más alto = más confianza de que es un peatón.
                score = model.decision_function([feat])[0]
                windows += 1

                # Guardar la ventana con mayor score para dibujarla en el display
                if score > max_score:
                    max_score = score
                    # Convertir coordenadas de vuelta al espacio original de bgr
                    bx1 = max(0, int(x / scale))
                    by1 = max(0, int(y / scale) + y_off)
                    bx2 = min(w,  int((x + HOG_WIN_W) / scale))
                    by2 = min(h,  int((y + HOG_WIN_H) / scale) + y_off)
                    best_box = (bx1, by1, bx2, by2)

                if score >= SVM_THRESHOLD:
                    total += 1   # esta ventana cuenta como detección positiva

    # Una sola ventana positiva (MIN_HITS=1) es suficiente para marcar este scan como positivo
    detected = total >= MIN_HITS
    return detected, total, max_score, windows, best_box


# =============================================================================
# SECCIÓN 2-B — LIDAR (documentado, deshabilitado en runtime macOS)
# =============================================================================
#
# El Sick LMS 291 es un sensor láser de 180° con hasta 720 rayos por barrido.
# En Linux/Windows, descomentar las líneas marcadas "LIDAR" en main().
# NO usar enablePointCloud() — también causa freeze en macOS.
#
# def lidar_obstacle(lidar):
#     """
#     Lee el cono frontal ±12.5° del Sick LMS 291.
#     Devuelve True si hay algún objeto a menos de LIDAR_MAX_M metros.
#     Complementa al SVM: detecta obstáculos no-humanos (barriles, vehículos).
#     """
#     ranges  = lidar.getRangeImage()
#     if not ranges:
#         return False
#     n       = len(ranges)
#     fov_rad = lidar.getFov()
#     half    = max(1, int(n * (math.radians(LIDAR_CONE_DEG) / fov_rad) / 2))
#     center  = n // 2
#     cone    = [r for r in ranges[center-half: center+half]
#                if not (math.isnan(r) or math.isinf(r))]
#     return bool(cone) and min(cone) < LIDAR_MAX_M


# =============================================================================
# SECCIÓN 3 — MAIN: LOOP PRINCIPAL DE CONTROL
# =============================================================================

def main():

    # --- Cargar el modelo SVM desde disco ---
    # Pipeline(StandardScaler → SVC rbf) entrenado con INRIA Person Dataset.
    # Si no existe el archivo, el controlador corre solo con PID (sin detección).
    model_path = os.path.normpath(MODEL_PATH)
    svm_model  = joblib.load(model_path) if os.path.exists(model_path) else None
    print("[OK] Modelo SVM cargado" if svm_model else "[AVISO] Sin modelo — solo PID")

    # --- Inicializar el driver del vehículo ---
    # Driver() hereda de Robot — una sola instancia cubre motor, dirección y frenos.
    # NO usar Car() + Driver() juntos — Webots lanza error "only one Robot instance".
    driver   = Driver()

    # CRÍTICO: usar getBasicTimeStep() SIN multiplicador.
    # timestep * 2 o * 3 causa freeze (beachball) en macOS con este mundo.
    timestep = int(driver.getBasicTimeStep())

    # --- Activar sensores ---
    camera  = driver.getDevice("camera")
    camera.enable(timestep)   # cámara frontal del BMW

    display  = driver.getDevice("display_image")   # display de diagnóstico en Webots
    keyboard = Keyboard()
    keyboard.enable(timestep)   # tecla A → guarda captura de pantalla

    # --- LiDAR (deshabilitado en macOS) ---
    # En Linux/Windows reemplazar las dos líneas siguientes por:
    #   lidar = driver.getDevice("Sick LMS 291")
    #   lidar.enable(100)   # solo getRangeImage(), NO enablePointCloud()
    lidar = None   # None → lidar_alert=True siempre (SVM siempre evalúa)

    # --- Variables de estado del loop ---
    dw, dh   = display.getWidth(), display.getHeight()   # 200×150 px
    setpoint = dw / 2.0   # el auto debe estar centrado horizontalmente

    integral, prev_err, prev_t = 0.0, 0.0, time.time()
    steering, no_line_frames   = 0.0, 0
    frame_cnt  = 0       # contador de frames (controla DETECT_EVERY)
    pos_streak = 0       # scans positivos consecutivos del SVM
    neg_streak = 0       # scans negativos consecutivos del SVM
    brake_hold = 0       # frames restantes de freno garantizado (HOLD_FRAMES)
    threat     = 'none'  # estado actual: 'none' | 'pedestrian' | 'barrel'
    last_press = {}      # registro de última pulsación por tecla (debounce)

    # Variables de diagnóstico (último resultado del SVM — para display)
    last_hits  = 0
    last_score = 0.0
    last_wins  = 0
    last_box   = None    # coordenadas de la ventana con mayor score

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("Controlador listo — PID + SVM (LiDAR documentado, ver Sección 2-B)")

    # =========================================================================
    # LOOP PRINCIPAL — ejecuta cada timestep (≈10 ms)
    # =========================================================================
    while driver.step() != -1:
        t  = time.time()
        dt = max(t - prev_t, 1e-3)   # delta de tiempo real (protegido contra 0)
        frame_cnt += 1

        # --- Capturar y preparar imagen de cámara ---
        # get_image → array BGRA  →  convertir a BGR  →  redimensionar al display
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))   # frame = imagen de trabajo para PID y display
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- LiDAR ---
        # Con LiDAR real: lidar_alert = lidar_obstacle(lidar) si lidar else True
        # En macOS: siempre True → el SVM siempre evalúa (no depende del LiDAR)
        lidar_alert = True

        # =====================================================================
        # BLOQUE SVM — ejecutar cada DETECT_EVERY frames (no cada frame)
        # =====================================================================
        # Motivo: HOG + SVM sobre ~20 ventanas tarda ~8 ms en CPU.
        # Correrlo cada 5 frames (50 ms) es suficiente para detectar peatones
        # a distancia media sin saturar el timestep del simulador.
        if svm_model and lidar_alert and frame_cnt % DETECT_EVERY == 0:
            detected, last_hits, last_score, last_wins, last_box = svm_detect(bgr, svm_model)

            if detected:
                # Scan positivo: incrementar racha positiva, resetear negativa
                pos_streak = min(pos_streak + 1, CONFIRM_N + 3)
                neg_streak = 0
            else:
                # Scan negativo: incrementar racha negativa, decrementar positiva
                neg_streak = min(neg_streak + 1, RELEASE_N + 3)
                pos_streak = max(pos_streak - 1, 0)

            # Log de diagnóstico en consola (visible durante grabación del video)
            print(f"[SVM] f={frame_cnt:05d} wins={last_wins} "
                  f"hits={last_hits}/{MIN_HITS} score={last_score:.3f} "
                  f"thresh={SVM_THRESHOLD} pos={pos_streak}/{CONFIRM_N} "
                  f"neg={neg_streak}/{RELEASE_N} threat={threat}")

        # =====================================================================
        # LÓGICA DE AMENAZA — máquina de estados simple
        # =====================================================================
        if pos_streak >= CONFIRM_N:
            # CONFIRM_N=2 scans positivos consecutivos → confirmar peatón
            # Se resetea la racha negativa para evitar liberación inmediata
            threat     = 'pedestrian'
            brake_hold = HOLD_FRAMES   # garantiza al menos 0.8 s de freno
            neg_streak = 0

        elif neg_streak >= RELEASE_N and brake_hold <= 0:
            # RELEASE_N=4 scans negativos + hold expirado → liberar el freno
            # Más estricto que la confirmación para evitar arranques prematuros
            threat     = 'none'
            pos_streak = 0

        # Contador de hold: decrementar cada frame mientras está activo
        if brake_hold > 0:
            brake_hold -= 1
            if brake_hold == 0 and pos_streak < CONFIRM_N:
                threat = 'none'   # hold expiró y no hay nueva confirmación

        # Si el LiDAR detecta obstáculo pero el SVM aún no confirmó → barril
        # (Solo aplica cuando el LiDAR real está habilitado en Linux/Windows)
        if lidar and lidar_alert and threat == 'none':
            threat     = 'barrel'
            brake_hold = HOLD_FRAMES

        # =====================================================================
        # MÓDULO PID — seguimiento de carril amarillo
        # =====================================================================
        # Paso 1: Convertir el frame a HSV y segmentar solo el amarillo
        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask   = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)   # máscara binaria amarillo

        # Paso 2: Dilatar la máscara 3×3 para rellenar gaps en la línea pintada
        # Canny se aplica SOLO sobre la máscara amarilla dilatada, NO sobre la imagen gris.
        # Razón: Canny sobre gris capturaba los bordes de peatones y cruces peatonales,
        # cuyos gradientes competían con los de la línea amarilla y desviaban el PID.
        ymask_d = cv2.dilate(ymask, np.ones((3, 3), np.uint8), iterations=1)

        # Paso 3: Canny → bordes de los píxeles amarillos
        edges   = cv2.Canny(ymask_d, 50, 150)

        # Paso 4: Aplicar ROI trapezoidal → descartar bordes fuera del área del carril
        roi   = apply_roi(edges, dh, dw)

        # Paso 5: Transformada de Hough probabilística → detectar segmentos de línea
        lines = filter_lines(cv2.HoughLinesP(roi, 1, np.pi/180, 20,
                                             minLineLength=20, maxLineGap=15))

        # Paso 6: Estimar el centro del carril a partir de líneas izquierda/derecha
        center = compute_center(lines)

        # =====================================================================
        # DISPLAY — visualización de diagnóstico en tiempo real
        # =====================================================================
        viz = frame.copy()   # copia del frame BGR sobre la que dibujar

        # Rectángulo cian: banda vertical donde el SVM busca peatones (51%–71%)
        roi_y0 = int(dh * 0.51)
        roi_y1 = int(dh * 0.71)
        cv2.rectangle(viz, (0, roi_y0), (dw - 1, roi_y1), (200, 200, 0), 1)

        # Líneas naranjas: límites horizontales del barrido SVM (30%–70%)
        # Líneas rojas oscuras: zona central excluida (42%–58%) — líneas amarillas
        hx0 = int(dw * 0.30)
        hx1 = int(dw * 0.70)
        cx0 = int(dw * 0.42)
        cx1 = int(dw * 0.58)
        cv2.line(viz, (hx0, roi_y0), (hx0, roi_y1), (0, 140, 255), 1)   # límite izq
        cv2.line(viz, (hx1, roi_y0), (hx1, roi_y1), (0, 140, 255), 1)   # límite der
        cv2.line(viz, (cx0, roi_y0), (cx0, roi_y1), (0, 0, 160), 1)     # inicio hueco central
        cv2.line(viz, (cx1, roi_y0), (cx1, roi_y1), (0, 0, 160), 1)     # fin hueco central

        # Rectángulo sobre la ventana con mayor score SVM:
        #   Rojo  → score ≥ umbral (posible peatón)
        #   Verde → score < umbral (descartado como fondo)
        if last_box is not None:
            bx1, by1, bx2, by2 = last_box
            ih, iw = bgr.shape[:2]
            fx1 = int(bx1 * dw / iw);  fy1 = int(by1 * dh / ih)
            fx2 = int(bx2 * dw / iw);  fy2 = int(by2 * dh / ih)
            box_col = (0, 0, 255) if last_score >= SVM_THRESHOLD else (0, 200, 0)
            cv2.rectangle(viz, (fx1, fy1), (fx2, fy2), box_col, 2)

        # Líneas del carril detectadas por Hough (amarillo)
        if lines is not None:
            for l in lines:
                cv2.line(viz, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 255), 1)

        # Enviar la imagen con overlays al Display de Webots
        display_color(display, viz)

        # --- Textos de estado sobre el display ---
        # Fila 0: estado del sistema (PEATON / BARRIL / PID OK)
        if threat == 'barrel':
            display.setColor(0xFF6600)
            display.drawText("BARRIL", 2, 2)
        elif threat == 'pedestrian':
            display.setColor(0xFF0000)
            display.drawText("PEATON", 2, 2)
        else:
            display.setColor(0x00FF00)
            display.drawText("PID OK", 2, 2)

        # Fila 1: velocidad y ángulo de dirección actual
        display.setColor(0xFFFFFF)
        display.drawText(f"V:{CRUISE_SPEED} St:{steering:.2f}", 2, 12)

        # Fila 2: score SVM (rojo si supera umbral, naranja si positivo pero bajo, gris si negativo)
        score_color = 0xFF4444 if last_score >= SVM_THRESHOLD else (0xFFAA00 if last_score >= 0.0 else 0xCCCCCC)
        display.setColor(score_color)
        display.drawText(f"SVM:{last_score:.3f}(>{SVM_THRESHOLD})", 2, 22)

        # Fila 3: número de ventanas positivas vs total evaluadas
        display.setColor(0xFFFFFF)
        display.drawText(f"hits:{last_hits}/{MIN_HITS} w:{last_wins}", 2, 32)

        # Fila 4: racha positiva (rojo si confirma, naranja si acumulando)
        pos_color = 0xFF2222 if pos_streak >= CONFIRM_N else 0xFF8800
        display.setColor(pos_color)
        display.drawText(f"pos:{pos_streak}/{CONFIRM_N}", 2, 42)

        # Fila 5: racha negativa (verde = sistema liberándose del freno)
        display.setColor(0x44FF44)
        display.drawText(f"neg:{neg_streak}/{RELEASE_N}", 2, 52)

        # Fila 6: frames restantes de hold y número de frame actual
        display.setColor(0xAAAAAA)
        display.drawText(f"hold:{brake_hold} f:{frame_cnt}", 2, 62)

        # =====================================================================
        # CONTROL DEL ACTUADOR — freno de emergencia vs conducción normal
        # =====================================================================
        if threat != 'none':
            # MODO FRENO: peatón o barril detectado
            # Se anula la velocidad y se aplica freno máximo.
            # Las luces intermitentes se encienden solo para barriles.
            # Se resetean integral y error para que el PID arranque limpio.
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            driver.setHazardFlashers(threat == 'barrel')
            steering, integral = 0.0, 0.0
            prev_t = t
            continue   # saltar el bloque PID — no tiene sentido calcular dirección si está frenando

        # MODO CONDUCCIÓN NORMAL: sin amenaza activa
        driver.setHazardFlashers(False)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        # --- PID de dirección ---
        if center is not None:
            no_line_frames = 0

            # Error normalizado: cuánto se desvía el centro del carril del centro del frame
            # Negativo → carril a la izquierda → girar izquierda
            # Positivo → carril a la derecha → girar derecha
            error    = (center - setpoint) / setpoint

            # Término integral: clamp ±0.5 para evitar windup (acumulación descontrolada)
            integral = max(-0.5, min(0.5, integral + error * dt))

            # Ángulo de dirección crudo: suma de los tres términos PID
            raw_s    = Kp*error + Ki*integral + Kd*(error-prev_err)/dt
            raw_s    = max(-MAX_ANGLE, min(MAX_ANGLE, raw_s))   # clamp al rango físico

            # Suavizado de dirección: limitar cambio máximo por frame (MAX_STEER_RATE)
            # Evita oscilaciones bruscas y hace la conducción más estable
            steering = max(steering-MAX_STEER_RATE,
                           min(steering+MAX_STEER_RATE, raw_s))
            prev_err = error

        else:
            # Sin línea visible: reducir integral para no acumular error fantasma
            # y disminuir gradualmente la dirección (el auto tiende a seguir recto)
            no_line_frames += 1
            integral       *= 0.6
            prev_err        = 0.0
            if no_line_frames > 10:
                steering *= 0.95   # straighten out suavemente después de 10 frames sin línea

        driver.setSteeringAngle(steering)
        prev_t = t

        # --- Teclado: tecla A guarda captura de la cámara ---
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and t-last_press[key] < DEBOUNCE_TIME):
                last_press[key] = t
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd()+"/"+ts+".png", 1)
                    print(f"[A] {ts}.png guardada")

if __name__ == "__main__":
    main()
