# =============================================================================
# AG_simple_controller_pedestrian.py
# Controlador de Vehículo Autónomo con Detección Dual (LiDAR + SVM)
# Maestría en Inteligencia Artificial — Navegación Autónoma
# =============================================================================
#
# DESCRIPCIÓN DEL FUNCIONAMIENTO:
#   1. Seguidor de Línea (PID): Sigue la línea amarilla central mediante la cámara.
#      Aplica segmentación HSV, detector de bordes Canny y transformada de Hough.
#   2. Detección por LiDAR (SICK LMS 291): Escanea el frente en un cono estrecho
#      de 20° (±10° del eje frontal) limitado a un rango de 20 metros.
#   3. Validación por SVM (HOG + SVC rbf): Si el LiDAR detecta un obstáculo, se
#      ejecuta de forma síncrona el clasificador SVM mediante Sliding Window
#      y una Región de Interés (ROI) central para validar si es un peatón.
#   4. Comportamiento de Freno Dual:
#      - Peatón (SVM=1) -> Freno de emergencia, luces intermitentes APAGADAS.
#      - Barril  (SVM=0) -> Freno de emergencia, luces intermitentes ENCENDIDAS,
#                           espera hasta que el obstáculo desaparezca.
#
# RESOLUCIÓN DE PROBLEMAS PREVIOS (Para evitar freezes e inestabilidades):
#   - Se deshabilitó 'enablePointCloud()' en el LiDAR, utilizando únicamente 
#     la lectura de rangos en 1D con 'getRangeImage()'. Esto evita que Webots 
#     se trabe (beachball) en macOS.
#   - Se configuró el 'timestep' nativo de Webots sin ningún multiplicador.
#   - La inferencia SVM se ejecuta de forma síncrona y SÓLO cuando el LiDAR detecta
#     un obstáculo a menos de 20 metros. Esto ahorra CPU al no correr SVM por 
#     frame, y elimina la contención del GIL que causaban los hilos (threading).
#   - El cono del LiDAR se redujo a 20° (±10°) para evitar falsos positivos con 
#     peatones situados en las banquetas a 15-20m.
#   - El escaneo SVM aplica una ROI central (35% a 65% de ancho) para ignorar
#     siluetas de edificios o peatones lejanos fuera de la calle.
#
# MODIFICACIONES RESPECTO AL CÓDIGO ORIGINAL DE DETECCIÓN DE VEHÍCULOS:
#   - Dataset: Se cambió del dataset de vehículos a INRIA Person Dataset para 
#     detectar formas humanas (cuerpo completo vertical).
#   - HOG: Se aumentaron las orientaciones a 11 (vs 9 original) para capturar 
#     mejor el contorno humano.
#   - Ventana Deslizante: Ajustada a 64x128 píxeles (aspecto vertical) en lugar 
#     de 64x64 píxeles (autos).
#   - Multi-escala: Agregadas escalas de 1.0x y 2.0x para detectar peatones tanto 
#     cerca como a media distancia.
#   - Incorporación de LiDAR y sistema de lógica dual de luces/frenado.
#
# BIBLIOGRAFÍA Y REFERENCIAS:
#   1. Dalal, N. & Triggs, B. "Histograms of Oriented Gradients for Human Detection", CVPR 2005.
#   2. INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
#   3. Webots Driver and Automobile Library API:
#      https://cyberbotics.com/doc/automobile/driver-library
#   4. Sliding Window con OpenCV y Scikit-Learn:
#      https://medium.com/@ricardo.zuccolo/self-driving-cars-opencv-and-svm-machine-learning-with-scikit-learn-for-vehicle-detection-on-the-bf88860e055a
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
from datetime import datetime

# Rutas de archivo
_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))

# Parámetros del Vehículo y PID
CRUISE_SPEED   = 30       # velocidad crucero en km/h
MAX_ANGLE      = 0.5      # ángulo máximo de giro del volante en radianes
MAX_STEER_RATE = 0.03     # límite de cambio de volante por paso
DEBOUNCE_TIME  = 0.1      # rebote del teclado
Kp, Ki, Kd     = 0.28, 0.01, 0.01

# Filtro de Líneas Hough y Color Amarillo (Espacio HSV)
MIN_ABS_SLOPE  = 0.4
YELLOW_LOW     = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH    = np.array([35, 255, 255], dtype=np.uint8)

# Parámetros del LiDAR Sick LMS 291
LIDAR_CONE_DEG = 20       # Cono estrecho frontal de ±10° (20° totales) para evitar banquetas
LIDAR_MAX_M    = 20.0     # Rango límite de detección: 20 metros

# CONFIGURACIÓN DE RESPALDO PARA MAC (FREEZES DE LIDAR)
# Cambiar a False si Webots se congela (beachball) en macOS al iniciar.
# - En False: LiDAR se deshabilita y el SVM evalúa por cámara constantemente (seguro para Mac).
# - En True: LiDAR se habilita y el SVM es selectivo (ideal para entrega final).
USE_LIDAR      = False     

# Parámetros SVM
HOG_WIN_W     = 64
HOG_WIN_H     = 128
SLIDE_STEP    = 32
CONFIRM_N     = 2         # Frames consecutivos positivos para confirmar peatón
RELEASE_N     = 3         # Frames consecutivos vacíos para liberar freno
HOLD_FRAMES   = 30        # Duración mínima del freno activo
SVM_THRESHOLD = 0.18      # Umbral de la función de decisión del SVM (más restrictivo para evitar falsos positivos)
DETECT_EVERY  = 5         # Frecuencia de escaneo SVM cuando el LiDAR está apagado


# ── SECCIÓN 1: SEGUIMIENTO DE CARRIL (OPENCV + PID) ───────────────────────

def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_gray(display, gray):
    rgb = np.dstack((gray, gray, gray))
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def apply_roi(edges, h, w):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.60)),
        (int(w * 0.65), int(h * 0.60)),
        (int(w * 0.90), h),
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines(lines):
    if lines is None:
        return None
    ok = [l for l in lines
          if l[0][2] != l[0][0]
          and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(ok) if ok else None

def compute_center(lines):
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
        return (np.mean(lx) + np.mean(rx)) / 2.0
    return np.mean(ax) if ax else None


# ── SECCIÓN 2: PROCESAMIENTO DE LIDAR SICK LMS 291 ────────────────────────

def check_lidar_obstacle(lidar):
    """
    Lee las distancias del LiDAR y filtra un cono central de LIDAR_CONE_DEG.
    Devuelve (True, distancia_m) si se detecta un objeto a menos de LIDAR_MAX_M (20m).
    """
    ranges = lidar.getRangeImage()
    if not ranges:
        return False, None
    
    n = len(ranges)
    fov_rad = lidar.getFov()  # FOV total (usualmente 180° o pi radianes)
    
    # Calcular el número de índices correspondientes a la apertura del cono
    cone_rad = math.radians(LIDAR_CONE_DEG)
    half_span = max(1, int(n * (cone_rad / fov_rad) / 2))
    center = n // 2
    
    # Recortar el cono central de la lectura
    start_idx = max(0, center - half_span)
    end_idx = min(n, center + half_span)
    cone_ranges = ranges[start_idx:end_idx]
    
    # Filtrar valores no numéricos o infinitos
    valid_ranges = [r for r in cone_ranges if not (math.isnan(r) or math.isinf(r))]
    
    if valid_ranges:
        min_dist = min(valid_ranges)
        if min_dist < LIDAR_MAX_M:
            return True, min_dist
            
    return False, None


# ── SECCIÓN 3: DETECCIÓN DE PEATONES POR SVM (SLIDING WINDOW + ROI) ───────

def svm_detect_pedestrian(bgr_image, model):
    """
    Realiza una búsqueda por Sliding Window sobre una ROI de la imagen.
    Devuelve True si el clasificador SVM confirma la presencia de un peatón.
    """
    h, w = bgr_image.shape[:2]
    
    # ROI: Recortar cielo (arriba de 30%) e ignorar cofre del vehículo (abajo del 80%)
    roi = bgr_image[int(h * 0.30): int(h * 0.80), :]
    
    # Escaneamos a dos escalas para cubrir diferentes distancias del peatón
    for scale in (1.0, 2.0):
        scaled = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)))
        sh, sw = scaled.shape[:2]
        if sh < HOG_WIN_H or sw < HOG_WIN_W:
            continue
            
        # ROI Horizontal: Concentrar el escaneo en el carril central (40% a 60%)
        x0 = int(sw * 0.40)
        x1 = int(sw * 0.60)
        ystep = max(16, HOG_WIN_H // 4)
        
        for y in range(0, sh - HOG_WIN_H + 1, ystep):
            for x in range(x0, min(x1, sw - HOG_WIN_W + 1), SLIDE_STEP):
                # Extraer ventana, convertir a gris y calcular descriptor HOG
                win = cv2.cvtColor(scaled[y:y+HOG_WIN_H, x:x+HOG_WIN_W], cv2.COLOR_BGR2GRAY)
                feat = hog(win, orientations=11, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), transform_sqrt=False,
                           feature_vector=True)
                
                # Clasificar con SVM usando la función de decisión continua
                score = model.decision_function([feat])[0]
                if score >= SVM_THRESHOLD:
                    return True  # Peatón validado -> Salida rápida
                    
    return False


# ── SECCIÓN 4: CONTROLADOR PRINCIPAL ──────────────────────────────────────

def main():
    # Cargar modelo SVM
    model_path = os.path.normpath(MODEL_PATH)
    if os.path.exists(model_path):
        svm_model = joblib.load(model_path)
        print(f"[OK] Modelo SVM cargado desde {model_path}")
    else:
        svm_model = None
        print(f"[AVISO] Modelo no encontrado en {model_path} — Corriendo sin SVM")

    # Inicializar Vehículo y Driver
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())  # Timestep nativo evitamos freeze

    # Dispositivos
    camera = driver.getDevice("camera")
    camera.enable(timestep)

    display = driver.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Inicializar LiDAR Sick LMS 291 (si está activado)
    lidar = None
    if USE_LIDAR:
        try:
            lidar = driver.getDevice("Sick LMS 291")
            if lidar:
                lidar.enable(timestep)
                print("[OK] LiDAR Sick LMS 291 habilitado")
            else:
                print("[AVISO] Dispositivo LiDAR no encontrado en el vehículo")
        except Exception as e:
            print(f"[ERROR] No se pudo inicializar el LiDAR: {e}")
            lidar = None
    else:
        print("[INFO] LiDAR deshabilitado por configuración (Mac Workaround)")

    dw, dh = display.getWidth(), display.getHeight()
    setpoint = dw / 2.0

    # Variables de Control y Estados
    integral, prev_err, prev_t = 0.0, 0.0, time.time()
    steering = 0.0
    no_line_frames = 0
    frame_cnt = 0
    
    # Contadores de filtrado temporal para robustez
    pos_streak = 0
    neg_streak = 0
    brake_hold = 0
    threat = 'none'  # 'none', 'barrel', 'pedestrian'
    
    # Variables de diagnóstico de visualización
    last_svm_score = 0.0
    last_lidar_dist = 0.0

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("Controlador AG PID + LiDAR + SVM Pedestrian Iniciado")

    while driver.step() != -1:
        t = time.time()
        dt = max(t - prev_t, 1e-3)
        frame_cnt += 1

        # ── 1. Lectura y preparación de Imagen ──────────────────────────────
        image = get_image(camera)
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 2. Detección de Obstáculo por LiDAR ──────────────────────────────
        lidar_alert = False
        last_lidar_dist = 0.0
        if lidar is not None:
            lidar_alert, last_lidar_dist = check_lidar_obstacle(lidar)

        # ── 3. Clasificación SVM (Síncrona y Selectiva) ──────────────────────
        # Si el LiDAR está encendido, solo escaneamos con SVM cuando detecta obstáculos.
        # Si el LiDAR está apagado (Mac workaround), escaneamos cada DETECT_EVERY frames.
        pedestrian_detected = False
        should_scan_svm = False
        if svm_model:
            if lidar is not None:
                should_scan_svm = lidar_alert
            else:
                should_scan_svm = (frame_cnt % DETECT_EVERY == 0)

        if should_scan_svm:
            pedestrian_detected = svm_detect_pedestrian(bgr, svm_model)
            
            if pedestrian_detected:
                pos_streak = min(pos_streak + 1, CONFIRM_N + 2)
                neg_streak = 0
            else:
                neg_streak = min(neg_streak + 1, RELEASE_N + 2)
                pos_streak = max(pos_streak - 1, 0)
        else:
            # Si no se escaneó, decrementamos progresivamente las rachas si el peligro pasó
            if lidar is not None:
                pos_streak = max(pos_streak - 1, 0)
                neg_streak = min(neg_streak + 1, RELEASE_N + 2)

        # ── 4. Lógica de Determinación de Amenaza Dual ────────────────────────
        # Peatón confirmado por SVM
        if pos_streak >= CONFIRM_N:
            threat = 'pedestrian'
            brake_hold = HOLD_FRAMES
            neg_streak = 0
        # Barril (LiDAR detectó, pero el SVM no validó persona tras suficientes frames)
        elif lidar_alert and neg_streak >= RELEASE_N:
            threat = 'barrel'
            brake_hold = HOLD_FRAMES
        # Sin peligro
        elif neg_streak >= RELEASE_N and brake_hold <= 0:
            threat = 'none'
            pos_streak = 0

        # Disminuir el tiempo de espera mínimo de freno
        if brake_hold > 0:
            brake_hold -= 1
            if brake_hold == 0 and not lidar_alert:
                threat = 'none'

        # ── 5. Procesamiento PID de Seguimiento de Carril ─────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        edges = cv2.bitwise_or(cv2.Canny(grey, 50, 150),
                               cv2.Canny(ymask, 50, 150))
        roi_edges = apply_roi(edges, dh, dw)
        lines = filter_lines(cv2.HoughLinesP(roi_edges, 1, np.pi/180, 20,
                                             minLineLength=20, maxLineGap=15))
        center = compute_center(lines)

        # ── 6. Actualización del HUD del Display ──────────────────────────────
        viz = grey.copy()
        if lines is not None:
            for l in lines:
                cv2.line(viz, (l[0][0], l[0][1]), (l[0][2], l[0][3]), 255, 2)
        display_gray(display, viz)

        # Configurar colores y textos según la amenaza
        if threat == 'barrel':
            display.setColor(0xFF6600)  # Naranja
            display.drawText("BARRIL - DETECTADO", 2, 2)
        elif threat == 'pedestrian':
            display.setColor(0xFF0000)  # Rojo
            display.drawText("PEATON - ALERTA", 2, 2)
        else:
            display.setColor(0x00FF00)  # Verde
            display.drawText("SEGUIDOR PID - OK", 2, 2)

        # Métricas HUD
        display.setColor(0xFFFFFF)
        display.drawText(f"V:{CRUISE_SPEED} km/h  St:{steering:.2f}", 2, 12)
        if lidar_alert:
            display.drawText(f"Obstaculo: {last_lidar_dist:.2f} m", 2, 22)
        else:
            display.drawText("Camino libre", 2, 22)
            
        display.drawText(f"Freno Hold: {brake_hold} f", 2, 32)

        # ── 7. Control del Vehículo (Comportamiento Dual) ─────────────────────
        if threat != 'none':
            # FRENADO DE EMERGENCIA
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            
            # Activación diferencial de luces intermitentes (Hazard Flashers)
            # Solo se encienden si la amenaza es el BARRIL, permanecen APAGADAS si es PEATÓN
            driver.setHazardFlashers(threat == 'barrel')
            
            steering = 0.0
            integral = 0.0
            prev_t = t
            continue

        # NAVEGACIÓN NORMAL (PID)
        driver.setHazardFlashers(False)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        if center is not None:
            no_line_frames = 0
            error = (center - setpoint) / setpoint
            integral = max(-0.5, min(0.5, integral + error * dt))
            raw_steer = Kp * error + Ki * integral + Kd * (error - prev_err) / dt
            raw_steer = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steer))
            
            # Limitar la tasa de cambio de la dirección (suavizado)
            steering = max(steering - MAX_STEER_RATE,
                           min(steering + MAX_STEER_RATE, raw_steer))
            prev_err = error
        else:
            no_line_frames += 1
            integral *= 0.6
            prev_err = 0.0
            if no_line_frames > 10:
                steering *= 0.95  # Decaer al centro gradualmente

        driver.setSteeringAngle(steering)
        prev_t = t

        # Captura de pantalla manual al presionar 'A'
        key = keyboard.getKey()
        if key == ord('A'):
            ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            camera.saveImage(os.path.join(os.getcwd(), f"AG_capture_{ts_str}.png"), 1)
            print(f"[A] Captura guardada como AG_capture_{ts_str}.png")

if __name__ == "__main__":
    main()
