# =============================================================================
# simple_controller_pedestrian.py
# Actividad 3.1 — Detección de Peatones con SVM + LiDAR
# Navegación Autónoma — Maestría en Inteligencia Artificial
# =============================================================================
#
# QUÉ HACE ESTE CONTROLADOR
# ─────────────────────────
# El vehículo maneja solo siguiendo la línea amarilla del carril (PID).
# Al mismo tiempo monitorea dos fuentes de peligro:
#
#   LIDAR  → Si detecta un objeto a menos de 20 m enfrente,
#             frena de emergencia y enciende los intermitentes.
#
#   CÁMARA → Busca peatones en la imagen con un modelo SVM entrenado.
#             Si detecta una persona, frena de emergencia.
#
#   Cuando la amenaza desaparece, el auto retoma la marcha automáticamente.
# =============================================================================


# =============================================================================
# SECCIÓN 1 — IMPORTACIONES Y RUTA DEL MODELO
# Cargamos las bibliotecas necesarias: Webots (simulador), visión artificial
# (OpenCV, skimage), el modelo de IA (joblib) y utilidades del sistema.
# También definimos dónde está guardado el modelo SVM entrenado.
# =============================================================================

from controller import Display, Keyboard   # dispositivos de Webots
from vehicle import Car, Driver            # control del vehículo
from skimage.feature import hog            # extractor de características HOG
import numpy as np                         # operaciones numéricas
import cv2                                 # visión artificial (OpenCV)
import joblib                              # cargar/guardar modelos de ML
import math                                # funciones matemáticas (radians, isnan)
import os                                  # manejo de rutas de archivos
import time                                # medir tiempo entre frames
from datetime import datetime              # timestamps para nombres de foto

# Ruta al modelo: sube dos niveles desde controllers/ hasta la carpeta de la actividad
_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib')


# =============================================================================
# SECCIÓN 2 — PARÁMETROS DE CONFIGURACIÓN
# Todos los valores que controlan el comportamiento del auto están aquí.
# Cambiar estos números ajusta el comportamiento sin modificar la lógica.
# =============================================================================

# Vehículo
CRUISE_SPEED  = 30      # velocidad de crucero en km/h
MAX_ANGLE     = 0.5     # máximo ángulo de giro del volante en radianes
DEBOUNCE_TIME = 0.1     # tiempo mínimo entre pulsaciones de teclado (segundos)

# Filtro de líneas (Hough)
MIN_ABS_SLOPE = 0.4     # descarta líneas con pendiente menor — elimina cruces peatonales

# Detección de color amarillo en espacio HSV
# El amarillo de la línea central del carril cae en H≈25, S>80, V>80
YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)   # límite inferior del rango amarillo
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)   # límite superior del rango amarillo

# Detección de bordes (Canny)
CANNY_LOW  = 50    # gradiente por debajo de este valor → no es borde
CANNY_HIGH = 150   # gradiente por encima de este valor → siempre es borde

# Transformada de Hough (detección de líneas)
HOUGH_RHO        = 1             # resolución de distancia: 1 píxel
HOUGH_THETA      = np.pi / 180   # resolución angular: 1 grado
HOUGH_THRESHOLD  = 20            # votos mínimos para aceptar una línea
HOUGH_MIN_LENGTH = 20            # longitud mínima de segmento en píxeles
HOUGH_MAX_GAP    = 15            # brecha máxima para unir dos segmentos

# Controlador PID (seguimiento de carril)
# Kp reacciona al error actual, Ki corrige deriva acumulada, Kd suaviza oscilaciones
Kp = 0.28
Ki = 0.01
Kd = 0.01
MAX_STEER_RATE = 0.03   # máximo cambio de ángulo por frame — evita giros bruscos

# LiDAR
LIDAR_CONE_DEG = 25     # ángulo del cono de detección: ±25° del eje frontal
LIDAR_DANGER_M = 20.0   # si hay algo a menos de esta distancia (metros) → frena

# Detección SVM
HOG_WIN_W      = 64     # ancho de la ventana deslizante en píxeles
HOG_WIN_H      = 128    # alto de la ventana deslizante en píxeles
SLIDE_STEP     = 32     # desplazamiento horizontal entre ventanas
DETECT_EVERY_N = 3      # corre el SVM cada 3 frames para no sobrecargar el CPU
CONFIRM_THRESH = 3      # detecciones consecutivas necesarias para confirmar amenaza
BRAKE_HOLD_FRAMES = 30  # frames que el freno permanece activo tras la última detección


# =============================================================================
# SECCIÓN 3 — FUNCIONES DE VISIÓN (seguimiento de carril)
# Estas funciones procesan la imagen de la cámara para encontrar la línea
# amarilla del carril y calcular hacia dónde debe girar el auto.
# Pipeline: imagen → escala de grises → bordes Canny → ROI → Hough → centro
# =============================================================================

def get_image(camera):
    # Obtiene la imagen actual de la cámara del simulador.
    # La convierte de bytes crudos a un array de píxeles (alto × ancho × 4 canales BGRA).
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def display_gray(display, gray):
    # Muestra una imagen en escala de grises en el display integrado del robot.
    # Webots requiere formato RGB, así que duplicamos el canal gris 3 veces.
    rgb = np.dstack((gray, gray, gray))
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)   # libera memoria del display


def apply_roi(edges, h, w):
    # Define una región de interés trapezoidal en la parte inferior de la imagen.
    # Ignora el cielo y los edificios — solo analiza la zona de carretera
    # que está directamente frente al auto.
    # El trapecio cubre: base ancha abajo (10%-90%) y techo estrecho arriba (35%-65%).
    mask = np.zeros_like(edges)                    # máscara negra del mismo tamaño
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),                        # esquina inferior izquierda
        (int(w * 0.35), int(h * 0.6)),             # esquina superior izquierda
        (int(w * 0.65), int(h * 0.6)),             # esquina superior derecha
        (int(w * 0.90), h)                         # esquina inferior derecha
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)            # aplica la máscara sobre los bordes


def hough_lines(roi_edges):
    # Busca segmentos de línea recta dentro de la región de interés.
    # Hough transforma la imagen de bordes en líneas matemáticas detectables.
    return cv2.HoughLinesP(roi_edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                           minLineLength=HOUGH_MIN_LENGTH, maxLineGap=HOUGH_MAX_GAP)


def filter_lines_by_slope(lines):
    # Filtra líneas casi horizontales (pendiente baja) que no son el carril.
    # Los cruces peatonales y marcas transversales tienen pendiente ≈ 0
    # y confundirían el cálculo del centro del carril.
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:                               # línea vertical — evita división por cero
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) >= MIN_ABS_SLOPE:            # solo acepta líneas suficientemente inclinadas
            filtered.append(line)
    return np.array(filtered) if filtered else None


def compute_lane_center(lines):
    # Calcula la posición horizontal del centro del carril.
    # Clasifica cada línea como borde izquierdo (pendiente negativa) o derecho (positiva).
    # Si hay líneas de ambos lados: centro = promedio entre ellas.
    # Si solo hay un lado: usa ese lado como referencia.
    if lines is None:
        return None
    left_x, right_x, all_x = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid   = (x1 + x2) / 2                     # punto medio horizontal del segmento
        all_x.append(mid)
        (left_x if slope < 0 else right_x).append(mid)   # clasifica por lado
    if left_x and right_x:
        return (np.mean(left_x) + np.mean(right_x)) / 2.0   # centro entre ambos bordes
    return np.mean(all_x) if all_x else None       # fallback: promedio de todo lo detectado


# =============================================================================
# SECCIÓN 4 — DETECCIÓN DE OBSTÁCULOS CON LIDAR
# El sensor Sick LMS 291 escanea el entorno con un rayo láser.
# Filtramos solo el cono frontal (±25°) y verificamos si algo está muy cerca.
# =============================================================================

def check_lidar_barrel(lidar):
    # Lee todas las distancias del sensor láser (un valor por ángulo barrido).
    # Selecciona solo los rayos que apuntan hacia adelante (±25° del centro).
    # Si la distancia mínima en ese cono es menor a LIDAR_DANGER_M → hay obstáculo.
    ranges = lidar.getRangeImage()                 # lista de distancias en metros
    if not ranges:
        return False
    n         = len(ranges)                        # total de rayos del sensor
    fov_rad   = lidar.getFov()                     # ángulo total de visión del sensor
    cone_rad  = math.radians(LIDAR_CONE_DEG)       # convertimos 25° a radianes
    half_span = max(1, int(n * (cone_rad / fov_rad)))   # índices que cubre el cono
    center    = n // 2                             # rayo central = dirección frontal
    forward   = [                                  # filtra valores inválidos (nan, inf)
        r for r in ranges[max(0, center - half_span): center + half_span]
        if not (math.isnan(r) or math.isinf(r))
    ]
    return bool(forward) and min(forward) < LIDAR_DANGER_M


# =============================================================================
# SECCIÓN 5 — DETECCIÓN DE PEATONES CON HOG + SVM
# La cámara toma una foto y la recorremos con una ventana deslizante.
# En cada posición extraemos características HOG (forma/silueta) y el modelo
# SVM decide si hay una persona. Al primer positivo, confirmamos detección.
# =============================================================================

def extract_hog(gray_window):
    # Extrae el descriptor HOG de una ventana de imagen en escala de grises.
    # HOG divide la imagen en celdas y mide la dirección del gradiente (cambio de brillo)
    # en cada una — esto captura la silueta característica de una persona.
    # Resultado: un vector de 924 números que describe la forma del objeto.
    return hog(gray_window,
               orientations=11,          # mide gradientes en 11 direcciones
               pixels_per_cell=(16, 16), # cada celda cubre 16×16 píxeles
               cells_per_block=(2, 2),   # normaliza grupos de 2×2 celdas
               transform_sqrt=False,
               feature_vector=True)      # devuelve un vector plano de 924 valores


def sliding_window_detect(bgr_image, svm_model):
    # Recorre la imagen completa desplazando una ventana de 64×128 píxeles.
    # En cada posición: extrae HOG → consulta al SVM → ¿es una persona?
    # Para en cuanto encuentra la primera detección positiva (early exit)
    # para no gastar tiempo procesando el resto de la imagen.
    h, w   = bgr_image.shape[:2]
    y_step = max(16, HOG_WIN_H // 4)               # paso vertical entre filas de ventanas
    for y in range(0, h - HOG_WIN_H + 1, y_step):
        for x in range(0, w - HOG_WIN_W + 1, SLIDE_STEP):
            win      = bgr_image[y: y + HOG_WIN_H, x: x + HOG_WIN_W]   # recorta la ventana
            gray_win = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)            # convierte a gris
            feat     = extract_hog(gray_win)                            # extrae HOG
            if svm_model.predict([feat])[0] == 1:                       # pregunta al SVM
                return True    # persona detectada → sale inmediatamente
    return False               # ninguna ventana dio positivo


# =============================================================================
# SECCIÓN 6 — FUNCIÓN PRINCIPAL (main)
# Inicializa el simulador, carga el modelo y entra en el loop de control.
# Cada iteración del loop representa un frame de la simulación (~32 ms).
# =============================================================================

def main():

    # ── Cargar modelo SVM ─────────────────────────────────────────────────────
    # Intenta cargar el archivo .joblib con el pipeline entrenado (Scaler + SVM).
    # Si no existe, avisa y corre en modo solo-PID sin detección de peatones.
    model_path = os.path.normpath(MODEL_PATH)
    if os.path.exists(model_path):
        svm_model = joblib.load(model_path)        # carga el pipeline completo
        print(f"[OK] Modelo cargado: {model_path}")
    else:
        svm_model = None
        print(f"[ADVERTENCIA] Modelo no encontrado en: {model_path}")
        print("  Ejecuta primero pedestrian_svm_training.ipynb")
        print("  Corriendo en modo PID sin detección de peatones.")

    # ── Inicializar sensores y actuadores ─────────────────────────────────────
    # Crea las instancias del vehículo y habilita todos los dispositivos.
    robot    = Car()                               # instancia del vehículo
    driver   = Driver()                            # control de velocidad y dirección
    timestep = int(robot.getBasicTimeStep())       # duración de cada frame en ms

    camera = robot.getDevice("camera")             # cámara frontal 256×128
    camera.enable(timestep)

    display_img = robot.getDevice("display_image") # pantalla integrada del robot

    keyboard = Keyboard()
    keyboard.enable(timestep)

    lidar = robot.getDevice("Sick LMS 291")        # sensor láser frontal
    lidar.enable(timestep)
    lidar.enablePointCloud()                        # activa la nube de puntos 3D

    # El setpoint es el centro horizontal del display.
    # El PID intentará mantener la línea detectada en esa posición x.
    dw       = display_img.getWidth()              # ancho del display en píxeles
    dh       = display_img.getHeight()             # alto del display en píxeles
    setpoint = dw / 2.0                            # objetivo: centro de la imagen

    # ── Variables de estado del PID ───────────────────────────────────────────
    integral       = 0.0    # acumulado del error en el tiempo (término I)
    previous_error = 0.0    # error del frame anterior (para el término D)
    previous_time  = time.time()
    steering       = 0.0    # ángulo actual del volante
    no_line_frames = 0      # frames consecutivos sin detectar línea

    # ── Variables de estado de detección ──────────────────────────────────────
    # Los scores suben con cada detección y bajan cuando no hay detección.
    # Esto evita frenazos por un solo falso positivo.
    ped_score    = 0    # puntuación acumulada de detección de peatón
    barrel_score = 0    # puntuación acumulada de detección de barril
    brake_hold   = 0    # frames restantes de frenazo activo
    frame_count  = 0    # contador global de frames

    last_press = {}

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("Controlador iniciado — PID + LiDAR + SVM peatones")
    print("A: captura imagen")

    # =========================================================================
    # LOOP PRINCIPAL — se ejecuta una vez por frame del simulador (~32 ms)
    # Orden de operaciones cada frame:
    #   1. Captura imagen de cámara
    #   2. Consulta LiDAR → ¿barril?
    #   3. Cada 3 frames: ventana deslizante → ¿peatón?
    #   4. Pipeline PID: HSV → Canny → ROI → Hough → centro del carril
    #   5. Actualiza display
    #   6. Aplica freno o PID según el estado actual
    # =========================================================================
    while robot.step() != -1:
        current_time = time.time()
        dt           = max(current_time - previous_time, 1e-3)  # tiempo entre frames
        frame_count += 1

        # ── 1. Captura y preprocesamiento de imagen ───────────────────────────
        image   = get_image(camera)                             # imagen BGRA de la cámara
        bgr     = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)      # elimina canal alfa
        resized = cv2.resize(bgr, (dw, dh))                    # ajusta al tamaño del display
        grey    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)    # convierte a escala de grises

        # ── 2. LiDAR: detección de barril ─────────────────────────────────────
        # Cada frame consultamos el sensor. El score sube/baja gradualmente
        # para requerir detecciones consistentes antes de frenar.
        if check_lidar_barrel(lidar):
            barrel_score = min(barrel_score + 1, CONFIRM_THRESH + 2)  # sube con límite
        else:
            barrel_score = max(barrel_score - 1, 0)                   # baja hasta cero

        # ── 3. SVM: detección de peatón (cada DETECT_EVERY_N frames) ──────────
        # No corremos el SVM todos los frames porque es computacionalmente costoso.
        if svm_model is not None and frame_count % DETECT_EVERY_N == 0:
            if sliding_window_detect(resized, svm_model):
                ped_score = min(ped_score + 1, CONFIRM_THRESH + 2)
            else:
                ped_score = max(ped_score - 1, 0)

        # Confirmamos amenaza solo si el score supera el umbral
        barrel_active = barrel_score >= CONFIRM_THRESH
        ped_active    = ped_score    >= CONFIRM_THRESH

        # Si hay amenaza, reiniciamos el contador de frenazo
        if barrel_active or ped_active:
            brake_hold = BRAKE_HOLD_FRAMES

        # ── 4. Pipeline PID: detección de carril ──────────────────────────────
        # Segmentamos el amarillo con HSV, detectamos bordes con Canny,
        # aplicamos la ROI para ignorar el cielo y encontramos líneas con Hough.
        hsv          = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        yellow_mask  = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)  # máscara de color amarillo
        canny        = cv2.bitwise_or(
                           cv2.Canny(grey,        CANNY_LOW, CANNY_HIGH),   # bordes generales
                           cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH))   # bordes amarillos
        roi_edges    = apply_roi(canny, dh, dw)                   # recorta zona de carretera
        lines        = filter_lines_by_slope(hough_lines(roi_edges))        # filtra líneas
        lane_center  = compute_lane_center(lines)                 # posición del centro del carril

        # ── 5. Actualizar display del robot ───────────────────────────────────
        # Muestra la imagen en escala de grises con las líneas detectadas superpuestas.
        # El HUD indica velocidad, ángulo y estado actual.
        frame_viz = grey.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame_viz, (x1, y1), (x2, y2), 255, 2)   # dibuja líneas en blanco
        display_gray(display_img, frame_viz)

        if barrel_active:
            estado = "BARRIL"
            color  = 0xFF4400   # naranja
        elif ped_active:
            estado = "PEATON"
            color  = 0xFF0000   # rojo
        else:
            estado = "PID"
            color  = 0xFFFFFF   # blanco

        display_img.setColor(color)
        display_img.drawText(f"V:{CRUISE_SPEED}km/h",  2, 2)
        display_img.drawText(f"St:{steering:.3f}r",    2, 12)
        display_img.drawText(estado,                   2, 22)

        # ── 6. Salida de control: freno o PID ─────────────────────────────────
        if brake_hold > 0:
            # FRENAZO DE EMERGENCIA
            # Reduce el contador cada frame. Mientras sea > 0 el auto permanece frenado.
            brake_hold -= 1
            driver.setCruisingSpeed(0)          # velocidad objetivo = 0
            driver.setBrakeIntensity(1.0)        # freno al máximo
            driver.setHazardFlashers(barrel_active)  # intermitentes solo si es barril
            steering = 0.0                       # endereza el volante
            integral = 0.0                       # resetea el acumulado del PID
        else:
            # SEGUIMIENTO DE CARRIL CON PID
            driver.setHazardFlashers(False)
            driver.setBrakeIntensity(0.0)
            driver.setCruisingSpeed(CRUISE_SPEED)

            if lane_center is not None:
                # Línea visible: calcula error y ajusta el volante
                no_line_frames = 0
                error     = (lane_center - setpoint) / setpoint  # error normalizado [-1, 1]
                integral += error * dt
                integral  = max(-0.5, min(0.5, integral))        # límite anti-saturación
                raw_steer = (Kp * error
                             + Ki * integral
                             + Kd * (error - previous_error) / dt)
                raw_steer = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steer))
                # Rate limiter: el volante no puede girar más de MAX_STEER_RATE por frame
                steering  = max(steering - MAX_STEER_RATE,
                                min(steering + MAX_STEER_RATE, raw_steer))
                previous_error = error
            else:
                # Sin línea (intersección): mantiene ángulo y decae a recto gradualmente
                no_line_frames += 1
                integral       *= 0.6        # reduce el acumulado para evitar deriva
                previous_error  = 0.0
                if no_line_frames > 10:
                    steering *= 0.95         # decae suavemente hacia recto

            driver.setSteeringAngle(steering)

        previous_time = current_time

        # ── Teclado: captura de imagen ────────────────────────────────────────
        # Presiona A para guardar una foto con timestamp en el directorio actual.
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and current_time - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = current_time
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"Imagen guardada: {ts}.png")


if __name__ == "__main__":
    main()
