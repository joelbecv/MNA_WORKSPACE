# =============================================================================
# ACTIVIDAD 4.2 — Evasión de Obstáculos con Wall-Following
# Maestría en Inteligencia Artificial Aplicada — Navegación Autónoma
# Tecnológico de Monterrey · MR4010
#
# Integrante: Joel Arturo Becerril Balderas  (A01797427)
#
# Herramienta de IA utilizada:
#   Anthropic. (2026). Claude Sonnet 4.6 [Modelo de lenguaje grande],
#   utilizado para generación de código, depuración y optimización.
#   https://claude.ai/claude-code
#
# DESCRIPCIÓN GENERAL
# -------------------
# Controlador para vehículo BMW en Webots R2023b que implementa:
#   1. Seguimiento de carril central con controlador PID (base: Actividad 2.1)
#   2. Detección de autobús con nodo Recognition de la cámara
#   3. Reducción de velocidad al confirmar el autobús (PASO 3 rúbrica)
#   4. Evasión por wall-following derecha con sensores de distancia (PASO 4)
#   5. Recuperación de orientación con giroscopio (PASO 5)
#   6. Retorno al seguimiento de línea de forma autónoma
#
# MÁQUINA DE ESTADOS
# ------------------
#   LINE_FOLLOW  →  WALL_FOLLOW  →  REORIENT  →  RECENTER  →  LINE_FOLLOW
# =============================================================================

from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
import time
import os

# =============================================================================
# SECCIÓN 1 — PARÁMETROS GLOBALES
# Todos los valores calibrables están concentrados aquí para facilitar ajuste.
# =============================================================================

MAX_ANGLE     = 0.5       # rad: ángulo máximo del volante (límite físico del BMW)
DEBOUNCE_TIME = 0.1       # s: tiempo mínimo entre pulsaciones de teclado

# ── PASO BASE: PID seguidor de línea (portado de Actividad 2.1) ───────────────
# Velocidad de crucero normal durante el seguimiento de carril
SPEED_FOLLOW  = 30        # km/h

# Ganancias PID calibradas para error normalizado en rango [-1, +1]
# Kp: respuesta proporcional al error actual — valor mayor = reacción más rápida
# Ki: corrige error acumulado (deriva) — valor bajo para evitar oscilaciones
# Kd: amortigua cambios bruscos — suaviza la respuesta ante variaciones rápidas
Kp = 0.28
Ki = 0.01
Kd = 0.01

DEFAULT_ANGLE = 0.0       # rad: ángulo inicial del volante

# Umbrales del detector de bordes Canny
# CANNY_LOW:  umbral inferior — borde débil, se acepta si conecta a borde fuerte
# CANNY_HIGH: umbral superior — borde fuerte, siempre se acepta
CANNY_LOW   = 50
CANNY_HIGH  = 150

# Parámetros de la Transformada de Hough probabilística (HoughLinesP)
# HOUGH_RHO:   resolución de distancia en la acumulación (1 pixel)
# HOUGH_THETA: resolución angular (1 grado en radianes)
# HOUGH_THRESH: votos mínimos para que una línea sea aceptada
# HOUGH_MIN:   longitud mínima de un segmento de línea en píxeles
# HOUGH_GAP:   brecha máxima entre segmentos para unirlos en uno solo
HOUGH_RHO    = 1
HOUGH_THETA  = np.pi / 180
HOUGH_THRESH = 20
HOUGH_MIN    = 20
HOUGH_GAP    = 15

# Filtro de pendiente: descarta líneas casi horizontales (rayas de cebra, ruido)
# Una línea con |pendiente| < 0.4 es casi horizontal → se descarta
MIN_ABS_SLOPE  = 0.4

# Rate limiter: máximo cambio de volante por frame
# Evita sacudidas bruscas que desestabilicen el vehículo
MAX_STEER_RATE = 0.03     # rad/frame

# Hold/decay sin línea: cuántos frames se mantiene el último steering antes de decaer
# A 30 km/h, 10 frames (~0.1s) es suficiente para cruzar una cebra sin desviarse
NO_LINE_HOLD   = 10

# ── PASO 2 RÚBRICA: LiDAR — configuración de sectores ───────────────────────
# El sensor Sick LMS 291 escanea 181 rayos en un arco de 180°
# Índice 0 = extremo izquierdo (-90°), índice 90 = frente (0°), índice 180 = derecha (+90°)
#
# Sector FRONTAL: ±20° del frente → detecta autobús adelante
# Sector LATERAL DERECHO PURO: 150-180 (65°-90° del frente)
#   → Se usa para wall-following cuando el bus está al costado del vehículo
#   NOTA: El sector original 135-165° (45°-75°) no detectaba el bus al costado puro;
#         se cambió a 150-180° para cubrir la zona lateral real donde aparece el bus.
BUS_LIDAR_THRESH = 14.5   # m: distancia de umbral para iniciar evasión (PASO 3 rúbrica)
                           # Se usa 14.5m porque el Recognition pierde el bus a <12m
LIDAR_FOV_DEG    = 20     # grados del sector frontal

# ── PASO 3 RÚBRICA: Reducción de velocidad a la mitad ────────────────────────
SPEED_EVADE   = 15        # km/h = SPEED_FOLLOW / 2 (exigido explícitamente por la rúbrica)

# ── PASO 4 RÚBRICA: Wall-following — parámetros del controlador P ────────────
# WALL_TARGET: distancia objetivo al costado del autobús durante la evasión
#   Calibrado experimentalmente: 2.9m da margen suficiente sin exceder el carril contrario
WALL_TARGET   = 2.9       # m

# KP_WALL: ganancia del controlador P de pared
#   error = right_dist - WALL_TARGET
#   steering = KP_WALL × error
#   Valor bajo (0.10) para respuesta suave, evita oscilaciones laterales
KP_WALL       = 0.10

# DS_CLEAR_DIST: distancia mínima del sensor trasero para considerar el bus superado
#   El sensor ds_right_rear tiene rango 5m; >4.8m significa "no hay obstáculo en 5m"
DS_CLEAR_DIST = 4.8       # m

# DS_ENGAGE_DIST: umbral para detectar que el bus ya está al costado (no al frente)
#   Phase A → Phase B: cuando right_dist < DS_ENGAGE_DIST el bus está al costado
DS_ENGAGE_DIST = 4.5      # m

# Filtros de detección (anti falsos positivos)
# BUS_MIN_PX_AREA: área mínima en px² para considerar un objeto como bus
#   Valor bajo (400) permite detectar el bus desde lejos (>14m)
BUS_MIN_PX_AREA = 400
# BUS_CONFIRM_FRAMES: frames consecutivos con bus válido para confirmar detección
#   Con color exacto (tolerancia 0.05), 1 frame es suficiente sin falsos positivos
BUS_CONFIRM_FRAMES = 1

# ── PASO 5 RÚBRICA: Recuperación de orientación con giroscopio ───────────────
SPEED_REORIENT = 20       # km/h durante la recuperación de heading

# KP_HEADING: ganancia del controlador P de heading
#   heading_error = saved_heading - heading_actual
#   steering = KP_HEADING × heading_error
KP_HEADING     = 1.0

# HEADING_TOL: error angular mínimo para declarar heading recuperado
#   0.08 rad ≈ 4.6° de tolerancia — suficiente para retomar el seguimiento de línea
HEADING_TOL    = 0.08     # rad

# ── Colores de reconocimiento de cada autobús en el mundo ────────────────────
# El mundo incluye 4 autobuses, cada uno con su propio recognitionColor.
# Estos valores se extraen del archivo .wbt y deben coincidir exactamente.
# Tolerancia de comparación: ±0.05 por canal RGB (evita falsos positivos de asfalto)
BUS_COLOR_MAP = {
    "vehicle(1)": (0.0313726, 0.121569, 0.419608),  # azul marino
    "vehicle(2)": (1.000000,  0.000000, 0.000000),  # rojo puro
    "vehicle(3)": (0.862745,  0.541176, 0.866667),  # lavanda
    "vehicle(4)": (0.180392,  0.760784, 0.494118),  # verde
}

# ── Identificadores de estados de la máquina de estados ──────────────────────
STATE_LINE_FOLLOW = 0     # Seguimiento de carril PID (estado base)
STATE_WALL_FOLLOW = 1     # Evasión lateral — wall-following derecha
STATE_REORIENT    = 2     # Recuperación de heading con giroscopio
STATE_RECENTER    = 3     # Búsqueda de línea amarilla para retomar carril
STATE_NAMES = {0: "LINE_FOLLOW", 1: "WALL_FOLLOW", 2: "REORIENT", 3: "RECENTER"}


# =============================================================================
# SECCIÓN 2 — FUNCIONES DE PROCESAMIENTO DE IMAGEN
# Pipeline H2: HSV amarillo → Canny → ROI trapezoidal → HoughLinesP → PID
# =============================================================================

def get_image(camera):
    """
    Lee la imagen cruda de la cámara Webots y la convierte a array NumPy BGRA.

    Parámetros:
        camera: objeto Camera de Webots (128×128 px, BGRA)

    Retorna:
        numpy.ndarray shape (height, width, 4) dtype uint8
    """
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def preprocess(image, proc_w, proc_h):
    """
    Pipeline H2: filtra solo píxeles amarillos y aplica detector de bordes Canny.

    Por qué solo amarillo:
        - Las líneas del carril son amarillas en este mundo Webots
        - Filtrar por color antes de Canny elimina ruido de asfalto gris,
          edificios blancos y rayas blancas de cebra que causaban líneas falsas

    Rango HSV amarillo: H∈[15,35], S∈[80,255], V∈[80,255]
        - H (tono): 15-35° corresponde a amarillo en el espectro HSV
        - S (saturación): >80 descarta blanco y gris
        - V (valor): >80 descarta negro y sombras oscuras

    Parámetros:
        image   : array BGRA de la cámara
        proc_w  : ancho del display en px (dimensión de procesamiento)
        proc_h  : alto del display en px

    Retorna:
        edges       : imagen binaria de bordes (Canny sobre máscara amarilla)
        yellow_mask : máscara binaria de píxeles amarillos (para yellow_frac)
    """
    bgr         = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    resized     = cv2.resize(bgr, (proc_w, proc_h))
    hsv         = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    # Canny sobre la máscara ya filtrada — solo detecta bordes de líneas amarillas
    edges       = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)
    return edges, yellow_mask


def apply_roi(edges, height, width):
    """
    Aplica una ROI trapezoidal que descarta el cielo y los bordes del encuadre.

    Geometría del trapecio (H2):
        - Base inferior: 10% a 90% del ancho (full road width)
        - Techo:         35% a 65% del ancho, a 60% de la altura
        - Esto incluye el carril pero excluye edificios y cielo

    Por qué trapezoidal y no rectangular:
        Una ROI rectangular incluiría las esquinas superiores donde aparecen
        edificios y postes que generan líneas falsas. El trapecio sigue la
        perspectiva natural de la carretera.

    Parámetros:
        edges  : imagen binaria de bordes
        height : alto de la imagen en px
        width  : ancho de la imagen en px

    Retorna:
        numpy.ndarray con bordes solo dentro del trapecio
    """
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (int(width * 0.10), height),           # esquina inferior izquierda
        (int(width * 0.35), int(height * 0.6)), # esquina superior izquierda
        (int(width * 0.65), int(height * 0.6)), # esquina superior derecha
        (int(width * 0.90), height),           # esquina inferior derecha
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(roi_edges):
    """
    Detecta segmentos de línea con la Transformada de Hough Probabilística.

    HoughLinesP vs HoughLines:
        HoughLinesP es más eficiente y devuelve segmentos (x1,y1,x2,y2)
        en lugar de líneas infinitas, lo que facilita calcular el centro.

    Parámetros usados (definidos en sección de parámetros):
        rho=1 px, theta=1°, threshold=20 votos, minLength=20px, maxGap=15px

    Parámetros:
        roi_edges: imagen binaria con bordes dentro de la ROI

    Retorna:
        Lista de segmentos [[x1,y1,x2,y2], ...] o lista vacía si no detecta nada
    """
    lines = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH,
        minLineLength=HOUGH_MIN,
        maxLineGap=HOUGH_GAP,
    )
    if lines is None:
        return []
    return lines.reshape(-1, 4)


def filter_lines_by_slope(lines):
    """
    Descarta líneas casi horizontales usando el filtro de pendiente H2.

    Por qué filtrar por pendiente:
        Las rayas de la cebra peatonal son horizontales (pendiente ≈ 0).
        Al usar solo amarillo en preprocess() estas ya se eliminan, pero el
        filtro actúa como segunda capa de seguridad para cualquier artefacto
        horizontal que pase el filtro de color.

    Una línea con |pendiente| < MIN_ABS_SLOPE (0.4) es descartada.
    Las líneas de carril típicamente tienen pendiente ≥ 0.5 en perspectiva.

    Parámetros:
        lines: lista de segmentos [x1,y1,x2,y2]

    Retorna:
        Lista filtrada con solo líneas de suficiente inclinación
    """
    filtered = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:             # línea vertical: pendiente infinita, se acepta
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            continue             # descarta líneas casi horizontales
        filtered.append([x1, y1, x2, y2])
    return filtered


def compute_lane_center(lines):
    """
    Calcula el centro del carril promediando los puntos X de líneas izquierda y derecha.

    Clasificación izquierda/derecha por signo de pendiente:
        - Pendiente negativa (y decrece cuando x crece) → línea izquierda del carril
        - Pendiente positiva → línea derecha del carril

    En perspectiva, las líneas del carril convergen hacia el punto de fuga:
        izquierda tiene pendiente negativa, derecha tiene pendiente positiva.

    Si solo hay un lado visible (curva): se usa el promedio de todos los puntos.
    Si no hay líneas: retorna None → activa hold/decay sin línea.

    Parámetros:
        lines: lista de segmentos filtrados

    Retorna:
        float: coordenada X del centro del carril en píxeles, o None
    """
    if not lines:
        return None
    left_pts  = []
    right_pts = []
    all_pts   = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        all_pts.extend([x1, x2])
        if slope < 0:
            left_pts.extend([x1, x2])   # línea izquierda
        else:
            right_pts.extend([x1, x2])  # línea derecha
    if left_pts and right_pts:
        # Centro = promedio entre el promedio izquierdo y el promedio derecho
        return (np.mean(left_pts) + np.mean(right_pts)) / 2.0
    if all_pts:
        # Solo un lado visible → usar promedio de todos los puntos detectados
        return np.mean(all_pts)
    return None


def identify_bus_color(obj, debug=False):
    """
    PASO 1 RÚBRICA: Identifica si un RecognitionObject es un autobús conocido.

    El nodo Recognition de Webots asigna un recognitionColor a cada objeto.
    Esta función compara el color del objeto contra el BUS_COLOR_MAP con
    tolerancia ±0.05 por canal RGB.

    Por qué tolerancia 0.05 y no valor exacto:
        Webots reporta el color con precisión flotante que puede tener
        pequeñas variaciones. La tolerancia 0.05 absorbe esas variaciones
        sin introducir falsos positivos (el asfalto gris 0.2,0.2,0.2
        está a 0.18 del autobús azul — suficiente separación).

    Nota API Webots R2023b:
        getColors() devuelve un LP_c_double (ctypes), no una lista de Python.
        No tiene len() — se usa getNumberOfColors() para verificar que hay datos.

    Parámetros:
        obj   : RecognitionObject de camera.getRecognitionObjects()
        debug : si True, imprime info del objeto en consola

    Retorna:
        tuple (nombre_bus: str, color_str: str)
        nombre_bus = "desconocido" si no coincide con ningún autobús
    """
    n = obj.getNumberOfColors()
    if n < 1:
        return "desconocido", "N/A"
    colors = obj.getColors()            # LP_c_double indexable [r, g, b, r2, g2, b2, ...]
    r, g, b = colors[0], colors[1], colors[2]
    color_str = f"RGB({r:.3f},{g:.3f},{b:.3f})"
    if debug:
        model = obj.getModel()
        sz    = obj.getSizeOnImage()
        pos   = obj.getPositionOnImage()
        print(f"[DBG] modelo='{model}' color={color_str} size={sz[0]}x{sz[1]} pos=({pos[0]},{pos[1]})")
    for name, (cr, cg, cb) in BUS_COLOR_MAP.items():
        # Comparación canal a canal con tolerancia ±0.05
        if abs(r - cr) < 0.05 and abs(g - cg) < 0.05 and abs(b - cb) < 0.05:
            return name, color_str
    return "desconocido", color_str


# =============================================================================
# SECCIÓN 3 — FUNCIONES DE VISUALIZACIÓN
# =============================================================================

def display_with_lines(display_dev, edges, lines, width, height):
    """
    Visualización de diagnóstico: muestra bordes ROI + segmentos Hough detectados.

    Permite verificar visualmente qué está "viendo" el PID en cada frame.
    Los bordes aparecen en gris (60% intensidad) y las líneas detectadas en blanco.

    Parámetros:
        display_dev : dispositivo Display de Webots
        edges       : imagen de bordes (ROI aplicada)
        lines       : lista de segmentos Hough a dibujar
        width, height: dimensiones del display
    """
    line_layer = np.zeros((height, width), dtype=np.uint8)
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_layer, (x1, y1), (x2, y2), 255, 2)
    debug = cv2.addWeighted(edges, 0.6, line_layer, 1.0, 0)
    rgb   = np.dstack((debug, debug, debug))
    ref   = display_dev.imageNew(rgb.tobytes(), Display.RGB, width=width, height=height)
    display_dev.imagePaste(ref, 0, 0, False)
    display_dev.imageDelete(ref)


# =============================================================================
# SECCIÓN 4 — MAIN: Inicialización y máquina de estados principal
# =============================================================================

def main():
    # ── Inicialización del robot y driver ────────────────────────────────────
    robot    = Car()
    driver   = Driver()
    # getBasicTimeStep() devuelve el paso de simulación en ms (típicamente 10ms)
    # Se convierte a int porque la API espera entero para enable()
    timestep = int(robot.getBasicTimeStep())
    dt       = timestep / 1000.0   # paso en segundos para integración (0.01s a 10ms)

    # ── PASO 1 RÚBRICA: Cámara con nodo Recognition ──────────────────────────
    # camera.enable(timestep)           → activa la cámara para capturar imágenes
    # camera.recognitionEnable(timestep) → activa el nodo Recognition para detectar
    #                                      objetos con recognitionColor definido en el .wbt
    # IMPORTANTE: ambos deben estar inicializados con el mismo timestep
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    width    = camera.getWidth()    # 128 px
    height   = camera.getHeight()  # 128 px
    setpoint = width / 2.0         # centro horizontal de la imagen para el PID

    # ── Display para visualización (HUD) ─────────────────────────────────────
    display = robot.getDevice("display_image")

    # ── Teclado (tecla M para debug: forzar LINE_FOLLOW) ─────────────────────
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── PASO 2 RÚBRICA: LiDAR Sick LMS 291 ──────────────────────────────────
    # El Sick LMS 291 escanea 181 rayos en 180°:
    #   índice 0   = extremo izquierdo (-90° del frente)
    #   índice 90  = frente (0°)
    #   índice 180 = extremo derecho (+90° del frente)
    #
    # Se habilita con sick.enable(timestep) — mismo timestep que la cámara
    # NO llamar sick.enablePointCloud() en LiDAR 2D → congela el simulador
    sick   = robot.getDevice("Sick LMS 291")
    sick.enable(timestep)
    n_rays = sick.getHorizontalResolution()   # 181 rayos

    # ── PASO 3 RÚBRICA: Giroscopio ───────────────────────────────────────────
    # getValues() devuelve [vx, vy, vz] en rad/s en el frame del sensor.
    # En el BMW de Webots R2023b, el eje de yaw (rotación vertical) corresponde
    # al ÍNDICE 1 (eje Y del sensor), no al índice 2 (eje Z) como sería intuitivo.
    # Esto fue validado experimentalmente: usar índice 2 producía integración
    # divergente (heading_error crecía de 0.55 a 0.78 rad sin converger).
    # La rúbrica menciona "eje z" pero en este vehículo específico el yaw
    # se obtiene con getValues()[1].
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    heading       = 0.0    # heading acumulado: integral de yaw rate × dt
    saved_heading = 0.0    # heading guardado al inicio de la evasión (PASO 3)

    # ── PASO 4 RÚBRICA: Sensores de distancia en costado derecho ─────────────
    # Tres sensores de un solo rayo (genéricos) montados en slots del costado derecho.
    # Lookup table: [0, 0, 0,  5, 5, 0] → distancia 0m = valor 0, distancia 5m = valor 5
    # (rango lineal 0-5m)
    # El algoritmo de wall-following lee los tres para determinar:
    #   ds_front: si el bus ya está al costado (Phase A → Phase B)
    #   ds_rear:  si el bus ya fue superado completamente (salida de WALL_FOLLOW)
    ds_front = robot.getDevice("ds_right_front")
    ds_mid   = robot.getDevice("ds_right_mid")
    ds_rear  = robot.getDevice("ds_right_rear")
    ds_front.enable(timestep)
    ds_mid.enable(timestep)
    ds_rear.enable(timestep)

    # ── Variables de estado ───────────────────────────────────────────────────
    state          = STATE_LINE_FOLLOW  # estado inicial: seguir la línea
    integral       = 0.0               # término integral del PID
    prev_error     = 0.0               # error del frame anterior (para derivada)
    steering       = 0.0               # ángulo de volante actual
    no_line_frames = 0                 # contador de frames consecutivos sin línea
    prev_time      = time.time()
    last_press     = {}

    # Variables de control de estados secundarios
    _prev_state      = -1
    _bus_print_cd    = 0      # cooldown para no saturar la consola con prints de bus
    _bus_streak      = 0      # frames consecutivos con bus válido detectado
    _wall_engaged    = False  # True cuando el bus está al costado (Phase B activa)
    _clear_hold      = 0      # buffer de frames rectos tras liberar sensor trasero
    _recenter_frames = 0      # contador de frames en RECENTER (timeout de seguridad)

    # ── Logs de diagnóstico ───────────────────────────────────────────────────
    # act42_diag.log: registra LINE_FOLLOW — útil para diagnóstico del PID
    _log_path = os.path.join(os.path.dirname(__file__), "act42_diag.log")
    _log_file = open(_log_path, "w")
    _log_file.write("frame,sim_t,state,yellow_frac,n_raw,n_filt,cx,err,steer,nolf,lidar,"
                    "n_obj,top_model,top_color,top_area,top_px,bus_str\n")
    # act42_wall.log: registra WALL_FOLLOW y RECENTER — diagnóstico de evasión
    _wall_log_path = os.path.join(os.path.dirname(__file__), "act42_wall.log")
    _wall_log_file = open(_wall_log_path, "w")
    _wall_log_file.write("sim_t,state,lidar_front,lidar_right,dist_rf,dist_rm,dist_rr,steering,engaged\n")
    _frame = 0

    print("[INIT] Actividad 4.2 — Controlador iniciado")
    print(f"[INIT] Umbral LiDAR para evasión: {BUS_LIDAR_THRESH} m  |  Vel. evasión: {SPEED_EVADE} km/h")

    # =========================================================================
    # LOOP PRINCIPAL — se ejecuta cada timestep (10ms por defecto)
    # =========================================================================
    while robot.step() != -1:
        current_time = time.time()
        real_dt = current_time - prev_time
        if real_dt <= 0:
            real_dt = 1e-6
        prev_time = current_time

        # Imprime en consola cada cambio de estado (para el video: indicador visual)
        if state != _prev_state:
            print(f"\n[ESTADO] → {STATE_NAMES[state]}")
            _prev_state = state

        # ── [A] Integración del giroscopio (PASO 3 rúbrica) ─────────────────
        # Se integra en cada frame, no solo durante la evasión, para mantener
        # un heading acumulado preciso desde el inicio de la simulación.
        # gyro.getValues()[1] = yaw rate en rad/s (eje Y del sensor en BMW R2023b)
        heading += gyro.getValues()[1] * dt

        # ── [B] Lectura LiDAR — sectores frontal y lateral ───────────────────
        ranges = list(sick.getRangeImage())

        # Sector FRONTAL: ±LIDAR_FOV_DEG° alrededor del índice central
        # Detecta el autobús mientras está adelante del vehículo
        span        = max(1, int(n_rays * LIDAR_FOV_DEG / 180))
        center      = n_rays // 2
        front_vals  = [r for r in ranges[center - span : center + span] if r < 99.0]
        lidar_dist  = float(min(front_vals)) if front_vals else 99.0

        # Sector LATERAL DERECHO PURO (PASO 4 rúbrica): índices 150-180
        # Cubre 65°-90° a la derecha del frente — zona donde aparece el bus al costado
        # El sector anterior (135-165°) era demasiado frontal y perdía el bus lateral
        r_start     = n_rays * 150 // 180   # índice ~150 = 65° derecha
        r_end       = n_rays                # índice 180 = 90° derecha (puro lateral)
        right_vals  = [r for r in ranges[r_start:r_end] if r < 99.0]
        lidar_right = float(min(right_vals)) if right_vals else 99.0

        # ── [C] Sensores de distancia laterales ──────────────────────────────
        dist_rf = float(ds_front.getValue())   # sensor frontal derecho (0-5m)
        dist_rm = float(ds_mid.getValue())     # sensor medio derecho
        dist_rr = float(ds_rear.getValue())    # sensor trasero derecho

        # ── [D] PASO 1 RÚBRICA: Recognition — detección del autobús ─────────
        bus_in_front = False
        bus_name     = "desconocido"
        color_str    = "N/A"
        objects      = camera.getRecognitionObjects()

        # Captura el objeto más grande en cámara (para el log de diagnóstico)
        top_model, top_color, top_area, top_px = "none", "none", 0, -1
        for _o in objects:
            _si  = _o.getSizeOnImage()
            _a   = _si[0] * _si[1]
            if _a > top_area:
                top_area  = _a
                top_px    = int(_o.getPositionOnImage()[0])
                top_model = _o.getModel()
                _, top_color = identify_bus_color(_o)

        # Filtros para identificar el autobús al frente (evita falsos positivos)
        for obj in objects:
            pos_img  = obj.getPositionOnImage()
            img_size = obj.getSizeOnImage()
            px_area  = img_size[0] * img_size[1]

            # Filtro 1: el objeto debe estar en el 50% central de la imagen
            #   (un bus en el carril de frente estará centrado horizontalmente)
            if abs(pos_img[0] - setpoint) > width * 0.50:
                continue

            # Filtro 2: tamaño mínimo en píxeles para descartar objetos lejanos
            if px_area < BUS_MIN_PX_AREA:
                continue

            # Filtro 3: color del objeto debe coincidir con un bus conocido (±0.05 RGB)
            bus_name, color_str = identify_bus_color(obj)
            if bus_name == "desconocido":
                continue

            bus_in_front = True
            break

        # Acumulador de frames consecutivos con bus válido
        # (BUS_CONFIRM_FRAMES=1 → basta 1 frame para confirmar)
        if bus_in_front:
            _bus_streak += 1
        else:
            _bus_streak = 0

        bus_confirmed = _bus_streak >= BUS_CONFIRM_FRAMES

        # PASO 1 RÚBRICA: print en consola cuando se detecta el bus
        if bus_in_front and _bus_print_cd <= 0:
            print(f"[BUS DETECTADO] {bus_name} | color={color_str} | "
                  f"streak={_bus_streak} | LiDAR={lidar_dist:.1f}m")
            _bus_print_cd = 60   # limita a ~1 print por segundo
        _bus_print_cd -= 1

        # ── [E] Teclado: tecla M para forzar LINE_FOLLOW (debug) ─────────────
        key = keyboard.getKey()
        if key != -1:
            now = time.time()
            if key not in last_press or (now - last_press[key] >= DEBOUNCE_TIME):
                last_press[key] = now
                if key == ord('M'):
                    state = STATE_LINE_FOLLOW
                    integral = 0.0
                    print("[KEY-M] Forzado a LINE_FOLLOW")

        # =====================================================================
        # MÁQUINA DE ESTADOS
        # =====================================================================

        if state == STATE_LINE_FOLLOW:
            # ── PASO 3 RÚBRICA: Condición de activación de evasión ───────────
            # Se activan AMBAS condiciones:
            #   1. bus_confirmed: el Recognition identificó el autobús
            #   2. lidar_dist < BUS_LIDAR_THRESH: LiDAR confirma distancia al bus
            #
            # BUS_LIDAR_THRESH = 14.5m porque el Recognition pierde el bus a <12m.
            # Con 14.5m el trigger se activa a ~14m con bus_str≈6, evitando la
            # zona ciega del Recognition.
            if bus_confirmed and lidar_dist < BUS_LIDAR_THRESH:
                # PASO 3 RÚBRICA: guardar orientación del giroscopio
                # heading es la integral del yaw rate — representa la orientación
                # acumulada desde el inicio de la simulación
                saved_heading = heading
                state         = STATE_WALL_FOLLOW
                integral      = 0.0        # resetear PID al cambiar de estado
                _wall_engaged = False
                _clear_hold   = 0
                _bus_streak   = 0
                print(f"[EVASION] Bus confirmado a {lidar_dist:.1f}m — activando wall-follow")
                print(f"[EVASION] Heading guardado = {saved_heading:.4f} rad")
                continue

            # ── PID seguimiento de línea ─────────────────────────────────────
            proc_w = display.getWidth()
            proc_h = display.getHeight()

            img                = get_image(camera)
            edges, yellow_mask = preprocess(img, proc_w, proc_h)
            roi                = apply_roi(edges, proc_h, proc_w)
            raw_lines          = detect_lines(roi)
            filt_lines         = filter_lines_by_slope(raw_lines)
            lane_center_x      = compute_lane_center(filt_lines)

            # yellow_frac: fracción de píxeles amarillos en el 40% inferior de la imagen
            # Usado por RECENTER para confirmar que el carril fue encontrado
            yellow_frac = float((yellow_mask[int(proc_h * 0.6):] > 0).mean())

            # Guardar imágenes de diagnóstico en frames 5 y 50
            if _frame in (5, 50):
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_edges_{_frame}.png"), edges)
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_roi_{_frame}.png"), roi)
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_yellow_{_frame}.png"), yellow_mask)

            display_with_lines(display, roi, filt_lines, proc_w, proc_h)

            # HUD: muestra estado, LiDAR y bus en el display del vehículo
            # Color rojo = bus detectado, verde = sin bus
            bus_label = bus_name if bus_in_front else "---"
            hud_color = 0xFF0000 if bus_in_front else 0x00FF00
            display.setColor(0x000000)
            display.fillRectangle(0, 0, proc_w, 22)
            display.setColor(hud_color)
            # PASO 2 RÚBRICA: distancia LiDAR impresa en el display
            display.drawText(f"LIDAR:{lidar_dist:.1f}m  BUS:{bus_label}", 2, 2)
            display.setColor(0xFFFFFF)
            display.drawText(f"ST:{STATE_NAMES[state]}  E:{prev_error:.2f}", 2, 12)

            if lane_center_x is not None:
                # Línea detectada: calcular error normalizado y aplicar PID
                no_line_frames = 0
                image_center   = proc_w / 2.0
                # Error normalizado [-1, +1]: positivo = carril a la derecha
                error_norm     = (lane_center_x - image_center) / image_center
                integral      += error_norm * real_dt
                integral       = max(-0.5, min(0.5, integral))  # anti-windup
                derivative     = (error_norm - prev_error) / real_dt
                raw_steering   = Kp * error_norm + Ki * integral + Kd * derivative
                raw_steering   = float(np.clip(raw_steering, -MAX_ANGLE, MAX_ANGLE))
                # Rate limiter: máximo 0.03 rad de cambio por frame (evita sacudidas)
                steering       = max(steering - MAX_STEER_RATE,
                                     min(steering + MAX_STEER_RATE, raw_steering))
                prev_error = error_norm
            else:
                # Sin línea: hold por NO_LINE_HOLD frames, luego decaimiento suave
                no_line_frames += 1
                integral       *= 0.6     # reducir integral para evitar windup
                prev_error      = 0.0
                if no_line_frames > NO_LINE_HOLD:
                    steering *= 0.95      # decay suave hacia recto

            steering = float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))
            driver.setCruisingSpeed(SPEED_FOLLOW)
            driver.setSteeringAngle(steering)

            # Registro CSV para diagnóstico
            lane_cx_log = f"{lane_center_x:.1f}" if lane_center_x is not None else "None"
            err_log     = f"{prev_error:.3f}"    if lane_center_x is not None else "None"
            _log_file.write(
                f"{_frame},{robot.getTime():.2f},{STATE_NAMES[state]},"
                f"{yellow_frac:.3f},{len(raw_lines)},{len(filt_lines)},"
                f"{lane_cx_log},{err_log},"
                f"{steering:.4f},{no_line_frames},{lidar_dist:.2f},"
                f"{len(objects)},{top_model},{top_color},{top_area},{top_px},{_bus_streak}\n"
            )
            _log_file.flush()
            _frame += 1

        elif state == STATE_WALL_FOLLOW:
            # ================================================================
            # PASO 4 RÚBRICA: Wall-following derecha con sensores del costado
            # ================================================================
            # El vehículo mantiene una distancia objetivo al autobús usando
            # el sector lateral del LiDAR + sensores de distancia laterales.
            #
            # Phase A: bus aún no al costado (right_dist > DS_ENGAGE_DIST)
            #   → steering = -0.08 (girar izquierda para rodear el bus)
            #
            # Phase B: bus al costado derecho (right_dist < DS_ENGAGE_DIST)
            #   → P-controller: steering = KP_WALL × (right_dist - WALL_TARGET)
            #   → Si right_dist > WALL_TARGET (demasiado lejos): girar derecha
            #   → Si right_dist < WALL_TARGET (demasiado cerca): girar izquierda
            #
            # PASO 3 (cont.): velocidad reducida a 15 km/h (mitad de 30 km/h)

            # right_dist toma el mínimo entre LiDAR lateral y sensor frontal derecho
            # para garantizar que cualquier obstáculo cercano sea detectado
            right_dist = min(lidar_right, dist_rf)

            if right_dist > DS_ENGAGE_DIST:
                # Phase A: el bus todavía no apareció al costado → girar izquierda
                steering = -0.08
            else:
                # Phase B: bus al costado → P-controller de pared
                error_wall = right_dist - WALL_TARGET
                steering   = float(np.clip(KP_WALL * error_wall, -MAX_ANGLE, MAX_ANGLE))

            # PASO 3 RÚBRICA: velocidad a la mitad durante la evasión
            driver.setCruisingSpeed(SPEED_EVADE)
            driver.setSteeringAngle(steering)

            # Indicador en consola cuando Phase B se activa (bus al costado)
            if not _wall_engaged and right_dist < DS_ENGAGE_DIST:
                _wall_engaged = True
                print(f"[WALL] Bus al costado ({right_dist:.2f}m) — seguimiento activo")

            # Log para diagnóstico de la evasión
            _wall_log_file.write(
                f"{robot.getTime():.3f},WALL_FOLLOW,{lidar_dist:.2f},{lidar_right:.2f},"
                f"{dist_rf:.2f},{dist_rm:.2f},{dist_rr:.2f},{steering:.4f},{int(_wall_engaged)}\n"
            )
            _wall_log_file.flush()

            if int(robot.getTime() * 10) % 10 == 0:
                print(f"[WALL] lidar_r={lidar_right:.2f}m ds_rf={dist_rf:.2f}m "
                      f"rear={dist_rr:.2f}m | steer={steering:+.3f}")

            # ================================================================
            # PASO 5 RÚBRICA: Fin del wall-following
            # ================================================================
            # Condición de salida: sensor trasero derecho > DS_CLEAR_DIST
            #   → el último sensor lateral confirma que el bus quedó atrás
            #
            # Buffer de seguridad (30 frames ≈ 1.25m a 15 km/h):
            #   Al girar de vuelta a la derecha, la cola del vehículo hace swing
            #   hacia la izquierda y puede rozar la esquina trasera del bus.
            #   Los 30 frames de marcha recta dan suficiente separación antes
            #   de iniciar el giro de recuperación.
            if _wall_engaged and dist_rr > DS_CLEAR_DIST:
                if _clear_hold == 0:
                    _clear_hold = 30   # ~1.25m de marcha recta de seguridad
                    print(f"[WALL] Sensor trasero libre ({dist_rr:.2f}m) — buffer recto")
                _clear_hold -= 1
                steering = 0.0         # recto durante el buffer
                driver.setSteeringAngle(steering)
                if _clear_hold == 0:
                    state = STATE_REORIENT
                    print(f"[WALL] Iniciando REORIENT — target={saved_heading:.4f} rad")

        elif state == STATE_REORIENT:
            # ================================================================
            # PASO 5 RÚBRICA (cont.): Recuperación de orientación con giroscopio
            # ================================================================
            # El vehículo compara el heading actual con el heading guardado
            # en el momento de iniciar la evasión y aplica un controlador P
            # para recuperar la orientación original.
            #
            # heading_error > 0: vehículo apunta más a la derecha → girar izquierda
            # heading_error < 0: vehículo apunta más a la izquierda → girar derecha
            heading_error = saved_heading - heading
            steering = float(np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE))

            driver.setCruisingSpeed(SPEED_REORIENT)
            driver.setSteeringAngle(steering)

            if int(robot.getTime() * 10) % 10 == 0:
                print(f"[REORIENT] heading={heading:.4f} target={saved_heading:.4f} "
                      f"error={heading_error:+.4f} rad")

            # Tolerancia: |error| < 0.08 rad ≈ 4.6° → heading recuperado
            if abs(heading_error) < HEADING_TOL:
                state            = STATE_RECENTER
                _recenter_frames = 0
                integral         = 0.0
                prev_error       = 0.0
                print("[REORIENT] Heading recuperado — buscando carril (RECENTER)")

        elif state == STATE_RECENTER:
            # ================================================================
            # RECUPERACIÓN DE CARRIL (extensión del PASO 5)
            # ================================================================
            # REORIENT devuelve el heading correcto pero no la posición lateral.
            # Durante la evasión el vehículo se desplazó ~1-2m a la izquierda,
            # fuera de las líneas amarillas del carril original.
            #
            # RECENTER aplica steering = +0.10 (deriva suave a la derecha) a
            # 20 km/h hasta que detecta líneas amarillas (yellow_frac > 0.015).
            # Esto cierra el loop con el sensor real: el vehículo vuelve al
            # carril cuando las líneas amarillas aparecen en cámara.
            _recenter_frames += 1

            proc_w = display.getWidth()
            proc_h = display.getHeight()
            img    = get_image(camera)
            _, yellow_mask = preprocess(img, proc_w, proc_h)
            yellow_frac = float((yellow_mask[int(proc_h * 0.6):] > 0).mean())

            steering = 0.10    # deriva constante a la derecha hacia el carril
            driver.setCruisingSpeed(SPEED_REORIENT)
            driver.setSteeringAngle(steering)

            # Log del estado RECENTER
            _wall_log_file.write(
                f"{robot.getTime():.3f},RECENTER,{lidar_dist:.2f},{lidar_right:.2f},"
                f"{dist_rf:.2f},{dist_rm:.2f},{dist_rr:.2f},{steering:.4f},0\n"
            )
            _wall_log_file.flush()

            if int(robot.getTime() * 10) % 20 == 0:
                print(f"[RECENTER] frame={_recenter_frames} yfrac={yellow_frac:.3f}")

            # Transición a LINE_FOLLOW cuando se detectan líneas amarillas
            # Umbral 0.015: normal durante seguimiento = 0.034 → margen 2×
            if yellow_frac > 0.015:
                state    = STATE_LINE_FOLLOW
                steering = 0.0
                print(f"[RECENTER] Línea encontrada (yfrac={yellow_frac:.3f}) — retomando LINE_FOLLOW")

            elif _recenter_frames > 300:
                # Timeout de seguridad: después de 300 frames (~3s) sin línea
                # se retoma LINE_FOLLOW de todas formas para no quedar atascado
                state    = STATE_LINE_FOLLOW
                steering = 0.0
                print("[RECENTER] Timeout — retomando LINE_FOLLOW")


if __name__ == "__main__":
    main()
