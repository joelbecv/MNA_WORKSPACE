# Controlador Webots — Actividad 4.2
# Seguidor de línea (PID) + Detección de autobús (Recognition + LiDAR) + Evasión (Wall-Following derecha)
#
# Máquina de estados:
#   LINE_FOLLOW  — PID sobre líneas de carril (Canny + HoughLinesP)
#   WALL_FOLLOW  — Evasión lateral derecha usando sensores de distancia
#   REORIENT     — Recuperación del heading original con giroscopio
#
# Sensores habilitados:
#   - Cámara 128×128 + nodo Recognition → detección de autobuses
#   - SickLMS291 LiDAR frontal         → distancia al autobús
#   - Giroscopio (eje Y = yaw)          → integración de heading
#   - ds_right_front / ds_right_mid / ds_right_rear → seguimiento de pared
#
# Teclas: M = forzar LINE_FOLLOW (debug)

from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
import time

# =============================================================================
# PARÁMETROS
# =============================================================================

MAX_ANGLE     = 0.5       # rad: ángulo máximo de volante
DEBOUNCE_TIME = 0.1       # s: antirrebote de teclado

# ── PID seguimiento de línea ─────────────────────────────────────────────────
SPEED_FOLLOW  = 30        # km/h durante seguimiento de línea
Kp = 0.003                # ganancia proporcional: reacción al error actual
Ki = 0.0001               # ganancia integral: corrección de deriva acumulada
Kd = 0.001                # ganancia derivativa: amortiguación de oscilaciones
DEFAULT_ANGLE = 0.0       # volante recto cuando no hay línea detectada

# Canny
CANNY_LOW   = 50
CANNY_HIGH  = 150

# HoughLinesP
HOUGH_RHO    = 1
HOUGH_THETA  = np.pi / 180
HOUGH_THRESH = 30
HOUGH_MIN    = 30          # longitud mínima de segmento en px
HOUGH_GAP    = 100         # hueco máximo entre segmentos para unirlos

MIN_VERT_DIFF = 10         # filtro de líneas horizontales (cruces)

# ── LiDAR ────────────────────────────────────────────────────────────────────
BUS_LIDAR_THRESH = 8.0    # m: distancia a la que se activa la evasión
LIDAR_FOV_DEG    = 20     # grados a cada lado del centro para medir frente

# ── Wall-following (evasión) ─────────────────────────────────────────────────
SPEED_EVADE   = 20        # km/h durante la evasión
WALL_TARGET   = 2.5       # m: distancia objetivo al costado del autobús
KP_WALL       = 0.10      # ganancia P del controlador de pared derecha
DS_CLEAR_DIST = 4.8       # m: sensor trasero sin obstáculo → bus superado

# ── Reorientación con giroscopio ─────────────────────────────────────────────
SPEED_REORIENT = 20       # km/h mientras se recupera el heading
KP_HEADING     = 1.0      # ganancia P del corrector de heading
HEADING_TOL    = 0.08     # rad: tolerancia para declarar heading recuperado

# ── Bus colors (para identificación por nodo de reconocimiento) ──────────────
# Colores definidos en el mundo para cada autobús
BUS_COLOR_MAP = {
    "vehicle(1)": (0.031, 0.122, 0.420),   # azul marino
    "vehicle(2)": (1.000, 0.000, 0.000),   # rojo
    "vehicle(3)": (0.863, 0.541, 0.867),   # lavanda
    "vehicle(4)": (0.180, 0.761, 0.494),   # verde
}

# ── Estados ───────────────────────────────────────────────────────────────────
STATE_LINE_FOLLOW = 0
STATE_WALL_FOLLOW = 1
STATE_REORIENT    = 2
STATE_NAMES = {0: "LINE_FOLLOW", 1: "WALL_FOLLOW", 2: "REORIENT"}

# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN (Canny + Hough)
# =============================================================================

def get_image(camera):
    """Lee imagen raw de la cámara y la convierte a array numpy BGRA."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def preprocess(image):
    """Convierte BGRA a escala de grises y aplica Canny para detectar bordes."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return edges


def apply_roi(edges, height, width):
    """
    Aplica Region of Interest trapézoidal al 40% inferior de la imagen.
    Filtra edificios y cielo; solo deja la carretera visible al frente.
    """
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0,     height),
        (0,     int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(roi_edges):
    """
    HoughLinesP detecta segmentos de línea en la imagen de bordes+ROI.
    Devuelve lista de [x1, y1, x2, y2]; lista vacía si no hay líneas.
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


def compute_error(lines, setpoint):
    """
    Calcula el error PID como la distancia horizontal entre el punto medio
    de la línea más cercana al centro y el setpoint (centro de imagen).
    Ignora líneas casi horizontales (marcas de intersección).
    """
    best_error = None
    for x1, y1, x2, y2 in lines:
        if abs(y2 - y1) < MIN_VERT_DIFF:
            continue
        mid_x = (x1 + x2) / 2
        error = mid_x - setpoint
        if best_error is None or abs(error) < abs(best_error):
            best_error = error
    return best_error


def display_image(display_dev, image):
    """Muestra imagen en escala de grises en el display del vehículo."""
    rgb = np.dstack((image, image, image))
    ref = display_dev.imageNew(
        rgb.tobytes(), Display.RGB,
        width=rgb.shape[1], height=rgb.shape[0],
    )
    display_dev.imagePaste(ref, 0, 0, False)


def identify_bus_color(colors_flat):
    """
    Compara los primeros 3 valores RGB del objeto reconocido contra los colores
    conocidos de los autobuses para identificar cuál es.
    Retorna (nombre, color_str) o ("desconocido", color_str).
    """
    if len(colors_flat) < 3:
        return "desconocido", "N/A"
    r, g, b = colors_flat[0], colors_flat[1], colors_flat[2]
    color_str = f"RGB({r:.2f},{g:.2f},{b:.2f})"
    for name, (cr, cg, cb) in BUS_COLOR_MAP.items():
        if abs(r - cr) < 0.15 and abs(g - cg) < 0.15 and abs(b - cb) < 0.15:
            return name, color_str
    return "desconocido", color_str

# =============================================================================
# MAIN
# =============================================================================

def main():
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())
    dt       = timestep / 1000.0          # paso de simulación en segundos

    # ── Cámara + Recognition ─────────────────────────────────────────────────
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    camera.enableRecognition(timestep)    # habilitar nodo de reconocimiento
    width    = camera.getWidth()
    height   = camera.getHeight()
    setpoint = width / 2.0               # objetivo PID: línea al centro horizontal

    # ── Display ──────────────────────────────────────────────────────────────
    display = robot.getDevice("display_image")

    # ── Teclado ──────────────────────────────────────────────────────────────
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── LiDAR (Sick LMS 291) ─────────────────────────────────────────────────
    sick   = robot.getDevice("Sick LMS 291")
    sick.enable(timestep)
    n_rays = sick.getHorizontalResolution()   # número total de rayos (típico: 181)

    # ── Giroscopio ───────────────────────────────────────────────────────────
    # Devuelve velocidad angular [rad/s] en ejes [X, Y, Z].
    # Eje Y = yaw (rotación vertical); integramos para obtener heading absoluto.
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    heading       = 0.0    # heading acumulado (rad) desde inicio de simulación
    saved_heading = 0.0    # heading al momento de iniciar evasión

    # ── Sensores de distancia — costado derecho ───────────────────────────────
    # Montados en sensorsSlotRight del BmwX5 (modificación en .wbt)
    # Miden distancia al obstáculo derecho (autobús); rango 0–5 m, retorno lineal
    ds_front = robot.getDevice("ds_right_front")   # posición frontal
    ds_mid   = robot.getDevice("ds_right_mid")     # posición media
    ds_rear  = robot.getDevice("ds_right_rear")    # posición trasera
    ds_front.enable(timestep)
    ds_mid.enable(timestep)
    ds_rear.enable(timestep)

    # ── Estado inicial ────────────────────────────────────────────────────────
    state      = STATE_LINE_FOLLOW
    integral   = 0.0
    prev_error = 0.0
    prev_time  = time.time()
    last_press = {}

    # Banderas de impresión para no saturar consola
    _prev_state   = -1
    _bus_print_cd = 0     # cooldown en frames entre prints de recognition

    print("[INIT] Actividad 4.1 — Controlador iniciado")
    print("[INIT] Estado: LINE_FOLLOW | velocidad PID: 30 km/h")
    print(f"[INIT] Umbral LiDAR para evasión: {BUS_LIDAR_THRESH} m")

    while robot.step() != -1:
        current_time = time.time()
        real_dt = current_time - prev_time
        if real_dt <= 0:
            real_dt = 1e-6
        prev_time = current_time

        # ── [1] Impresión de cambio de estado ─────────────────────────────────
        if state != _prev_state:
            print(f"\n[STATE] → {STATE_NAMES[state]}")
            _prev_state = state

        # ── [2] Lectura giroscopio — integración de yaw ───────────────────────
        # getValues()[1] = velocidad angular alrededor del eje Y (vertical = yaw)
        heading += gyro.getValues()[1] * dt

        # ── [3] Lectura LiDAR — distancia frontal ─────────────────────────────
        # El LiDAR tiene 181 rayos en 180°; tomamos el sector frontal ±LIDAR_FOV_DEG
        ranges = list(sick.getRangeImage())
        span   = max(1, int(n_rays * LIDAR_FOV_DEG / 180))
        center = n_rays // 2
        front_vals = [r for r in ranges[center - span : center + span] if r < 99.0]
        lidar_dist = float(min(front_vals)) if front_vals else 99.0

        # ── [4] Lectura sensores laterales derecha ────────────────────────────
        dist_rf = float(ds_front.getValue())   # distancia sensor frontal derecho
        dist_rm = float(ds_mid.getValue())     # distancia sensor medio derecho
        dist_rr = float(ds_rear.getValue())    # distancia sensor trasero derecho

        # ── [5] Recognition — detección de autobús ───────────────────────────
        bus_in_front = False
        bus_name     = "desconocido"
        objects = camera.getRecognitionObjects()
        for obj in objects:
            pos_img = obj.getPositionOnImage()    # [x_pixel, y_pixel] del objeto
            # Filtrar objetos que estén cerca del centro horizontal de la imagen
            if abs(pos_img[0] - setpoint) < width * 0.45:
                colors = obj.getColors()          # lista plana [r,g,b,...]
                bus_name, color_str = identify_bus_color(colors)
                bus_in_front = True
                break

        # Imprimir detección de autobús cada 30 frames para no saturar consola
        if bus_in_front and state == STATE_LINE_FOLLOW:
            if _bus_print_cd <= 0:
                print(f"[RECOGNITION] Autobús en frente: {bus_name} | "
                      f"Color: {color_str} | LiDAR: {lidar_dist:.1f} m")
                _bus_print_cd = 30
        _bus_print_cd -= 1

        # ── [6] Teclado ───────────────────────────────────────────────────────
        key = keyboard.getKey()
        if key != -1:
            now = time.time()
            if key not in last_press or (now - last_press[key] >= DEBOUNCE_TIME):
                last_press[key] = now
                if key == ord('M'):
                    state = STATE_LINE_FOLLOW
                    integral = 0.0
                    print("[KEY-M] Forzado a LINE_FOLLOW")

        # ======================================================================
        # MÁQUINA DE ESTADOS
        # ======================================================================

        if state == STATE_LINE_FOLLOW:
            # ── Transición: autobús detectado Y LiDAR dentro del umbral ───────
            # Paso 3 de la actividad: guardar heading y activar evasión
            if bus_in_front and lidar_dist < BUS_LIDAR_THRESH:
                saved_heading = heading
                state         = STATE_WALL_FOLLOW
                integral      = 0.0
                print(f"[EVASION] Bus a {lidar_dist:.1f} m — activando wall-follow")
                print(f"[EVASION] heading guardado = {saved_heading:.4f} rad")
                continue

            # ── PID seguimiento de línea (pasos 1–2 activos) ──────────────────
            img    = get_image(camera)
            edges  = preprocess(img)
            roi    = apply_roi(edges, height, width)
            lines  = detect_lines(roi)
            error  = compute_error(lines, setpoint)
            display_image(display, edges)

            if error is None:
                steering   = DEFAULT_ANGLE
                integral   = 0.0
                prev_error = 0.0
            else:
                integral   += error * real_dt
                derivative  = (error - prev_error) / real_dt
                steering    = Kp * error + Ki * integral + Kd * derivative
                prev_error  = error

            steering = float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))
            driver.setCruisingSpeed(SPEED_FOLLOW)
            driver.setSteeringAngle(steering)

        elif state == STATE_WALL_FOLLOW:
            # ── Paso 4: seguimiento de pared derecha ───────────────────────────
            # El sensor frontal mantiene la distancia objetivo al autobús.
            # error_wall > 0 → carro demasiado lejos → girar derecha (+ steering)
            # error_wall < 0 → carro demasiado cerca → girar izquierda (- steering)
            error_wall = dist_rf - WALL_TARGET
            steering   = float(np.clip(KP_WALL * error_wall, -MAX_ANGLE, MAX_ANGLE))

            driver.setCruisingSpeed(SPEED_EVADE)
            driver.setSteeringAngle(steering)

            print(f"[WALL] front={dist_rf:.2f}m mid={dist_rm:.2f}m rear={dist_rr:.2f}m "
                  f"| steer={steering:+.3f} | LiDAR={lidar_dist:.1f}m")

            # ── Paso 5: sensor trasero libre → autobús superado ───────────────
            # DS_CLEAR_DIST se alcanza cuando no hay obstáculo dentro del rango
            # del sensor trasero, lo que indica que el autobús quedó atrás.
            if dist_rr > DS_CLEAR_DIST:
                state = STATE_REORIENT
                print(f"[WALL] Sensor trasero libre ({dist_rr:.2f}m) — bus superado")
                print(f"[WALL] Iniciando recuperación de heading: target={saved_heading:.4f} rad")

        elif state == STATE_REORIENT:
            # ── Paso 5 (cont.): recuperar heading original con giroscopio ─────
            # La diferencia entre el heading guardado y el actual es el error.
            # KP_HEADING determina qué tan agresivamente gira para alinearse.
            heading_error = saved_heading - heading
            steering = float(np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE))

            driver.setCruisingSpeed(SPEED_REORIENT)
            driver.setSteeringAngle(steering)

            print(f"[REORIENT] heading={heading:.4f} target={saved_heading:.4f} "
                  f"error={heading_error:+.4f} rad")

            # ── Transición: heading recuperado → retomar seguimiento de línea ──
            if abs(heading_error) < HEADING_TOL:
                state      = STATE_LINE_FOLLOW
                integral   = 0.0
                prev_error = 0.0
                print("[REORIENT] Heading recuperado — retomando LINE_FOLLOW")


if __name__ == "__main__":
    main()
