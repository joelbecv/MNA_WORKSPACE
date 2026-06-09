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
import os

# =============================================================================
# PARÁMETROS
# =============================================================================

MAX_ANGLE     = 0.5       # rad: ángulo máximo de volante
DEBOUNCE_TIME = 0.1       # s: antirrebote de teclado

# ── PID seguimiento de línea — parámetros iguales a H2 ───────────────────────
SPEED_FOLLOW  = 30        # km/h durante seguimiento de línea
Kp = 0.28                 # ganancia proporcional (H2)
Ki = 0.01                 # ganancia integral (H2)
Kd = 0.01                 # ganancia derivativa (H2)
DEFAULT_ANGLE = 0.0

# Canny
CANNY_LOW   = 50
CANNY_HIGH  = 150

# HoughLinesP — exactamente H2
HOUGH_RHO    = 1
HOUGH_THETA  = np.pi / 180
HOUGH_THRESH = 20
HOUGH_MIN    = 20
HOUGH_GAP    = 15

MIN_ABS_SLOPE  = 0.4      # filtro de slope H2: funciona sobre yellow-only Canny
MAX_STEER_RATE = 0.03     # rad/frame rate limiter (H2)
NO_LINE_HOLD   = 10       # frames sin línea antes de decay (H2: 10 a 50km/h)

# ── LiDAR ────────────────────────────────────────────────────────────────────
BUS_LIDAR_THRESH = 14.5   # m: iniciar evasión con suficiente margen para rodear el bus
LIDAR_FOV_DEG    = 20     # grados a cada lado del centro para medir frente

# ── Wall-following (evasión) ─────────────────────────────────────────────────
SPEED_EVADE   = 15        # km/h durante la evasión (mitad de SPEED_FOLLOW=30)
WALL_TARGET   = 2.9       # m: distancia objetivo al costado del autobús
KP_WALL       = 0.10      # ganancia P del controlador de pared derecha
DS_CLEAR_DIST = 4.8       # m: sensor trasero sin obstáculo → bus superado
DS_ENGAGE_DIST = 4.5      # m: sensor frontal debe leer < esto para considerar bus detectado lateralmente

# ── Filtros anti-falso-positivo en recognition ────────────────────────────────
BUS_MIN_PX_AREA = 400     # px²: detectar bus desde lejos (era 2000, bloqueaba hasta <6m)
BUS_CONFIRM_FRAMES = 1    # 1 frame basta — color exacto ya garantiza no hay falsos positivos

# ── Reorientación con giroscopio ─────────────────────────────────────────────
SPEED_REORIENT = 20       # km/h mientras se recupera el heading
KP_HEADING     = 1.0      # ganancia P del corrector de heading
HEADING_TOL    = 0.08     # rad: tolerancia para declarar heading recuperado

# ── Bus colors (para identificación por nodo de reconocimiento) ──────────────
# Colores definidos en el mundo para cada autobús
BUS_COLOR_MAP = {
    "vehicle(1)": (0.0313726, 0.121569, 0.419608),  # azul marino
    "vehicle(2)": (1.000000,  0.000000, 0.000000),  # rojo
    "vehicle(3)": (0.862745,  0.541176, 0.866667),  # lavanda
    "vehicle(4)": (0.180392,  0.760784, 0.494118),  # verde
}

# ── Estados ───────────────────────────────────────────────────────────────────
STATE_LINE_FOLLOW = 0
STATE_WALL_FOLLOW = 1
STATE_REORIENT    = 2
STATE_RECENTER    = 3
STATE_NAMES = {0: "LINE_FOLLOW", 1: "WALL_FOLLOW", 2: "REORIENT", 3: "RECENTER"}

# =============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMAGEN (Canny + Hough)
# =============================================================================

def get_image(camera):
    """Lee imagen raw de la cámara y la convierte a array numpy BGRA."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def preprocess(image, proc_w, proc_h):
    """
    Pipeline H2: yellow-only Canny sobre imagen resizeada a dimensiones del display.
    Yellow HSV [15,80,80]-[35,255,255] — rango H2 original.
    Solo sobreviven píxeles amarillos: cebra blanca y ruido gris desaparecen.
    Retorna (edges, yellow_mask).
    """
    bgr         = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    resized     = cv2.resize(bgr, (proc_w, proc_h))
    hsv         = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
    edges       = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)
    return edges, yellow_mask


def apply_roi(edges, height, width):
    """
    ROI trapezoidal H2: base 10-90% ancho, techo 35-65% a altura 60%.
    Excluye cielo y bordes de edificios preservando el centro de carretera.
    """
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.35), int(height * 0.6)),
        (int(width * 0.65), int(height * 0.6)),
        (int(width * 0.90), height),
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


def filter_lines_by_slope(lines):
    """Filtra líneas casi-horizontales (slope < MIN_ABS_SLOPE). H2 usa 0.4."""
    filtered = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            continue
        filtered.append([x1, y1, x2, y2])
    return filtered


def compute_lane_center(lines):
    """
    Calcula el centro del carril promediando puntos izquierdos y derechos (H2).
    Clasifica por slope: negativo=izquierda, positivo=derecha.
    Retorna x-coordenada del centro, o None si no hay líneas.
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
            left_pts.extend([x1, x2])
        else:
            right_pts.extend([x1, x2])
    if left_pts and right_pts:
        return (np.mean(left_pts) + np.mean(right_pts)) / 2.0
    if all_pts:
        return np.mean(all_pts)
    return None


def display_with_lines(display_dev, edges, lines, width, height):
    """
    Visualización H2: ROI edges en gris + líneas Hough detectadas en blanco.
    Permite ver exactamente qué está detectando el PID en tiempo real.
    """
    line_layer = np.zeros((height, width), dtype=np.uint8)
    if lines is not None and len(lines) > 0:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_layer, (x1, y1), (x2, y2), 255, 2)
    debug = cv2.addWeighted(edges, 0.6, line_layer, 1.0, 0)
    rgb = np.dstack((debug, debug, debug))
    ref = display_dev.imageNew(
        rgb.tobytes(), Display.RGB, width=width, height=height)
    display_dev.imagePaste(ref, 0, 0, False)
    display_dev.imageDelete(ref)


def display_image(display_dev, image):
    """Muestra imagen en escala de grises en el display del vehículo."""
    rgb = np.dstack((image, image, image))
    ref = display_dev.imageNew(
        rgb.tobytes(), Display.RGB,
        width=rgb.shape[1], height=rgb.shape[0],
    )
    display_dev.imagePaste(ref, 0, 0, False)


def identify_bus_color(obj, debug=False):
    """
    Extrae el color de un RecognitionObject y lo compara contra los autobuses conocidos.
    En Webots R2023b getColors() devuelve LP_c_double (ctypes); usar getNumberOfColors().
    Retorna (nombre, color_str) o ("desconocido", color_str).
    """
    n = obj.getNumberOfColors()
    if n < 1:
        return "desconocido", "N/A"
    colors = obj.getColors()           # LP_c_double — indexable pero sin len()
    r, g, b = colors[0], colors[1], colors[2]
    color_str = f"RGB({r:.3f},{g:.3f},{b:.3f})"
    if debug:
        model = obj.getModel()
        sz    = obj.getSizeOnImage()
        pos   = obj.getPositionOnImage()
        print(f"[DBG] modelo='{model}' color={color_str} size={sz[0]}x{sz[1]} pos=({pos[0]},{pos[1]})")
    for name, (cr, cg, cb) in BUS_COLOR_MAP.items():
        if abs(r - cr) < 0.05 and abs(g - cg) < 0.05 and abs(b - cb) < 0.05:
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
    camera.recognitionEnable(timestep)    # habilitar nodo de reconocimiento (API R2023b)
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
    # Eje Z = yaw en el frame del vehículo (instrucción: "ángulo en el eje z").
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    heading       = 0.0    # heading acumulado (rad) desde inicio de simulación
    saved_heading = 0.0    # heading al momento de iniciar evasión

    # ── Sensores de distancia — costado derecho ───────────────────────────────
    # Montados en sensorsSlotCenter con translation -1.0 en Y (derecha del carro).
    # R2023b BmwX5 no tiene sensorsSlotRight; se usan translations explícitas.
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
    steering   = 0.0        # ángulo actual (necesario para rate limiter y hold/decay)
    no_line_frames = 0      # contador de frames sin línea detectada (para hold/decay H2)
    prev_time  = time.time()
    last_press = {}

    # Banderas de impresión y control anti-falso-positivo
    _prev_state      = -1
    _bus_print_cd    = 0   # cooldown entre prints de recognition
    _bus_streak      = 0   # frames consecutivos con bus válido detectado
    _wall_engaged    = False  # True cuando ds_right_front detectó el bus lateralmente
    _clear_hold      = 0     # frames de buffer recto tras liberar sensor trasero
    _recenter_frames = 0   # frames en RECENTER (timeout de seguridad)

    # ── Log de diagnóstico ────────────────────────────────────────────────────
    # n_raw   = líneas Hough antes del filtro de slope (si 0: ROI/Canny falla)
    # n_filt    = líneas tras slope filter
    # cx        = centro de carril detectado en px
    # err       = error PID normalizado [-1,1]
    # nolf      = frames sin línea (hold/decay counter)
    # lidar     = distancia frontal LiDAR (m)
    # n_obj     = objetos totales en recognition
    # top_model = modelo del objeto más grande detectado
    # top_color = color RGB del objeto más grande
    # top_area  = área en px² del objeto más grande
    # top_px    = posición x del objeto más grande
    # bus_str   = streak de confirmación de bus
    _log_path = os.path.join(os.path.dirname(__file__), "act42_diag.log")
    _log_file = open(_log_path, "w")
    _log_file.write("frame,sim_t,state,yellow_frac,n_raw,n_filt,cx,err,steer,nolf,lidar,"
                    "n_obj,top_model,top_color,top_area,top_px,bus_str\n")
    _wall_log_path = os.path.join(os.path.dirname(__file__), "act42_wall.log")
    _wall_log_file = open(_wall_log_path, "w")
    _wall_log_file.write("sim_t,state,lidar_front,lidar_right,dist_rf,dist_rm,dist_rr,steering,engaged\n")
    _frame = 0
    print(f"[INIT] Log de diagnóstico → {_log_path}")
    print(f"[INIT] Log wall-follow     → {_wall_log_path}")

    print("[INIT] Actividad 4.2 — Controlador iniciado")
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
        # getValues()[1] = yaw rate en BmwX5 Webots R2023b (validado en Semana_7)
        heading += gyro.getValues()[1] * dt

        # ── [3] Lectura LiDAR — sectores frontal y lateral derecho ───────────────
        # índice 0=izquierda, 90=frente, 180=derecha (igual que Semana 7)
        ranges = list(sick.getRangeImage())
        span   = max(1, int(n_rays * LIDAR_FOV_DEG / 180))
        center = n_rays // 2
        front_vals  = [r for r in ranges[center - span : center + span] if r < 99.0]
        lidar_dist  = float(min(front_vals)) if front_vals else 99.0

        # Sector lateral derecho puro (índices 150-180 = 65°-90° del frente)
        # El bus al costado aparece a ~90° del frente (índice 180); antes 135-165 lo perdía
        r_start = n_rays * 150 // 180
        r_end   = n_rays
        right_vals  = [r for r in ranges[r_start:r_end] if r < 99.0]
        lidar_right = float(min(right_vals)) if right_vals else 99.0

        # ── [4] Lectura sensores laterales derecha ────────────────────────────
        dist_rf = float(ds_front.getValue())   # distancia sensor frontal derecho
        dist_rm = float(ds_mid.getValue())     # distancia sensor medio derecho
        dist_rr = float(ds_rear.getValue())    # distancia sensor trasero derecho

        # ── [5] Recognition — detección de autobús ───────────────────────────
        bus_in_front = False
        bus_name     = "desconocido"
        color_str    = "N/A"
        objects      = camera.getRecognitionObjects()

        # Capturar el objeto más grande para el log (sin filtros)
        top_model = "none"
        top_color = "none"
        top_area  = 0
        top_px    = -1
        for _o in objects:
            _si  = _o.getSizeOnImage()
            _a   = _si[0] * _si[1]
            if _a > top_area:
                top_area  = _a
                top_px    = int(_o.getPositionOnImage()[0])
                top_model = _o.getModel()
                _, top_color = identify_bus_color(_o)

        for obj in objects:
            pos_img  = obj.getPositionOnImage()
            img_size = obj.getSizeOnImage()
            px_area  = img_size[0] * img_size[1]

            # Filtro 1: bus en el 50% central de la imagen
            if abs(pos_img[0] - setpoint) > width * 0.50:
                continue
            # Filtro 2: tamaño mínimo
            if px_area < BUS_MIN_PX_AREA:
                continue
            # Filtro 3: color coincide con un bus conocido
            bus_name, color_str = identify_bus_color(obj)
            if bus_name == "desconocido":
                continue

            bus_in_front = True
            break

        # Streak: acumular frames consecutivos con bus válido
        if bus_in_front:
            _bus_streak += 1
        else:
            _bus_streak = 0

        bus_confirmed = _bus_streak >= BUS_CONFIRM_FRAMES

        if bus_in_front and _bus_print_cd <= 0:
            print(f"[BUS] {bus_name} | color={color_str} | "
                  f"streak={_bus_streak}/{BUS_CONFIRM_FRAMES} | LiDAR={lidar_dist:.1f}m")
            _bus_print_cd = 60
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
            # ── Transición: bus confirmado (streak) Y LiDAR dentro del umbral ──
            # Paso 3: guardar heading (eje Z) y activar evasión
            if bus_confirmed and lidar_dist < BUS_LIDAR_THRESH:
                saved_heading = heading
                state         = STATE_WALL_FOLLOW
                integral      = 0.0
                _wall_engaged = False   # esperar a que sensor frontal detecte el bus
                _bus_streak   = 0
                print(f"[EVASION] Bus confirmado a {lidar_dist:.1f} m — activando wall-follow")
                print(f"[EVASION] heading guardado = {saved_heading:.4f} rad")
                continue

            # ── PID seguimiento de línea — pipeline H2 ────────────────────────
            proc_w = display.getWidth()
            proc_h = display.getHeight()

            img                = get_image(camera)
            edges, yellow_mask = preprocess(img, proc_w, proc_h)
            roi                = apply_roi(edges, proc_h, proc_w)
            raw_lines          = detect_lines(roi)           # lista antes del filtro
            filt_lines         = filter_lines_by_slope(raw_lines)
            lane_center_x      = compute_lane_center(filt_lines)
            yellow_frac        = float((yellow_mask[int(proc_h * 0.6):] > 0).mean())

            # Guardar frames de diagnóstico (frame 5 y 50) — tamaño display
            if _frame in (5, 50):
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_edges_{_frame}.png"), edges)
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_roi_{_frame}.png"), roi)
                cv2.imwrite(os.path.join(os.path.dirname(__file__), f"dbg_yellow_{_frame}.png"), yellow_mask)

            display_with_lines(display, roi, filt_lines, proc_w, proc_h)

            # HUD: texto sobre el display con info clave
            bus_label = bus_name if bus_in_front else "---"
            hud_color = 0xFF0000 if bus_in_front else 0x00FF00   # rojo si bus, verde si no
            display.setColor(0x000000)
            display.fillRectangle(0, 0, proc_w, 22)              # fondo negro para legibilidad
            display.setColor(hud_color)
            display.drawText(f"LIDAR:{lidar_dist:.1f}m  BUS:{bus_label}", 2, 2)
            display.setColor(0xFFFFFF)
            display.drawText(f"ST:{STATE_NAMES[state]}  E:{prev_error:.2f}", 2, 12)

            if lane_center_x is not None:
                no_line_frames = 0
                image_center   = proc_w / 2.0
                error_norm     = (lane_center_x - image_center) / image_center  # [-1, 1]
                integral      += error_norm * real_dt
                integral       = max(-0.5, min(0.5, integral))   # clamp windup (H2)
                derivative     = (error_norm - prev_error) / real_dt
                raw_steering   = Kp * error_norm + Ki * integral + Kd * derivative
                raw_steering   = float(np.clip(raw_steering, -MAX_ANGLE, MAX_ANGLE))
                steering       = max(steering - MAX_STEER_RATE,
                                     min(steering + MAX_STEER_RATE, raw_steering))
                prev_error = error_norm
            else:
                no_line_frames += 1
                integral       *= 0.6
                prev_error      = 0.0
                if no_line_frames > NO_LINE_HOLD:
                    steering *= 0.95

            steering = float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))
            driver.setCruisingSpeed(SPEED_FOLLOW)
            driver.setSteeringAngle(steering)

            lane_cx_log = f"{lane_center_x:.1f}" if lane_center_x is not None else "None"
            err_log     = f"{prev_error:.3f}" if lane_center_x is not None else "None"
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
            # ── Paso 4: seguimiento de pared derecha con LiDAR lateral ───────────
            # Usa el sector derecho del LiDAR (135-165°) — mayor alcance que los
            # sensores de 5m, mismo enfoque que Semana 7.
            # Fase A: bus aún enfrente → girar izquierda suavemente para rodearlo
            # Fase B: bus al costado derecho → P controller a WALL_TARGET (2.5m)
            right_dist = min(lidar_right, dist_rf)   # el más cercano de ambos sensores

            if right_dist > DS_ENGAGE_DIST:
                steering = -0.08   # giro izquierda suave para rodear el bus
            else:
                error_wall = right_dist - WALL_TARGET
                steering   = float(np.clip(KP_WALL * error_wall, -MAX_ANGLE, MAX_ANGLE))

            driver.setCruisingSpeed(SPEED_EVADE)
            driver.setSteeringAngle(steering)

            # Confirmar que el bus está al costado
            if not _wall_engaged and right_dist < DS_ENGAGE_DIST:
                _wall_engaged = True
                print(f"[WALL] Bus al costado ({right_dist:.2f}m) — seguimiento activo")

            _wall_log_file.write(
                f"{robot.getTime():.3f},WALL_FOLLOW,{lidar_dist:.2f},{lidar_right:.2f},"
                f"{dist_rf:.2f},{dist_rm:.2f},{dist_rr:.2f},{steering:.4f},{int(_wall_engaged)}\n"
            )
            _wall_log_file.flush()

            if int(robot.getTime() * 10) % 10 == 0:
                print(f"[WALL] lidar_r={lidar_right:.2f}m ds_rf={dist_rf:.2f}m "
                      f"rear={dist_rr:.2f}m | steer={steering:+.3f}")

            # ── Paso 5: sensor trasero libre → buffer recto → REORIENT ──────────
            # Cuando dist_rr se libera, el bus acaba de pasar la cola del coche.
            # Girar de inmediato hace que la cola barra hacia el bus → choque.
            # Buffer de 80 frames (~0.8s = ~3.3m a 15 km/h) da margen suficiente.
            if _wall_engaged and dist_rr > DS_CLEAR_DIST:
                if _clear_hold == 0:
                    _clear_hold = 27 #ajuste para nochocar con autobus al girar
                    print(f"[WALL] Sensor trasero libre ({dist_rr:.2f}m) — iniciando buffer recto")
                _clear_hold -= 1
                steering = 0.0   # recto durante el buffer
                driver.setSteeringAngle(steering)
                if _clear_hold == 0:
                    state = STATE_REORIENT
                    print(f"[WALL] Buffer completo — iniciando REORIENT (target={saved_heading:.4f} rad)")

        elif state == STATE_REORIENT:
            # ── Paso 5 (cont.): recuperar heading original con giroscopio ─────
            # La diferencia entre el heading guardado y el actual es el error.
            # KP_HEADING determina qué tan agresivamente gira para alinearse.
            heading_error = saved_heading - heading
            steering = float(np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE))

            driver.setCruisingSpeed(SPEED_REORIENT)
            driver.setSteeringAngle(steering)

            if int(robot.getTime() * 10) % 10 == 0:   # print cada ~1 segundo
                print(f"[REORIENT] heading={heading:.4f} target={saved_heading:.4f} "
                      f"error={heading_error:+.4f} rad")

            # ── Transición: heading recuperado → recentrar en carril ──────────
            if abs(heading_error) < HEADING_TOL:
                state            = STATE_RECENTER
                _recenter_frames = 0
                integral         = 0.0
                prev_error       = 0.0
                print("[REORIENT] Heading recuperado — buscando carril (RECENTER)")

        elif state == STATE_RECENTER:
            # ── Deriva derecha a baja velocidad hasta encontrar línea amarilla ──
            # El carro salió ~1-2m a la izquierda durante la evasión.
            # Steering fijo +0.10 (derecha) hasta que yellow_frac > 0.015.
            _recenter_frames += 1

            proc_w = display.getWidth()
            proc_h = display.getHeight()
            img = get_image(camera)
            _, yellow_mask = preprocess(img, proc_w, proc_h)
            yellow_frac = float((yellow_mask[int(proc_h * 0.6):] > 0).mean())

            steering = 0.10   # deriva suave hacia carril derecho
            driver.setCruisingSpeed(SPEED_REORIENT)
            driver.setSteeringAngle(steering)

            _wall_log_file.write(
                f"{robot.getTime():.3f},RECENTER,{lidar_dist:.2f},{lidar_right:.2f},"
                f"{dist_rf:.2f},{dist_rm:.2f},{dist_rr:.2f},{steering:.4f},0\n"
            )
            _wall_log_file.flush()

            if int(robot.getTime() * 10) % 20 == 0:
                print(f"[RECENTER] frame={_recenter_frames} yfrac={yellow_frac:.3f}")

            # Línea encontrada → retomar PID
            if yellow_frac > 0.015:
                state   = STATE_LINE_FOLLOW
                steering = 0.0
                print(f"[RECENTER] Línea encontrada (yfrac={yellow_frac:.3f}) — retomando LINE_FOLLOW")

            # Timeout de seguridad: 300 frames (~3s) sin línea
            elif _recenter_frames > 300:
                state   = STATE_LINE_FOLLOW
                steering = 0.0
                print("[RECENTER] Timeout — retomando LINE_FOLLOW sin línea")


if __name__ == "__main__":
    main()
