# =============================================================================
# collect_cil_data.py — Recolección autónoma de datos CIL (Codevilla 2017)
# Proyecto Final MR4010 — Equipo 25
# =============================================================================
#
# Modo AUTO (default): el coche sigue el carril solo y negocia las intersecciones
# de forma autónoma rotando entre RECTO / IZQ / DER para balancear el dataset.
# El etiquetado de nav_command se aplica automáticamente según la maniobra.
#
# CONTROLES (ventana 3D de Webots):
#   F       : alternar AUTO / MANUAL
#   ← / →   : volante (solo en MANUAL, AUTO lo ignora)
#   i / k   : velocidad +5 / -5 km/h
#   s       : CMD_CONTINUE  — carretera normal   (MANUAL: etiqueta manual)
#   w       : CMD_STRAIGHT  — recto en cruce     (MANUAL: etiqueta manual)
#   a       : CMD_LEFT      — giro izquierda     (MANUAL: etiqueta manual)
#   d       : CMD_RIGHT     — giro derecha       (MANUAL: etiqueta manual)
#   q       : guardar y salir
# =============================================================================

from controller import Keyboard
from vehicle import Car, Driver
import numpy as np, cv2, os, csv, math, time

# =============================================================================
# PARÁMETROS — seguidor de carril
# =============================================================================

CRUISE_SPEED  = 35          # km/h en recta
SPEED_MIN     = 10
SPEED_MAX     = 80
SPEED_STEP    = 5
MAX_ANGLE     = 0.50        # rad máximo volante
STEER_STEP    = 0.015       # paso manual
CENTER_DECAY  = 0.85        # decay en manual sin tecla

ROAD_THRESH   = 58          # umbral gris → asfalto (< thresh = oscuro = asfalto)
MIN_ROAD_PX   = 30
ROI_TOP       = 0.55
ROI_BOT       = 0.92
LANE_OFFSET   = 0.12        # sesgo al carril derecho
SMOOTH_ALPHA  = 0.25        # EMA — 0.25 = respuesta más rápida a curvas (era 0.15)
MAX_JUMP      = 0.15        # rechazo de outliers por frame
KP_AUTO       = 0.35        # ganancia proporcional
RATE_LIMIT    = 0.05        # rad max cambio por ciclo

# DistanceSensors — corrección lateral derecha
DS_TARGET     = 2.0         # m — distancia objetivo al obstáculo/borde derecho
KP_DS         = 0.08        # ganancia proporcional DS lateral
DS_BLEND      = 0.45        # fracción DS en la mezcla (1-DS_BLEND = flood fill)

# LiDAR — evasión de obstáculos frontales
LIDAR_OBS_DIST = 8.0        # m — activar sólo ante obstáculos realmente en la calzada (18m detectaba edificios laterales)
LIDAR_STEER    = 0.22       # rad — steer máximo de evasión (era 0.35, reducido para no sobrecorregir)

smooth_center = 0.0

# =============================================================================
# PARÁMETROS — navegación GPS pura (waypoints)
# =============================================================================
# El flood fill ya NO controla el steering — solo se llama para log/debug.
# El coche sigue esta lista de GPS waypoints en bucle; la FSM sólo gestiona
# los giros en SE y NW. NE se navega solo con waypoints (sin FSM).

KP_GPS         = 1.0    # ganancia proporcional para error de heading GPS
GPS_WP_RADIUS  = 18.0   # m — radio original que funcionó en el run f=3240

# Con radius=18m, WP en (105,111) hace que el coche avance exactamente en y=93:
#   dist((104,93) → (105,111)) = sqrt(1+324) = 18.03m → avanza en la intersección NE
# WP13=(105,-43) se salta automáticamente junto con WP12 (ambos <18m al mismo tiempo)
ROUTE_GPS_WAYPOINTS = [
    # ── Tramo A: Start → SW corner → camino sur  [IGUAL al run que llegó a f=3240]
    (-112.0,    0.0),
    (-112.0,  -55.0),
    (-112.0,  -90.0),
    (-112.0, -113.0),
    ( -90.0, -113.0),
    ( -65.0, -113.0),
    ( -30.0, -112.0),
    (   0.0, -112.0),
    (  25.0, -109.0),
    (  40.0, -100.0),
    (  46.0,  -90.0),
    # ── Tramo B: diagonal NE → carretera x=105  [IGUAL]
    (  65.0,  -43.0),
    (  92.0,  -33.0),
    ( 105.0,  -43.0),   # se salta junto con WP anterior con radius=18
    ( 105.0,  -15.0),
    ( 105.0,   20.0),
    ( 105.0,   58.0),
    # ── Tramo C: FIX — mantiene al coche en x=105 hasta la intersección NE ───
    ( 105.0,  111.0),   # 18m norte de (105,93) → avanza exactamente en y=93
    # ── Tramo D: carretera O confirmada en .wbt a y=93 ───────────────────────
    (  85.0,   93.0),   # StraightRoadSegment (85.5,93) confirmado en .wbt
    (  45.0,   93.0),
    (   5.0,   93.0),
    # ── Tramo E: curva SO hacia intersección NW (-45,45) ─────────────────────
    ( -35.0,   75.0),
    ( -45.0,   65.0),   # StraightRoadSegment (-45,64.5) confirmado en .wbt
    ( -45.0,   45.0),   # intersección NW (-45,45)
    ( -45.0,   25.0),   # StraightRoadSegment (-45,25.5) confirmado en .wbt
    # ── Tramo F: NW → Start ──────────────────────────────────────────────────
    ( -75.0,    8.0),
    (-112.0,    0.0),
]

# =============================================================================
# PARÁMETROS — state machine de intersecciones
# =============================================================================
# SE y NE excluidas de FSM: el coche las navega sólo con GPS waypoints
# NW también se navega por GPS (steer=0 en TURN es suficiente para recta NW)
WORLD_INTERS = []

INTER_APPROACH_DIST = 45.0   # m: empieza a preparar la maniobra
INTER_TURN_DIST     = 20.0   # m: ejecuta el giro (log muestra mín 16.6m alcanzado)
INTER_EXIT_DIST     = 40.0   # m: da por terminado el cruce
INTER_SPEED         = 20     # km/h durante la maniobra

# Frames que dura la maniobra de giro (32 ms × TURN_FRAMES)
# A 20 km/h = 5.56 m/s: 250 frames = 8 s = 44 m — suficiente para cruzar
TURN_FRAMES_STRAIGHT = 220   # ~7 s
TURN_FRAMES_TURN     = 260   # ~8.3 s

# Ángulo de volante para cada maniobra
STEER_FOR_CMD = {1: 0.00, 2: -0.38, 3: +0.38}  # STRAIGHT / LEFT / RIGHT

# Secuencia de maniobras por intersección (se rota en cada visita)
INTER_SEQ = [1, 2, 1, 3]   # STRAIGHT, LEFT, STRAIGHT, RIGHT

CMD_CONTINUE, CMD_STRAIGHT, CMD_LEFT, CMD_RIGHT = 0, 1, 2, 3
CMD_FULL  = {0:"CONTINUE", 1:"RECTO", 2:"IZQUIERDA", 3:"DERECHA"}
CMD_COLOR = {0:0x003300, 1:0x003366, 2:0x553300, 3:0x440044}
CMD_TCOL  = {0:0x00FF44, 1:0x44AAFF, 2:0xFFAA00, 3:0xFF66FF}
CMD_TARGET= {0:1500, 1:300, 2:350, 3:350}

# =============================================================================
# PARÁMETROS — captura de imágenes
# =============================================================================
CAPTURE_ROAD   = 5   # cada 5 frames en recta  (~6 img/s a 32 ms)
CAPTURE_INTER  = 3   # cada 3 frames en cruce  (~10 img/s)

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

robot    = Car()
driver   = Driver()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)
CAM_W, CAM_H = camera.getWidth(), camera.getHeight()

display = robot.getDevice("display_image")
DW, DH  = 200, 150

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

ds_rf = robot.getDevice("ds_right_front")
ds_rm = robot.getDevice("ds_right_mid")
ds_rr = robot.getDevice("ds_right_rear")
for _ds in [ds_rf, ds_rm, ds_rr]:
    _ds.enable(timestep)

# LiDAR frontal — evasión de obstáculos (SIN enablePointCloud → no freeze)
lidar = robot.getDevice("Sick LMS 291")
LIDAR_OK = lidar is not None
if LIDAR_OK:
    lidar.enable(timestep)
    print("[LIDAR] Sick LMS 291 habilitado — evasión frontal activa")
else:
    print("[LIDAR] WARNING: 'Sick LMS 291' no encontrado — evasión desactivada")

# =============================================================================
# RUTAS DE DATOS
# =============================================================================

CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "images"))
CSV_PATH  = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "dataset.csv"))
LOG_PATH  = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "autolog.csv"))
DEBUG_DIR = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "debug_frames"))
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Dataset CSV
csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
csv_file   = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow(["image_path", "steering_angle", "speed_kmh", "nav_command"])
    csv_file.flush()

# Autolog CSV
log_file   = open(LOG_PATH, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "frame", "sim_ms", "wall_t",
    "gps_x", "gps_y", "heading_deg",
    "state", "state_inter", "turn_frames",
    "error_norm", "smooth_center", "confidence",
    "avg_road_frac", "n_valid_rows", "avg_road_w_px",
    "is_intersection_algo", "found_seed",
    "gyro_yaw_rate",
    "ds_rf", "ds_rm", "ds_rr", "ds_steer",
    "lidar_fwd", "lidar_steer",
    "target_steer", "steer_delta", "steer",
    "nav_cmd", "cruise_speed",
    "dist_nearest_inter", "nearest_inter_id",
    "inter_is_ahead",
])
log_file.flush()

# Conteo de imágenes existentes
existing   = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
img_count  = len(existing)
cmd_counts = {0: 0, 1: 0, 2: 0, 3: 0}
if csv_exists:
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            try:
                cmd_counts[int(row["nav_command"])] += 1
            except Exception:
                pass

_wall_t0    = time.time()
_auto_calls = 0

print("=" * 62)
print("[CIL] Recolección autónoma de datos CIL — Equipo 25")
print(f"[CIL] Imágenes previas: {img_count}")
print("  F       — alternar AUTO / MANUAL")
print("  ← / →   — volante (solo MANUAL)")
print("  s/w/a/d — etiqueta: CONTINUE/RECTO/IZQ/DER (solo MANUAL)")
print("  i / k   — velocidad +/- 5 km/h")
print("  q       — guardar y salir")
print("=" * 62)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def nearest_world_inter(gps_pos):
    """Retorna (inter_dict, distancia_m) de la intersección más cercana del mundo."""
    if gps_pos is None:
        return None, float("inf")
    gx, gy = gps_pos[0], gps_pos[1]
    best, bd = None, float("inf")
    for inter in WORLD_INTERS:
        d = math.sqrt((gx - inter["x"])**2 + (gy - inter["y"])**2)
        if d < bd:
            bd, best = d, inter
    return best, round(bd, 2)


def road_center_x(bgr_frame):
    """
    Flood fill desde el coche → centro ponderado del carril.
    Retorna (error_norm, confidence, dbg_dict).
    """
    global smooth_center
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, road_bin = cv2.threshold(gray, ROAD_THRESH, 255, cv2.THRESH_BINARY_INV)

    seed_y = h - 2; seed_x = w // 2; found_seed = False
    for try_y in range(h - 2, h // 2, -3):
        if road_bin[try_y, seed_x] > 0:
            seed_y = try_y; found_seed = True; break
        row_dark = np.where(road_bin[try_y, w//4: 3*w//4] > 0)[0]
        if len(row_dark) > 0:
            seed_x = row_dark[len(row_dark)//2] + w//4
            seed_y = try_y; found_seed = True; break

    dbg = dict(avg_road_frac=0.0, n_valid_rows=0,
               n_total_rows=int(h * (ROI_BOT - ROI_TOP)),
               avg_road_w_px=0.0, is_intersection=False,
               found_seed=found_seed, road_mask=None)

    if not found_seed:
        smooth_center *= 0.85
        return smooth_center, 0.10, dbg

    connected = road_bin.copy()
    ff_mask   = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(connected, ff_mask, (int(seed_x), int(seed_y)), 128)
    road_conn = ((connected == 128).astype(np.uint8)) * 255

    row_start = int(h * ROI_TOP); row_end = int(h * ROI_BOT)
    n_rows    = row_end - row_start

    # Primer paso: medir anchos para calcular avg_road_frac
    road_widths = []
    row_data    = []   # (i, lx, rx) para reusar en el segundo paso
    for i, r in enumerate(range(row_start, row_end)):
        nz = np.where(road_conn[r, :] > 0)[0]
        if len(nz) < MIN_ROAD_PX:
            continue
        lx = float(nz[0]); rx = float(nz[-1])
        road_widths.append(rx - lx)
        row_data.append((i, lx, rx))

    avg_road_frac = (sum(road_widths) / len(road_widths)) / w if road_widths else 0.0

    # LANE_OFFSET gradual: pleno en recta normal, fade suave en zonas anchas (cruces/curvas grandes)
    # Se desactiva por completo sólo en áreas muy anchas (tanque industrial, cruce abierto) >0.92
    if avg_road_frac < 0.72:
        effective_offset = LANE_OFFSET
    elif avg_road_frac < 0.92:
        effective_offset = LANE_OFFSET * (1.0 - (avg_road_frac - 0.72) / 0.20)
    else:
        effective_offset = 0.0

    # Segundo paso: centros con offset efectivo
    centers = []; weights = []
    for (i, lx, rx) in row_data:
        cx = lx + (rx - lx) * (0.5 + effective_offset)
        centers.append(float(np.clip(cx, 0, w - 1)))
        weights.append((i / n_rows) ** 2 + 0.1)
    dbg.update(dict(
        avg_road_frac  = round(avg_road_frac, 4),
        n_valid_rows   = len(centers),
        n_total_rows   = n_rows,
        avg_road_w_px  = round(sum(road_widths) / len(road_widths), 1) if road_widths else 0.0,
        road_mask      = road_conn,
    ))

    if not centers:
        smooth_center *= 0.85
        return smooth_center, 0.20, dbg

    # Intersecciones detectadas por GPS (FSM) — ya no se usa avg_road_frac
    # para cortar el steering, porque causaba steer=0 en rectas con rf alto.
    if avg_road_frac > 0.75:
        dbg['is_intersection'] = True   # solo flag informativo para el log

    cx_w       = float(np.average(centers, weights=weights))
    confidence = min(1.0, len(centers) / (n_rows * 0.5))
    error_norm = (cx_w - w / 2.0) / (w / 2.0)
    delta      = error_norm - smooth_center
    if abs(delta) > MAX_JUMP:
        error_norm = smooth_center + MAX_JUMP * (1 if delta > 0 else -1)
    smooth_center = SMOOTH_ALPHA * error_norm + (1 - SMOOTH_ALPHA) * smooth_center
    return smooth_center, confidence, dbg


def plan_next_cmd(inter_id, counts):
    """Elige el siguiente nav_cmd para esta intersección priorizando el más escaso."""
    visits = inter_visit_count[inter_id]
    # Base sequence para variedad garantizada
    base_cmd = INTER_SEQ[visits % len(INTER_SEQ)]
    # Override si IZQ o DER están muy sub-representados vs STRAIGHT
    n_str = counts[CMD_STRAIGHT]
    n_izq = counts[CMD_LEFT]
    n_der = counts[CMD_RIGHT]
    if n_izq < n_str * 0.3 and base_cmd != CMD_LEFT:
        return CMD_LEFT
    if n_der < n_str * 0.3 and base_cmd != CMD_RIGHT:
        return CMD_RIGHT
    return base_cmd


def draw_hud(cmd, steer, speed, total, counts, dist_m, state_str):
    n = max(1, sum(counts.values()))
    display.setColor(CMD_COLOR[cmd])
    display.fillRectangle(0, 0, DW, DH)
    display.setColor(CMD_TCOL[cmd])
    display.drawText(f"{state_str} {CMD_FULL[cmd]}", 2, 2)
    display.setColor(0xFFFFFF)
    display.drawText(f"St:{steer:+.3f}  {speed:.0f}km/h", 2, 14)
    display.drawText(f"TOTAL:{total:,}", 2, 26)

    if dist_m < 12:
        display.setColor(0xFF0000)
        display.drawText(f"[EN CRUCE] {dist_m:.0f}m", 2, 38)
    elif dist_m < 42:
        display.setColor(0xFF8800)
        display.drawText(f"CRUCE: {dist_m:.0f}m", 2, 38)
    elif dist_m < 100:
        display.setColor(0xFFFF00)
        display.drawText(f"CRUCE: {dist_m:.0f}m", 2, 38)
    else:
        display.setColor(0x00FF44)
        display.drawText(f"CRUCE: {dist_m:.0f}m", 2, 38)

    prox = max(0.0, min(1.0, 1.0 - dist_m / 100.0))
    bw   = max(1, int(prox * (DW - 4)))
    col  = 0x00FF44 if prox < 0.45 else (0xFF8800 if prox < 0.85 else 0xFF0000)
    display.setColor(col)
    display.fillRectangle(2, 49, bw, 5)
    rem = DW - 4 - bw
    if rem > 0:
        display.setColor(0x222222)
        display.fillRectangle(2 + bw, 49, rem, 5)

    display.setColor(0x555555)
    display.drawLine(0, 57, DW, 57)
    for i, (c, lb) in enumerate([(0,"S"),(1,"W"),(2,"A"),(3,"D")]):
        y  = 60 + i * 26
        ct = counts[c]
        bw2 = max(1, int(min(1.0, ct / CMD_TARGET[c]) * (DW - 4)))
        rem2 = DW - 4 - bw2
        display.setColor(CMD_TCOL[c] if c == cmd else 0xAAAAAA)
        display.drawText(f"{lb}:{ct:4d}({ct/n*100:.0f}%)", 2, y)
        display.setColor(CMD_TCOL[c] if c == cmd else 0x444444)
        display.fillRectangle(2, y + 11, bw2, 4)
        if rem2 > 0:
            display.setColor(0x222222)
            display.fillRectangle(2 + bw2, y + 11, rem2, 4)


# =============================================================================
# ESTADO INICIAL
# =============================================================================

steer        = 0.0
nav_cmd      = CMD_CONTINUE
cruise_speed = CRUISE_SPEED
frame        = 0
gps_pos      = None
auto_mode    = True

# Heading — se estima de deltas GPS para evitar drift del giróscopo
# Inicial: coche en spawn rotation=0 0 1 -π/2 → apunta al sur (-Y)
heading      = -math.pi / 2   # rad [-π, π]
gps_history  = []              # historial de posiciones GPS (últimos 30 frames)

# Sensores laterales — valores por defecto (sin obstáculo = 5 m)
gyro_yaw     = 0.0
ds_rf_val    = 5.0
ds_rm_val    = 5.0
ds_rr_val    = 5.0
ds_steer     = 0.0
inter_ahead  = True   # flag para el log

# LiDAR — valores iniciales
lidar_fwd    = float("inf")   # distancia obstáculo adelante (m)
lidar_steer  = 0.0            # corrección de steer por LiDAR

# Estado GPS waypoints
current_wp_idx = 0        # índice actual en ROUTE_GPS_WAYPOINTS
gps_steer      = 0.0      # steering calculado por GPS heading

# State machine de intersecciones
# Estados: "ROAD" → "APPROACH" → "TURN" → "EXIT" → "ROAD"
fsm_state       = "ROAD"
fsm_inter       = None    # dict de la intersección activa
fsm_cmd         = CMD_CONTINUE
turn_frames     = 0
inter_visit_count = {w["id"]: 0 for w in WORLD_INTERS}

driver.setCruisingSpeed(cruise_speed)

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

while robot.step() != -1:
    frame += 1

    # ── GPS ──────────────────────────────────────────────────────────────────
    gps_vals = gps.getValues()
    if gps_vals and not any(math.isnan(v) for v in gps_vals):
        gps_pos = gps_vals

    nearest_inter, dist_inter = nearest_world_inter(gps_pos)
    gx = gps_pos[0] if gps_pos else float("nan")
    gy = gps_pos[1] if gps_pos else float("nan")

    # ── Heading (GPS deltas, sin drift) ──────────────────────────────────────
    if gps_pos:
        gps_history.append((gx, gy))
        if len(gps_history) > 30:
            gps_history.pop(0)
        if len(gps_history) >= 10:
            dx = gps_history[-1][0] - gps_history[-10][0]
            dy = gps_history[-1][1] - gps_history[-10][1]
            if dx * dx + dy * dy > 0.04:   # moviéndose ≥ 0.2 m en 10 frames
                heading = math.atan2(dy, dx)

    # Orientación del nearest_inter respecto al coche
    inter_ahead = True
    if nearest_inter is not None and gps_pos:
        dx_i = nearest_inter["x"] - gx
        dy_i = nearest_inter["y"] - gy
        inter_bear = math.atan2(dy_i, dx_i)
        hdg_diff = ((inter_bear - heading) + math.pi) % (2 * math.pi) - math.pi
        inter_ahead = abs(hdg_diff) < math.pi / 2

    # ── GPS Waypoint Steering ─────────────────────────────────────────────────
    # Navegación 100% por GPS — flood fill no controla el steering.
    # Solo se usa en ROAD/APPROACH/EXIT (en TURN el FSM fija steer directo).
    gps_steer = 0.0
    if gps_pos and auto_mode and fsm_state != "TURN":
        n_wp = len(ROUTE_GPS_WAYPOINTS)
        # Avanzar al siguiente waypoint si estamos dentro del radio
        safety = 0
        while safety < n_wp:
            wx, wy = ROUTE_GPS_WAYPOINTS[current_wp_idx % n_wp]
            if math.sqrt((gx - wx) ** 2 + (gy - wy) ** 2) < GPS_WP_RADIUS:
                current_wp_idx += 1
                safety += 1
            else:
                break
        wx, wy = ROUTE_GPS_WAYPOINTS[current_wp_idx % n_wp]
        target_hdg = math.atan2(wy - gy, wx - gx)
        gps_hdg_err = ((target_hdg - heading) + math.pi) % (2 * math.pi) - math.pi
        gps_steer = float(np.clip(-KP_GPS * gps_hdg_err, -MAX_ANGLE, MAX_ANGLE))

    # ── Gyro & DistanceSensors ────────────────────────────────────────────────
    gyro_yaw  = gyro.getValues()[2]          # yaw rate rad/s (solo para log)
    ds_rf_val = ds_rf.getValue()
    ds_rm_val = ds_rm.getValue()
    ds_rr_val = ds_rr.getValue()
    # Corrección lateral: ds_val > DS_TARGET → coche muy lejos del borde → steer RIGHT (+)
    # ds_val < DS_TARGET → coche muy cerca del borde → steer LEFT (-)
    if 0.15 < ds_rf_val < 4.5:
        ds_steer = float(np.clip(KP_DS * (ds_rf_val - DS_TARGET), -0.12, 0.12))
    else:
        ds_steer = 0.0

    # ── LiDAR — evasión de obstáculos frontales ───────────────────────────────
    lidar_steer = 0.0
    lidar_fwd   = float("inf")
    if LIDAR_OK and fsm_state != "TURN":
        lr = lidar.getRangeImage()
        if lr:
            n_l = len(lr)   # SickLms291 → 181 rayos, -90° a +90°
            mid = n_l // 2  # rayo central = recto adelante
            # Sectores: frontal ±25°, lateral izq/der ±(25°-75°)
            def _valid(rays):
                return [r for r in rays if 0.3 < r < 29.9 and not math.isnan(r)]
            fwd_v   = _valid(lr[mid - 25 : mid + 26])
            left_v  = _valid(lr[mid - 75 : mid - 24])
            right_v = _valid(lr[mid + 25 : mid + 76])
            lidar_fwd   = min(fwd_v)   if fwd_v   else float("inf")
            min_left    = min(left_v)  if left_v  else float("inf")
            min_right   = min(right_v) if right_v else float("inf")
            if lidar_fwd < LIDAR_OBS_DIST:
                intensity   = max(0.0, 1.0 - lidar_fwd / LIDAR_OBS_DIST)
                sign        = -1.0 if min_left >= min_right else 1.0  # izq si hay más espacio izq
                lidar_steer = float(np.clip(sign * LIDAR_STEER * intensity,
                                            -MAX_ANGLE, MAX_ANGLE))

    # ── Teclado ───────────────────────────────────────────────────────────────
    steer_pressed = False
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.LEFT:
            steer = max(-MAX_ANGLE, steer - STEER_STEP); steer_pressed = True
        elif key == Keyboard.RIGHT:
            steer = min(MAX_ANGLE, steer + STEER_STEP);  steer_pressed = True
        elif key == ord('I') or key == ord('i'):
            cruise_speed = min(SPEED_MAX, cruise_speed + SPEED_STEP)
            driver.setCruisingSpeed(cruise_speed)
            print(f"[SPD] {cruise_speed} km/h")
        elif key == ord('K') or key == ord('k'):
            cruise_speed = max(SPEED_MIN, cruise_speed - SPEED_STEP)
            driver.setCruisingSpeed(cruise_speed)
            print(f"[SPD] {cruise_speed} km/h")
        elif key == ord('S') or key == ord('s'):
            if not auto_mode:
                nav_cmd = CMD_CONTINUE; print("[NAV] CONTINUE")
        elif key == ord('W') or key == ord('w'):
            if not auto_mode:
                nav_cmd = CMD_STRAIGHT; print("[NAV] RECTO")
        elif key == ord('A') or key == ord('a'):
            if not auto_mode:
                nav_cmd = CMD_LEFT;     print("[NAV] IZQUIERDA")
        elif key == ord('D') or key == ord('d'):
            if not auto_mode:
                nav_cmd = CMD_RIGHT;    print("[NAV] DERECHA")
        elif key == ord('F') or key == ord('f'):
            auto_mode = not auto_mode
            smooth_center = 0.0; steer = 0.0
            if auto_mode:
                fsm_state = "ROAD"; fsm_inter = None; turn_frames = 0
            else:
                nav_cmd = CMD_CONTINUE
            print(f"[MODE] {'AUTO' if auto_mode else 'MANUAL'}")
        elif key == ord('Q') or key == ord('q'):
            csv_file.close(); log_file.close()
            print(f"[CIL] Fin. {img_count} imgs | "
                  f"S:{cmd_counts[0]} W:{cmd_counts[1]} A:{cmd_counts[2]} D:{cmd_counts[3]}")
            driver.setCruisingSpeed(0)
            break
        key = keyboard.getKey()

    # ── State machine de intersecciones (solo en auto) ────────────────────────
    if auto_mode:

        if fsm_state == "ROAD":
            nav_cmd = CMD_CONTINUE
            if dist_inter < INTER_APPROACH_DIST and nearest_inter is not None and inter_ahead:
                fsm_inter = nearest_inter
                fsm_cmd   = plan_next_cmd(fsm_inter["id"], cmd_counts)
                fsm_state = "APPROACH"
                driver.setCruisingSpeed(INTER_SPEED)
                print(f"[FSM] APPROACH → {fsm_inter['id']}  cmd={CMD_FULL[fsm_cmd]}  d={dist_inter:.1f}m  hdg={math.degrees(heading):.1f}°")

        elif fsm_state == "APPROACH":
            nav_cmd = fsm_cmd
            if dist_inter < INTER_TURN_DIST:
                fsm_state   = "TURN"
                turn_frames = 0
                smooth_center = 0.0  # reset EMA antes de tomar el mando
                print(f"[FSM] TURN → {CMD_FULL[fsm_cmd]}")

        elif fsm_state == "TURN":
            nav_cmd = fsm_cmd
            turn_frames += 1
            max_tf = TURN_FRAMES_STRAIGHT if fsm_cmd == CMD_STRAIGHT else TURN_FRAMES_TURN
            if turn_frames >= max_tf:
                inter_visit_count[fsm_inter["id"]] += 1
                fsm_state = "EXIT"
                driver.setCruisingSpeed(cruise_speed)
                print(f"[FSM] EXIT  visitas {fsm_inter['id']}={inter_visit_count[fsm_inter['id']]}")

        elif fsm_state == "EXIT":
            nav_cmd = CMD_CONTINUE
            if dist_inter > INTER_EXIT_DIST:
                fsm_state = "ROAD"
                fsm_inter = None
                smooth_center = 0.0

    # ── Steering ─────────────────────────────────────────────────────────────
    error_norm = 0.0; conf = 0.0
    dbg = dict(avg_road_frac=0.0, n_valid_rows=0, n_total_rows=0,
               avg_road_w_px=0.0, is_intersection=False, found_seed=False, road_mask=None)
    target = 0.0; delta = 0.0

    if auto_mode:
        if fsm_state == "TURN":
            # Maniobra: steer fijo con rampa suave de entrada/salida
            max_tf  = TURN_FRAMES_STRAIGHT if fsm_cmd == CMD_STRAIGHT else TURN_FRAMES_TURN
            tgt_raw = STEER_FOR_CMD[fsm_cmd]
            ramp    = 25
            if turn_frames < ramp:
                tgt_raw = tgt_raw * (turn_frames / ramp)
            elif turn_frames > max_tf - ramp:
                tgt_raw = tgt_raw * max(0.0, (max_tf - turn_frames) / ramp)
            target = float(np.clip(tgt_raw, -MAX_ANGLE, MAX_ANGLE))
            delta  = float(np.clip(target - steer, -RATE_LIMIT * 2, RATE_LIMIT * 2))
            steer  = steer + delta

        elif frame % 3 == 0:
            # ── Cámara: leer imagen para guardar en dataset + log/debug ───────
            raw_a = camera.getImage()
            bgr_a = cv2.cvtColor(
                np.frombuffer(raw_a, np.uint8).reshape((CAM_H, CAM_W, 4)),
                cv2.COLOR_BGRA2BGR)
            # Flood fill: sólo para diagnóstico en log/debug (NO controla steer)
            error_norm, conf, dbg = road_center_x(bgr_a)
            _auto_calls += 1

            # ── STEERING: 100% GPS — no flood fill ────────────────────────────
            # gps_steer ya fue calculado arriba (bloque GPS Waypoint Steering)
            target = gps_steer
            delta  = float(np.clip(target - steer, -RATE_LIMIT, RATE_LIMIT))
            steer  = steer + delta

            # ── Debug frame cada 30 llamadas ──────────────────────────────────
            if _auto_calls % 30 == 0:
                wx, wy = ROUTE_GPS_WAYPOINTS[current_wp_idx % len(ROUTE_GPS_WAYPOINTS)]
                dbg_img = bgr_a.copy()
                # Línea verde = heading actual; línea roja = heading a waypoint
                cv2.putText(dbg_img,
                    f"GPS NAV  wp={current_wp_idx%len(ROUTE_GPS_WAYPOINTS)} -> ({wx:.0f},{wy:.0f})",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
                cv2.putText(dbg_img,
                    f"st={steer:+.3f} gps_st={gps_steer:+.3f}  hdg={math.degrees(heading):.0f}deg",
                    (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)
                cv2.putText(dbg_img,
                    f"({gx:.1f},{gy:.1f}) {fsm_state}  rf={dbg['avg_road_frac']:.2f}",
                    (4, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
                cv2.imwrite(os.path.join(DEBUG_DIR, f"dbg_{frame:07d}.jpg"),
                            dbg_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    else:  # MANUAL
        if not steer_pressed:
            steer *= CENTER_DECAY
            if abs(steer) < 0.003:
                steer = 0.0

    driver.setSteeringAngle(steer)

    # ── Velocidad dinámica: reducir ante obstáculo muy cercano ───────────────
    if auto_mode and fsm_state != "TURN":
        spd = max(10, cruise_speed // 3) if lidar_fwd < 4.0 else cruise_speed
        driver.setCruisingSpeed(spd)

    # ── Log autolog ───────────────────────────────────────────────────────────
    if auto_mode and frame % 3 == 0:
        sim_ms = frame * timestep
        log_writer.writerow([
            frame, sim_ms, round(time.time() - _wall_t0, 3),
            round(gx, 3), round(gy, 3), round(math.degrees(heading), 1),
            fsm_state, fsm_inter["id"] if fsm_inter else "",
            turn_frames,
            round(error_norm, 5), round(smooth_center, 5), round(conf, 4),
            dbg["avg_road_frac"], dbg["n_valid_rows"], dbg["avg_road_w_px"],
            int(dbg["is_intersection"]), int(dbg["found_seed"]),
            round(gyro_yaw, 5),
            round(ds_rf_val, 3), round(ds_rm_val, 3), round(ds_rr_val, 3),
            round(ds_steer, 5),
            round(lidar_fwd, 2) if lidar_fwd != float("inf") else 30.0,
            round(lidar_steer, 5),
            round(target, 5), round(delta, 5), round(steer, 5),
            nav_cmd, cruise_speed,
            round(dist_inter, 2), nearest_inter["id"] if nearest_inter else "",
            int(inter_ahead),
        ])
        if frame % 60 == 0:
            log_file.flush()

    # ── Captura de imagen ────────────────────────────────────────────────────
    capture_rate = CAPTURE_INTER if fsm_state in ("APPROACH", "TURN") else CAPTURE_ROAD
    if frame % capture_rate == 0:
        raw = camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        name = f"img_{img_count:06d}.jpg"
        cv2.imwrite(os.path.join(DATA_DIR, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        spd = driver.getCurrentSpeed()
        csv_writer.writerow([f"data/images/{name}", round(steer, 5), round(spd, 2), nav_cmd])
        if frame % 30 == 0:
            csv_file.flush()
        cmd_counts[nav_cmd] += 1
        img_count += 1
        draw_hud(nav_cmd, steer, spd, img_count, cmd_counts, dist_inter, fsm_state)
        if img_count % 200 == 0:
            print(f"[CIL] {img_count} imgs | "
                  f"S:{cmd_counts[0]} W:{cmd_counts[1]} "
                  f"A:{cmd_counts[2]} D:{cmd_counts[3]}")

csv_file.close()
log_file.close()
