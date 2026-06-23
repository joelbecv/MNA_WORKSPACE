# =============================================================================
# auto_collect.py — Recolección autónoma CIL con seguridad multi-sensor
# Proyecto Final MR4010 — Equipo 25
# =============================================================================
# DETECCIÓN DE CARRIL: segmentación de asfalto oscuro vs terreno claro.
# Detecta bordes izquierdo y derecho del asfalto en múltiples filas y
# calcula el centro ponderado — funciona en curvas cerradas donde la
# línea amarilla sale del FOV.
# =============================================================================

from controller import Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2, os, csv, math, json

# =============================================================================
# PARÁMETROS DE CONDUCCIÓN
# =============================================================================

CRUISE_SPEED  = 18        # km/h normal
INTER_SPEED   = 10        # km/h en intersección
MAX_ANGLE     = 0.5       # rad
CAPTURE_EVERY = 3         # frames entre capturas

# PID carril
KP, KI, KD   = 0.55, 0.008, 0.01
RATE_LIMIT    = 0.12      # rad/frame — más agresivo para curvas cerradas

# Segmentación de asfalto
ROAD_THRESH   = 85        # píxeles < ROAD_THRESH → asfalto (oscuro)
MIN_ROAD_PX   = 30        # mínimo de píxeles de asfalto para validar fila
# Filas del ROI (fracción de altura de imagen)
ROI_TOP       = 0.55      # empezar análisis aquí (más arriba = más lookahead)
ROI_BOT       = 0.92      # hasta aquí

# Offset de setpoint: el centro del carril DERECHO está ligeramente a la
# derecha del centro del asfalto total (que incluye ambos carriles)
LANE_OFFSET   = 0.12      # fracción del ancho: positivo = más a la derecha

# =============================================================================
# PARÁMETROS DE SEGURIDAD
# =============================================================================

LIDAR_FOV_DEG = 25
LIDAR_STOP_M  = 4.0
LIDAR_SLOW_M  = 10.0
RADAR_STOP_M  = 4.0
RADAR_SLOW_M  = 12.0
DS_WALL_M     = 1.0
KP_WALL       = 0.06

GPS_X_MIN, GPS_X_MAX = -9999.0, 9999.0   # sin límite — el GPS no se usa como trigger
GPS_Y_MIN, GPS_Y_MAX = -9999.0, 9999.0

STUCK_SPEED_TH = 0.5
STUCK_FRAMES   = 80
REVERSE_FRAMES = 50
EMRG_HOLD      = 30

# =============================================================================
# PARÁMETROS DE INTERSECCIÓN
# =============================================================================

INTER_RADIUS   = 25.0
INTER_EXIT     = 38.0
TURN_FRAMES    = 90
RECOVER_FRAMES = 50

STEER_LEFT     = -0.42
STEER_RIGHT    = +0.42

CMD_CONTINUE, CMD_STRAIGHT, CMD_LEFT, CMD_RIGHT = 0, 1, 2, 3
TURN_CYCLE = [CMD_STRAIGHT, CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT]

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

robot    = Car()
driver   = Driver()
timestep = int(robot.getBasicTimeStep())
dt       = timestep / 1000.0

camera = robot.getDevice("camera")
camera.enable(timestep)
CAM_W, CAM_H = camera.getWidth(), camera.getHeight()

display = robot.getDevice("display_image")
DW = display.getWidth()
DH = display.getHeight()

gps = robot.getDevice("gps")
gps.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

lidar = robot.getDevice("Sick LMS 291")
lidar.enable(timestep)
LIDAR_RAYS = lidar.getNumberOfPoints()

radar = robot.getDevice("radar")
radar.enable(timestep)

ds_rf = robot.getDevice("ds_right_front")
ds_rm = robot.getDevice("ds_right_mid")
ds_rr = robot.getDevice("ds_right_rear")
for ds in [ds_rf, ds_rm, ds_rr]:
    ds.enable(timestep)

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# =============================================================================
# INTERSECCIONES
# =============================================================================

CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
INTER_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "intersections.json"))
intersections = json.load(open(INTER_PATH)) if os.path.exists(INTER_PATH) else []
inter_visits  = [0] * len(intersections)

def nearest_inter(gps_pos):
    if not intersections or gps_pos is None:
        return -1, 999.0
    gx, gy = gps_pos[0], gps_pos[1]
    dists  = [math.sqrt((gx-ix)**2 + (gy-iy)**2) for ix,iy in intersections]
    idx    = int(np.argmin(dists))
    return idx, dists[idx]

# =============================================================================
# DATASET
# =============================================================================

DATA_DIR = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "images"))
CSV_AUTO = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "dataset_auto.csv"))
os.makedirs(DATA_DIR, exist_ok=True)

existing  = [f for f in os.listdir(DATA_DIR) if f.startswith("img_") and f.endswith(".jpg")]
img_index = max((int(f[4:10]) for f in existing), default=-1) + 1

csv_new  = not os.path.exists(CSV_AUTO)
csv_file = open(CSV_AUTO, "a", newline="")
csv_w    = csv.writer(csv_file)
if csv_new:
    csv_w.writerow(["image_path", "steering_angle", "speed_kmh", "nav_command"])

print(f"[AUTO] img_index={img_index}  intersecciones={len(intersections)}")

# =============================================================================
# DETECCIÓN DE CARRIL — FLOOD FILL DESDE EL COCHE
# =============================================================================
# Flood fill desde el centro inferior aísla SOLO el asfalto conectado al
# coche. Objetos oscuros a los lados (paradas de bus, paredes, árboles)
# NO están conectados al asfalto debajo del coche → no los detecta.
# =============================================================================

SMOOTH_ALPHA  = 0.45   # EMA del centro: más bajo = más suave (más lag)
MAX_JUMP      = 0.30   # fracción máxima de salto por frame (outlier rejection)
smooth_center = 0.0    # estado del filtro EMA

def road_center_x(bgr_frame):
    """
    1. Umbral gris → máscara binaria asfalto
    2. Flood fill desde píxel-semilla en fila inferior → región conectada
    3. Análisis row-by-row solo sobre región conectada (elimina objetos laterales)
    4. EMA + rechazo de outliers para estabilidad ante perturbaciones bruscas
    """
    global smooth_center

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Máscara de asfalto ────────────────────────────────────────────────
    _, road_bin = cv2.threshold(gray, ROAD_THRESH, 255, cv2.THRESH_BINARY_INV)

    # ── Semilla: punto más bajo al centro del asfalto ─────────────────────
    seed_y = h - 2
    seed_x = w // 2
    found_seed = False

    for try_y in range(h - 2, h // 2, -3):
        if road_bin[try_y, seed_x] > 0:
            seed_y = try_y; found_seed = True; break
        # Buscar horizontalmente si el centro no es asfalto
        row_dark = np.where(road_bin[try_y, w//4: 3*w//4] > 0)[0]
        if len(row_dark) > 0:
            seed_x = row_dark[len(row_dark)//2] + w//4
            seed_y = try_y; found_seed = True; break

    if not found_seed:
        smooth_center *= 0.85
        return smooth_center, 0.1

    # ── Flood fill → solo región conectada al coche ───────────────────────
    connected = road_bin.copy()
    ff_mask   = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(connected, ff_mask, (int(seed_x), int(seed_y)), 128)
    road_conn = ((connected == 128).astype(np.uint8)) * 255

    # ── Análisis row-by-row en ROI ────────────────────────────────────────
    row_start = int(h * ROI_TOP)
    row_end   = int(h * ROI_BOT)
    n_rows    = row_end - row_start

    centers = []
    weights = []

    for i, r in enumerate(range(row_start, row_end)):
        nonzero = np.where(road_conn[r, :] > 0)[0]
        if len(nonzero) < MIN_ROAD_PX:
            continue

        left_x  = float(nonzero[0])
        right_x = float(nonzero[-1])
        road_w  = right_x - left_x

        cx = left_x + road_w * (0.5 + LANE_OFFSET)
        cx = float(np.clip(cx, 0, w - 1))

        weight = (i / n_rows) ** 2 + 0.1
        centers.append(cx)
        weights.append(weight)

    if not centers:
        smooth_center *= 0.85
        return smooth_center, 0.2

    cx_weighted = float(np.average(centers, weights=weights))
    confidence  = min(1.0, len(centers) / (n_rows * 0.5))

    # ── Normalizar ────────────────────────────────────────────────────────
    error_norm = (cx_weighted - w / 2.0) / (w / 2.0)

    # ── Outlier rejection: no saltar más de MAX_JUMP en un frame ─────────
    delta = error_norm - smooth_center
    if abs(delta) > MAX_JUMP:
        error_norm = smooth_center + MAX_JUMP * (1 if delta > 0 else -1)

    # ── EMA ───────────────────────────────────────────────────────────────
    smooth_center = SMOOTH_ALPHA * error_norm + (1 - SMOOTH_ALPHA) * smooth_center

    return smooth_center, confidence


# =============================================================================
# SENSORES DE SEGURIDAD
# =============================================================================

def lidar_front():
    ranges = lidar.getRangeImage()
    if not ranges: return 999.0
    c    = LIDAR_RAYS // 2
    half = int(LIDAR_FOV_DEG * LIDAR_RAYS / 180)
    cone = ranges[max(0, c-half): c+half]
    v    = [r for r in cone if 0.2 < r < 80.0]
    return min(v) if v else 999.0

def radar_front():
    return min((t.distance for t in radar.getTargets()), default=999.0)

def gps_ok(pos):
    if pos is None: return True
    return GPS_X_MIN < pos[0] < GPS_X_MAX and GPS_Y_MIN < pos[1] < GPS_Y_MAX

# =============================================================================
# PID
# =============================================================================

pid_integral   = 0.0
pid_prev_error = 0.0
no_road_frames = 0
current_steer  = 0.0

def pid_step(error_norm, confidence):
    global pid_integral, pid_prev_error, no_road_frames, current_steer

    if error_norm is not None and confidence > 0.2:
        no_road_frames = 0
        pid_integral   = float(np.clip(pid_integral + error_norm * dt, -0.5, 0.5))
        deriv          = (error_norm - pid_prev_error) / dt
        raw = float(np.clip(KP*error_norm + KI*pid_integral + KD*deriv,
                            -MAX_ANGLE, MAX_ANGLE))
        pid_prev_error = error_norm

        # Corrección por pared lateral derecha
        ds_m = ds_rm.getValue()
        if ds_m < DS_WALL_M:
            raw = float(np.clip(raw + KP_WALL*(DS_WALL_M - ds_m), -MAX_ANGLE, MAX_ANGLE))

        current_steer = current_steer + float(
            np.clip(raw - current_steer, -RATE_LIMIT, RATE_LIMIT))
    else:
        no_road_frames += 1
        pid_integral   *= 0.7
        pid_prev_error  = 0.0
        if no_road_frames > 15:
            current_steer *= 0.92   # decae — mantiene curvatura en intersección

    return current_steer

# =============================================================================
# ESTADOS
# =============================================================================

STATE_PID  = "PID"
STATE_TURN = "TURN"
STATE_REC  = "RECOVER"
STATE_EMRG = "EMERGENCY"
STATE_REV  = "REVERSE"

state          = STATE_PID
turn_cmd       = CMD_CONTINUE
turn_tgt       = 0.0
turn_frame     = 0
recover_frame  = 0
emrg_frame     = 0
rev_frame      = 0
last_inter_idx = -1
stuck_frames   = 0
nav_cmd_cap    = CMD_CONTINUE

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

frame    = 0
captured = 0
paused   = False
driver.setCruisingSpeed(CRUISE_SPEED)

print("=" * 60)
print("[AUTO] Segmentación asfalto + seguridad multi-sensor")
print(f"  LiDAR stop={LIDAR_STOP_M}m  Radar stop={RADAR_STOP_M}m")
print("  q=salir  p=pausa  r=reset ciclos")
print("=" * 60)

while robot.step() != -1:
    frame += 1

    # ── Teclado ───────────────────────────────────────────────────────────────
    key = keyboard.getKey()
    while key > 0:
        if key in (ord('Q'), ord('q')):
            csv_file.close(); driver.setCruisingSpeed(0)
            print(f"[AUTO] Fin. {captured} imágenes."); break
        elif key in (ord('P'), ord('p')):
            paused = not paused; print(f"[AUTO] {'PAUSA' if paused else 'REANUDADO'}")
        elif key in (ord('R'), ord('r')):
            inter_visits[:] = [0]*len(inter_visits); print("[AUTO] Ciclos reseteados")
        key = keyboard.getKey()

    # ── Sensores ──────────────────────────────────────────────────────────────
    raw_img  = camera.getImage()
    img      = np.frombuffer(raw_img, np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr      = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gps_vals = gps.getValues()
    gps_pos  = gps_vals if gps_vals and not any(math.isnan(v) for v in gps_vals) else None

    lidar_d  = lidar_front()
    radar_d  = radar_front()

    spd = driver.getCurrentSpeed()
    if math.isnan(spd) or spd < 0: spd = 0.0

    inter_idx, inter_dist = nearest_inter(gps_pos)

    # ── Anti-atasco ───────────────────────────────────────────────────────────
    if spd < STUCK_SPEED_TH and state not in (STATE_EMRG, STATE_REV):
        stuck_frames += 1
    else:
        stuck_frames = 0

    if stuck_frames >= STUCK_FRAMES and state == STATE_PID:
        state = STATE_REV; rev_frame = 0
        print("[SEG] Atascado → REVERSE")

    # ── Emergencia ────────────────────────────────────────────────────────────
    if (lidar_d < LIDAR_STOP_M or radar_d < RADAR_STOP_M or not gps_ok(gps_pos)) \
            and state not in (STATE_EMRG, STATE_REV):
        state = STATE_EMRG; emrg_frame = 0
        print(f"[SEG] Obstáculo L={lidar_d:.1f} R={radar_d:.1f} → EMERGENCY")

    # ── Velocidad adaptativa ──────────────────────────────────────────────────
    if state not in (STATE_EMRG, STATE_REV):
        if radar_d < RADAR_SLOW_M or lidar_d < LIDAR_SLOW_M:
            factor = min(radar_d/RADAR_SLOW_M, lidar_d/LIDAR_SLOW_M)
            driver.setCruisingSpeed(max(6, CRUISE_SPEED * factor))
        elif state == STATE_TURN:
            driver.setCruisingSpeed(INTER_SPEED)
        else:
            driver.setCruisingSpeed(CRUISE_SPEED)

    # ── Máquina de estados ────────────────────────────────────────────────────

    if state == STATE_EMRG:
        driver.setCruisingSpeed(0); driver.setBrakeIntensity(0.8)
        driver.setSteeringAngle(0.0)
        emrg_frame += 1; nav_cmd_cap = CMD_CONTINUE
        if emrg_frame >= EMRG_HOLD and lidar_d > LIDAR_STOP_M and radar_d > RADAR_STOP_M:
            driver.setBrakeIntensity(0.0); state = STATE_PID
            print("[SEG] Despejado → PID")

    elif state == STATE_REV:
        driver.setCruisingSpeed(-8)
        driver.setSteeringAngle(-current_steer * 0.4)
        rev_frame += 1; nav_cmd_cap = CMD_CONTINUE
        if rev_frame >= REVERSE_FRAMES:
            driver.setCruisingSpeed(CRUISE_SPEED)
            state = STATE_PID; stuck_frames = 0; pid_integral = 0.0
            print("[SEG] Reverse ok → PID")

    elif state == STATE_PID:
        err, conf = road_center_x(bgr)
        steer     = pid_step(err, conf)
        driver.setSteeringAngle(steer)
        nav_cmd_cap = CMD_CONTINUE

        if inter_dist < INTER_RADIUS and inter_idx != last_inter_idx:
            visit_n    = inter_visits[inter_idx]
            cycle_cmd  = TURN_CYCLE[visit_n % len(TURN_CYCLE)]
            inter_visits[inter_idx] += 1
            last_inter_idx = inter_idx

            if cycle_cmd == CMD_LEFT:
                turn_tgt = STEER_LEFT;  turn_cmd = CMD_LEFT
                print(f"[NAV] Inter#{inter_idx} v{visit_n} → IZQUIERDA")
            elif cycle_cmd == CMD_RIGHT:
                turn_tgt = STEER_RIGHT; turn_cmd = CMD_RIGHT
                print(f"[NAV] Inter#{inter_idx} v{visit_n} → DERECHA")
            else:
                turn_tgt = 0.0;         turn_cmd = CMD_STRAIGHT
                print(f"[NAV] Inter#{inter_idx} v{visit_n} → RECTO")

            state = STATE_TURN; turn_frame = 0; pid_integral = 0.0

    elif state == STATE_TURN:
        turn_frame += 1
        if turn_cmd == CMD_STRAIGHT:
            err, conf = road_center_x(bgr)
            steer = pid_step(err, conf)
        else:
            steer = current_steer + float(
                np.clip(turn_tgt - current_steer, -RATE_LIMIT*3, RATE_LIMIT*3))
            current_steer = steer
        driver.setSteeringAngle(steer)
        nav_cmd_cap = turn_cmd

        if turn_frame >= TURN_FRAMES:
            state = STATE_REC; recover_frame = 0; pid_integral = 0.0
            print("[NAV] Giro ok → RECOVER")

    elif state == STATE_REC:
        recover_frame += 1
        err, conf = road_center_x(bgr)
        steer = pid_step(err, conf)
        driver.setSteeringAngle(steer)
        nav_cmd_cap = CMD_CONTINUE
        if recover_frame >= RECOVER_FRAMES:
            state = STATE_PID

    if inter_dist > INTER_EXIT and last_inter_idx == inter_idx:
        last_inter_idx = -1

    if paused: continue

    # ── Captura ───────────────────────────────────────────────────────────────
    if frame % CAPTURE_EVERY == 0:
        name = f"img_{img_index:06d}.jpg"
        cv2.imwrite(os.path.join(DATA_DIR, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        csv_w.writerow([f"data/images/{name}", round(current_steer, 5),
                        round(spd, 2), nav_cmd_cap])
        csv_file.flush()
        img_index += 1; captured += 1

        # HUD
        col = {STATE_PID:0x002200, STATE_TURN:0x003366,
               STATE_REC:0x332200, STATE_EMRG:0x440000, STATE_REV:0x442200}
        display.setColor(col.get(state, 0x000000))
        display.fillRectangle(0, 0, DW, DH)
        display.setColor(0xFFFFFF)
        display.drawText(f"{state}", 2, 2)
        display.drawText(f"St:{current_steer:+.3f}", 2, 14)
        display.drawText(f"IMG:{captured}", 2, 26)
        display.drawText(f"L:{lidar_d:.1f} Ra:{radar_d:.1f}", 2, 38)
        display.drawText(f"I:{inter_idx} {inter_dist:.0f}m", 2, 50)
        display.drawText(f"V:{' '.join(str(v) for v in inter_visits)}", 2, 62)

        if captured % 500 == 0:
            print(f"[AUTO] {captured} imgs | {state} | "
                  f"L={lidar_d:.1f}m | visits={inter_visits}")

csv_file.close()
