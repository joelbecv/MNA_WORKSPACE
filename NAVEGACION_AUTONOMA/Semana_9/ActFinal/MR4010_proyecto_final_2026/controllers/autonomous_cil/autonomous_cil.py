# =============================================================================
# autonomous_cil.py — Controlador CIL + evasión de obstáculos
# Proyecto Final MR4010 · Equipo 25
# Tecnológico de Monterrey MNA — 2026
# =============================================================================
# EVASIÓN open-loop 5 pasos: L1→L2 (4m izq) → S (10m recto) → R1→R2 (4m der) → RECENTER
# =============================================================================

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os
import math

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

# ── Parámetros CIL ────────────────────────────────────────────────────────────
CRUISE_SPEED     = 25
MAX_ANGLE        = 0.5
IMG_W, IMG_H     = 200, 88
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STEER_RATE_LIMIT = 0.08   # solo para CIL_DRIVE, NO se usa en EVADE

CMD_CONTINUE = 0
CMD_STRAIGHT = 1
CMD_LEFT     = 2
CMD_RIGHT    = 3

# ── Detección de bus por Recognition ──────────────────────────────────────────
# recognitionColors tomados directamente del .wbt (vehicles 6, 7, 8)
BUS_COLOR_MAP = {
    "vehicle(6)": (0.878431, 0.105882, 0.141176),  # rojo
    "vehicle(7)": (0.149020, 0.635294, 0.411765),  # verde
    "vehicle(8)": (0.898039, 0.647059, 0.039216),  # naranja
}
BUS_COLOR_TOL      = 0.08  # tolerancia RGB por canal
BUS_MIN_PX_AREA    = 400   # igual que Act 4.2 — detectar desde lejos
BUS_CONFIRM_FRAMES = 10    # frames consecutivos para confirmar

# Frames de arranque antes de habilitar detección — evita disparar EVADE
# si un bus es visible desde la posición inicial del spawn
STARTUP_FRAMES     = 120   # ~4 s a 30 fps

# ── Radar ─────────────────────────────────────────────────────────────────────
RADAR_SAFE_M = 15.0
RADAR_STOP_M =  5.0

# ── Peatón ───────────────────────────────────────────────────────────────────
PED_CONFIRM_N    = 2
PED_RELEASE_N    = 4
PED_HOLD_F       = 100
PED_DETECT_EVERY = 10
PEDESTRIAN_KEYWORDS = ["pedestrian", "Pedestrian", "human", "person"]

# ── LiDAR ─────────────────────────────────────────────────────────────────────
LIDAR_RAYS_N     = 181
LIDAR_FOV_DEG    = 20    # sector frontal ±20° (igual Act 4.2)
LIDAR_LAT_START  = 150   # sector lateral derecho puro 65°-90° (igual Act 4.2)
OBS_LIDAR_THRESH = 14.5  # m — solo como respaldo para obstáculos sin Recognition

# ── Evasión — open-loop de 5 pasos ───────────────────────────────────────────
# 1. L1: steer=-STEER_EVADE durante N_TURN frames  (giro izquierda ~2m lateral)
# 2. L2: steer=+STEER_EVADE durante N_TURN frames  (enderezo, heading restaurado, ~4m izq total)
# 3. S : steer=0 — espera hasta lidar_f > S_CLEAR_DIST O timeout N_STRAIGHT_MAX
# 4. R1: steer=+STEER_EVADE durante N_TURN frames  (giro derecha ~2m lateral)
# 5. R2: steer=-STEER_EVADE durante N_TURN frames  (enderezo, heading y carril restaurados)
# → directo a RECENTER → CIL (cooldown BUS_COOLDOWN_FRAMES para no re-trigger)
#
# S sale sensor-based: al menos N_STRAIGHT frames Y lidar_f > S_CLEAR_DIST (camino libre)
# Si en N_STRAIGHT_MAX frames no se despeja, fuerza R1 igual (safety fallback)
SPEED_EVADE         = 15    # km/h
STEER_EVADE         = 0.30  # rad — ángulo giro izquierda/derecha
N_TURN              = 90    # frames cada medio giro (~2m lateral)
N_STRAIGHT          = 150   # frames mínimos en S antes de checar sensor
N_STRAIGHT_MAX      = 600   # timeout máximo de S (safety fallback)
S_CLEAR_DIST        = 20.0  # m — lidar_f debe superar este valor para salir de S
N_EXTRA             = 250   # frames rectilíneos adicionales DESPUÉS de que lidar_f > S_CLEAR_DIST
                             # Bus frente: X=-21.77m; coche sale S en X≈-31.00m → necesita ~9.23m
                             # 250f × 0.0667m = 16.7m → coche entra a R1 en X≈-14.3m (4.9m past bus front)
BUS_COOLDOWN_FRAMES = 300   # frames sin re-trigger después de EVADE

# ── Recentrado de carril ──────────────────────────────────────────────────────
RECENTER_STEER = 0.10   # deriva suave a la derecha para buscar línea amarilla
RECENTER_N     = 60     # frames máx RECENTER

# ── Estados ───────────────────────────────────────────────────────────────────
STATE_CIL      = "CIL_DRIVE"
STATE_PED      = "BRAKE_PED"
STATE_EVADE    = "EVADE_OBS"
STATE_REOR     = "REORIENT"
STATE_RECENTER = "RECENTER"

# =============================================================================
# INICIALIZACIÓN WEBOTS
# =============================================================================

driver   = Driver()
timestep = int(driver.getBasicTimeStep())

# Cámara — recognition cada step para no perder el bus
camera = driver.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
CAM_W = camera.getWidth()
CAM_H = camera.getHeight()

radar = driver.getDevice("radar")
radar.enable(timestep)

lidar = driver.getDevice("Sick LMS 291")
lidar.enable(timestep)

# Sensores laterales derechos (mismo nombre que Act 4.2: ds_right_front/mid/rear)
ds_rf = driver.getDevice("ds_right_front")
ds_rm = driver.getDevice("ds_right_mid")
ds_rr = driver.getDevice("ds_right_rear")
for ds in [ds_rf, ds_rm, ds_rr]:
    ds.enable(timestep)

gyro = driver.getDevice("gyro")
gyro.enable(timestep)

display = driver.getDevice("display_image")
DW = display.getWidth()
DH = display.getHeight()

keyboard = driver.getKeyboard()
keyboard.enable(timestep)

# =============================================================================
# MODELO CIL
# =============================================================================

CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(CTRL_DIR, "..", "..", "models", "cil_model_equipo25.h5"))

cil_model = None
if os.path.exists(MODEL_PATH):
    try:
        cil_model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
        print(f"[CIL] Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        print(f"[CIL] Error: {e}")
else:
    print(f"[CIL] ADVERTENCIA: modelo no encontrado en {MODEL_PATH}")


def cil_predict(bgr_frame, nav_cmd, speed_kmh):
    if cil_model is None:
        return 0.0
    rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(rgb, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    img  = (img - MEAN) / STD
    img  = img[np.newaxis, ...]
    spd  = np.array([[speed_kmh / 30.0]], dtype=np.float32)
    cmd  = np.zeros((1, 4), dtype=np.float32)
    cmd[0, nav_cmd] = 1.0
    pred = cil_model({'image': img, 'speed': spd, 'command': cmd}, training=False)
    return float(np.clip(float(pred.numpy()[0, 0]) * MAX_ANGLE, -MAX_ANGLE, MAX_ANGLE))

# =============================================================================
# FUNCIONES DE SENSORES
# =============================================================================

def get_lidar_ranges():
    """Retorna (dist_frontal, dist_lateral_derecha) igual que Act 4.2."""
    ranges = lidar.getRangeImage()
    if not ranges:
        return 999.0, 999.0
    n       = len(ranges)
    c       = n // 2
    span    = max(1, int(n * LIDAR_FOV_DEG / 180))
    front   = [r for r in ranges[max(0, c-span): c+span] if 0.1 < r < 99.0]
    r_start = n * LIDAR_LAT_START // 180
    lat     = [r for r in ranges[r_start:] if 0.1 < r < 99.0]
    return (min(front) if front else 999.0,
            min(lat)   if lat   else 999.0)


def get_radar_min_dist():
    tgts = radar.getTargets()
    return min((t.distance for t in tgts), default=999.0)


def detect_bus():
    """Recognition con los mismos filtros que Act 4.2.
    Sin filtro de posición horizontal (bus en paradero aparece al costado derecho).
    Retorna (detected: bool, max_area: int)."""
    max_area = 0
    for obj in camera.getRecognitionObjects():
        sz      = obj.getSizeOnImage()
        px_area = sz[0] * sz[1]
        if px_area > max_area:
            max_area = px_area
        if px_area < BUS_MIN_PX_AREA:
            continue
        n_col = obj.getNumberOfColors()
        if n_col < 1:
            continue
        col = obj.getColors()
        r, g, b = col[0], col[1], col[2]
        for (cr, cg, cb) in BUS_COLOR_MAP.values():
            if (abs(r-cr) < BUS_COLOR_TOL and
                abs(g-cg) < BUS_COLOR_TOL and
                abs(b-cb) < BUS_COLOR_TOL):
                return True, px_area
    return False, max_area


def check_pedestrian():
    for obj in camera.getRecognitionObjects():
        name = obj.getModel() if hasattr(obj, "getModel") else ""
        if any(kw in name for kw in PEDESTRIAN_KEYWORDS):
            return True
    return False


def apply_rate_limit(current, target, limit):
    """Solo para CIL_DRIVE — NO se usa en EVADE."""
    return current + float(np.clip(target - current, -limit, limit))


def draw_hud(state, steer, lidar_f, bus_area, right_dist, engaged):
    colors = {STATE_CIL: 0x002200, STATE_PED: 0x440000,
              STATE_EVADE: 0x002244, STATE_REOR: 0x222200, STATE_RECENTER: 0x222244}
    display.setColor(colors.get(state, 0x000000))
    display.fillRectangle(0, 0, DW, DH)
    display.setColor(0xFFFFFF)
    display.drawText(state[:12], 2, 2)
    display.drawText(f"St:{steer:+.3f}", 2, 16)
    display.drawText(f"LiF:{lidar_f:.1f}", 2, 30)
    display.drawText(f"Ri:{right_dist:.1f}", 2, 44)
    display.setColor(0xFF8800 if bus_area > 0 else 0x44FF44)
    display.drawText(f"Bus:{bus_area}px", 2, 58)
    display.setColor(0xFF4444 if engaged else 0x44FF44)
    display.drawText(f"Wall:{'ON' if engaged else 'off'}", 2, 72)

# =============================================================================
# ESTADO INICIAL
# =============================================================================

state            = STATE_CIL
nav_cmd          = CMD_CONTINUE
current_steer    = 0.0
ped_pos_streak   = 0
ped_neg_streak   = 0
ped_hold_count   = 0
_bus_streak      = 0
_evade_sub       = "L1"   # sub-estado EVADE: "L1"|"L2"|"S"|"R1"|"R2"
_evade_count     = 0      # frames en el sub-estado actual
_s_clear_count   = 0      # frames consecutivos con lidar_f > S_CLEAR_DIST durante S
_bus_cooldown    = 0      # frames restantes de cooldown post-EVADE
_recenter_frames = 0
heading_accum    = 0.0
heading_ref      = 0.0
frame_count      = 0

# Log de diagnóstico — mismo estilo que act42_wall.log de Act 4.2
_LOG_PATH = os.path.join(CTRL_DIR, "evade_log.csv")
_LOG = open(_LOG_PATH, "w")
_LOG.write("frame,state,lidar_f,lidar_lat,ds_rf,ds_rm,ds_rr,right_dist,bus_area,engaged,steer\n")
_LOG.flush()

print("=" * 60)
print("[AUTO] CIL + Evasión Equipo 25 — MR4010")
print(f"[AUTO] Modelo: {'CARGADO' if cil_model else 'NO ENCONTRADO'}")
print(f"[AUTO] BUS_MIN_PX={BUS_MIN_PX_AREA} | STEER_EVADE={STEER_EVADE} | "
      f"N_TURN={N_TURN}f N_STRAIGHT={N_STRAIGHT}f | cooldown={BUS_COOLDOWN_FRAMES}f")
print(f"[AUTO] gyro[1]=yaw | lidar_lat idx {LIDAR_LAT_START}-180")
print("  s=CONTINUE  w=RECTO  a=IZQ  d=DER  q=detener")
print("=" * 60)

driver.setCruisingSpeed(CRUISE_SPEED)

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

while driver.step() != -1:
    frame_count += 1

    # ── Teclado ───────────────────────────────────────────────────────────────
    key = keyboard.getKey()
    while key > 0:
        if   key in (ord('S'), ord('s')): nav_cmd = CMD_CONTINUE
        elif key in (ord('W'), ord('w')): nav_cmd = CMD_STRAIGHT
        elif key in (ord('A'), ord('a')): nav_cmd = CMD_LEFT
        elif key in (ord('D'), ord('d')): nav_cmd = CMD_RIGHT
        elif key in (ord('Q'), ord('q')):
            driver.setCruisingSpeed(0); driver.setBrakeIntensity(1.0)
        key = keyboard.getKey()

    # ── Imagen ────────────────────────────────────────────────────────────────
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Sensores ──────────────────────────────────────────────────────────────
    lidar_f, lidar_lat = get_lidar_ranges()
    radar_d            = get_radar_min_dist()
    ds_f_val           = float(ds_rf.getValue())
    ds_m_val           = float(ds_rm.getValue())
    ds_r_val           = float(ds_rr.getValue())

    # right_dist: mínimo de LiDAR lateral + ds_right_front — igual que Act 4.2
    right_dist = min(lidar_lat, ds_f_val)

    # Gyro — eje Y = yaw en BMW R2023b (validado en Act 4.2, comentario línea 498)
    gyro_vals      = gyro.getValues()
    heading_accum += gyro_vals[1] * (timestep / 1000.0)

    speed_now = driver.getCurrentSpeed()
    if math.isnan(speed_now) or speed_now < 0:
        speed_now = 0.0

    # Cooldown post-EVADE: decrementar cada frame
    if _bus_cooldown > 0:
        _bus_cooldown -= 1

    # ── Bus — recognition cada frame (igual Act 4.2: recognitionEnable(timestep)) ──
    bus_in_frame, bus_area = detect_bus()
    if bus_in_frame:
        _bus_streak += 1
    else:
        _bus_streak  = 0
    bus_confirmed = (_bus_streak >= BUS_CONFIRM_FRAMES)

    # ── Peatón cada PED_DETECT_EVERY frames ──────────────────────────────────
    ped_detected = False
    if frame_count % PED_DETECT_EVERY == 0:
        ped_detected = check_pedestrian()

    # ── Diagnóstico cada 30 frames en CIL_DRIVE ───────────────────────────────
    if state == STATE_CIL and frame_count % 30 == 0:
        print(f"[CIL f={frame_count}] lidar_f={lidar_f:.1f}m lat={lidar_lat:.1f}m "
              f"dsF={ds_f_val:.1f} dsR={ds_r_val:.1f} right={right_dist:.1f} "
              f"bus_streak={_bus_streak} area={bus_area}px")

    # =========================================================================
    # MÁQUINA DE ESTADOS
    # =========================================================================

    if state == STATE_CIL:

        # Peatón
        if ped_detected:
            ped_pos_streak += 1; ped_neg_streak = 0
        else:
            ped_neg_streak += 1; ped_pos_streak = 0
        if ped_pos_streak >= PED_CONFIRM_N:
            state = STATE_PED
            ped_hold_count = PED_HOLD_F
            print("[AUTO] PEATÓN → BRAKE_PED")

        # Detección solo después de STARTUP_FRAMES y sin cooldown activo
        elif frame_count > STARTUP_FRAMES and bus_confirmed and _bus_cooldown <= 0:
            state          = STATE_EVADE
            _evade_sub     = "L1"
            _evade_count   = 0
            _s_clear_count = 0
            _bus_streak    = 0
            print(f"[AUTO] Bus {bus_area}px → EVADE L1 (giro izq {N_TURN}f → recto {N_STRAIGHT}f → giro der {N_TURN}f)")

        elif frame_count > STARTUP_FRAMES and lidar_f < OBS_LIDAR_THRESH and _bus_cooldown <= 0:
            state          = STATE_EVADE
            _evade_sub     = "L1"
            _evade_count   = 0
            _s_clear_count = 0
            print(f"[AUTO] LiDAR {lidar_f:.1f}m → EVADE L1")

        else:
            # CIL normal
            if radar_d < RADAR_STOP_M:
                driver.setCruisingSpeed(0)
                driver.setBrakeIntensity(0.5)
            elif radar_d < RADAR_SAFE_M:
                driver.setCruisingSpeed(CRUISE_SPEED * radar_d / RADAR_SAFE_M)
                driver.setBrakeIntensity(0.0)
            else:
                driver.setCruisingSpeed(CRUISE_SPEED)
                driver.setBrakeIntensity(0.0)
            target        = cil_predict(bgr, nav_cmd, speed_now)
            current_steer = apply_rate_limit(current_steer, target, STEER_RATE_LIMIT)
            driver.setSteeringAngle(current_steer)

    # ── BRAKE_PED ─────────────────────────────────────────────────────────────
    elif state == STATE_PED:
        driver.setCruisingSpeed(0)
        driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)
        current_steer  = 0.0
        ped_hold_count -= 1
        if not ped_detected:
            ped_neg_streak += 1
        else:
            ped_neg_streak = 0
        if ped_hold_count <= 0 and ped_neg_streak >= PED_RELEASE_N:
            state = STATE_CIL
            driver.setBrakeIntensity(0.0)
            ped_pos_streak = ped_neg_streak = 0
            print("[AUTO] Peatón despejado → CIL_DRIVE")
        elif ped_hold_count <= 0:
            ped_hold_count = 20

    # ── EVADE_OBS — secuencia open-loop de 5 pasos ───────────────────────────
    elif state == STATE_EVADE:
        driver.setCruisingSpeed(SPEED_EVADE)
        driver.setBrakeIntensity(0.0)
        _evade_count += 1

        # L1: girar izquierda (primer medio giro del S)
        if _evade_sub == "L1":
            steer = -STEER_EVADE
            if _evade_count >= N_TURN:
                _evade_sub = "L2"; _evade_count = 0
                print(f"[EVADE] L1→L2")

        # L2: girar derecha (segundo medio giro del S, heading vuelve al original)
        elif _evade_sub == "L2":
            steer = +STEER_EVADE
            if _evade_count >= N_TURN:
                _evade_sub = "S"; _evade_count = 0; _s_clear_count = 0
                print(f"[EVADE] L2→S  (4m izq completado)")

        # S: avanzar recto hasta que el bus esté COMPLETAMENTE atrás del coche
        # Bus front-left corner (X=-21.77) cruza 20° cuando coche llega a X≈-31.00m
        # → lidar_f salta a 999 PERO el bus sigue 9.23m adelante en X
        # Fix: contar N_EXTRA frames consecutivos con lidar_f > S_CLEAR_DIST
        # 250f × 0.0667m = 16.7m → coche entra a R1 con 4.9m de clearance past bus front
        # Timeout N_STRAIGHT_MAX = 600f (40m viaje total desde S start) → siempre supera el bus
        elif _evade_sub == "S":
            steer = 0.0
            frontal_clear = lidar_f > S_CLEAR_DIST
            if frontal_clear and _evade_count >= N_STRAIGHT:
                _s_clear_count += 1
            else:
                _s_clear_count = 0
            if _s_clear_count >= N_EXTRA:
                _evade_sub = "R1"; _evade_count = 0; _s_clear_count = 0
                print(f"[EVADE] S→R1  {N_EXTRA}f libres  lidar={lidar_f:.1f}m (bus superado)")
            elif _evade_count >= N_STRAIGHT_MAX:
                _evade_sub = "R1"; _evade_count = 0; _s_clear_count = 0
                print(f"[EVADE] S→R1  TIMEOUT lidar={lidar_f:.1f}m")

        # R1: girar derecha (primer medio giro de regreso)
        elif _evade_sub == "R1":
            steer = +STEER_EVADE
            if _evade_count >= N_TURN:
                _evade_sub = "R2"; _evade_count = 0
                print(f"[EVADE] R1→R2")

        # R2: girar izquierda (heading y carril restaurados)
        elif _evade_sub == "R2":
            steer = -STEER_EVADE
            if _evade_count >= N_TURN:
                state         = STATE_RECENTER
                _recenter_frames = 0
                _bus_cooldown = BUS_COOLDOWN_FRAMES
                heading_accum = 0.0
                print(f"[EVADE] R2 completo → RECENTER | cooldown={BUS_COOLDOWN_FRAMES}f")

        driver.setSteeringAngle(steer)
        current_steer = steer

        # Log y print cada 10 frames
        right_dist = min(lidar_lat, ds_f_val)
        _LOG.write(f"{frame_count},{state},{lidar_f:.2f},{lidar_lat:.2f},"
                   f"{ds_f_val:.2f},{ds_m_val:.2f},{ds_r_val:.2f},"
                   f"{right_dist:.2f},{bus_area},{_evade_sub},{current_steer:.3f}\n")
        _LOG.flush()
        if frame_count % 20 == 0:
            print(f"[EVADE {_evade_sub}({_evade_count})] lidar_f={lidar_f:.1f} steer={current_steer:+.3f}")

    # ── RECENTER — deriva +0.10 para volver a encontrar la línea amarilla ───────
    elif state == STATE_RECENTER:
        _recenter_frames += 1
        driver.setCruisingSpeed(SPEED_EVADE)
        driver.setBrakeIntensity(0.0)

        # Detectar líneas amarillas como en Act 4.2 (para salir antes del timeout)
        hsv         = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv,
                                  np.array([15, 80, 80]),
                                  np.array([35, 255, 255]))
        yellow_frac = float((yellow_mask[int(CAM_H * 0.6):] > 0).mean())

        driver.setSteeringAngle(RECENTER_STEER)  # +0.10 derecha (Act 4.2 línea 899)
        current_steer = RECENTER_STEER

        if frame_count % 20 == 0:
            print(f"[RECENTER] f={_recenter_frames} yfrac={yellow_frac:.3f}")

        # Salida: línea amarilla detectada (Act 4.2 línea 915) o timeout
        if yellow_frac > 0.015:
            state         = STATE_CIL
            heading_accum = 0.0
            driver.setCruisingSpeed(CRUISE_SPEED)
            print(f"[RECENTER] Línea detectada (yfrac={yellow_frac:.3f}) → CIL_DRIVE")
        elif _recenter_frames >= RECENTER_N:
            state         = STATE_CIL
            heading_accum = 0.0
            driver.setCruisingSpeed(CRUISE_SPEED)
            print("[RECENTER] Timeout → CIL_DRIVE")

    # ── HUD ───────────────────────────────────────────────────────────────────
    draw_hud(state, current_steer, lidar_f, bus_area, right_dist,
             state == STATE_EVADE and _evade_sub == "S")
