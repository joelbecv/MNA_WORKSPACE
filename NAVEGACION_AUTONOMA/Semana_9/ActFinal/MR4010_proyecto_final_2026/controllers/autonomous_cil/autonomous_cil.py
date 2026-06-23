# =============================================================================
# autonomous_cil.py — Controlador autónomo CIL puro (sin PID)
# Proyecto Final MR4010 — Equipo 25
# Tecnológico de Monterrey MNA — 2026
# =============================================================================
#
# ARQUITECTURA (Codevilla ICRA 2018):
#   Imagen (88×200) + velocidad → CNN 8-capas → 4 ramas por comando → steering
#   Modelo: cil_model_equipo25.h5  (Keras / TensorFlow)
#
# MODOS DE NAVEGACIÓN (teclado en ventana 3D Webots):
#   s = CMD_CONTINUE  (seguir carretera, rama 0)
#   w = CMD_STRAIGHT  (cruzar intersección recto, rama 1)
#   a = CMD_LEFT      (girar izquierda en intersección, rama 2)
#   d = CMD_RIGHT     (girar derecha en intersección, rama 3)
#   q = detener
#
# ESTADOS DE SEGURIDAD (automáticos):
#   CIL_DRIVE → estado normal, CIL controla la dirección
#   BRAKE_PED → peatón detectado por reconocimiento de cámara, freno total
#   EVADE_OBS → obstáculo detectado por LiDAR < umbral, wall-following
#   REORIENT  → corrección de heading tras evasión (gyro)
#
# SENSORES USADOS:
#   Cámara  : imagen CIL + reconocimiento de objetos (peatones)
#   Radar   : distancia al vehículo más cercano → control de velocidad
#   LiDAR   : detección de obstáculos estáticos en frente
#   DS x3   : sensores laterales derechos para wall-following
#   Gyro    : acumulación de heading para reorientación
# =============================================================================

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os
import math

# TensorFlow / Keras para inferencia del modelo CIL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suprimir logs de TF en consola
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

CRUISE_SPEED = 25      # km/h velocidad de crucero
MAX_ANGLE    = 0.5     # rad límite físico del volante
IMG_W, IMG_H = 200, 88 # resolución del modelo (Codevilla 2017)

# Normalización ImageNet (misma que en train_cil.ipynb)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CMD_CONTINUE = 0
CMD_STRAIGHT = 1
CMD_LEFT     = 2
CMD_RIGHT    = 3
CMD_LABEL    = {0: "CONTINUE", 1: "RECTO", 2: "IZQUIERDA", 3: "DERECHA"}

# Suavizado de steering: evita cambios bruscos entre frames
STEER_RATE_LIMIT = 0.08   # rad/frame máximo cambio por paso

# =============================================================================
# PARÁMETROS — RADAR (seguimiento de vehículos, rúbrica Act 3)
# =============================================================================

RADAR_SAFE_M = 15.0   # por encima: velocidad normal
RADAR_STOP_M =  5.0   # por debajo: freno completo

# =============================================================================
# PARÁMETROS — PEATÓN (detección por cámara, rúbrica requerido)
# =============================================================================

PED_CONFIRM_N = 2     # detecciones consecutivas para activar freno
PED_RELEASE_N = 4     # no-detecciones para reanudar marcha
PED_HOLD_F    = 100   # frames mínimos de freno aunque desaparezca
DETECT_EVERY  = 10    # ejecutar reconocimiento cada N frames (costoso)
PEDESTRIAN_KEYWORDS = ["pedestrian", "Pedestrian", "human", "person"]

# =============================================================================
# PARÁMETROS — LIDAR / EVASIÓN DE OBSTÁCULOS (rúbrica requerido)
# =============================================================================

OBS_LIDAR_THRESH = 14.0  # m — distancia que activa evasión
LIDAR_FOV_DEG    = 20    # grados del cono frontal analizado
SPEED_EVADE      = 15    # km/h durante evasión
WALL_TARGET      = 2.9   # m distancia lateral objetivo en wall-follow
KP_WALL          = 0.10  # ganancia proporcional wall-follow
DS_CLEAR_DIST    = 4.8   # m — distancia derecha que indica obstáculo superado
DS_ENGAGE_DIST   = 4.5   # m — distancia frontal que permite salir de evasión

# =============================================================================
# PARÁMETROS — REORIENTACIÓN (corrección post-evasión)
# =============================================================================

SPEED_REORIENT = 20
KP_HEADING     = 1.0
HEADING_TOL    = 0.08   # rad — tolerancia para considerar heading corregido

# =============================================================================
# ESTADOS
# =============================================================================

STATE_CIL   = "CIL_DRIVE"
STATE_PED   = "BRAKE_PED"
STATE_EVADE = "EVADE_OBS"
STATE_REOR  = "REORIENT"

# =============================================================================
# INICIALIZACIÓN DE WEBOTS
# =============================================================================

driver   = Driver()
timestep = int(driver.getBasicTimeStep())

camera = driver.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep * DETECT_EVERY)
CAM_W = camera.getWidth()
CAM_H = camera.getHeight()

radar = driver.getDevice("radar")
radar.enable(timestep)

lidar = driver.getDevice("Sick LMS 291")
lidar.enable(timestep)
LIDAR_RAYS = lidar.getNumberOfPoints()

# Sensores laterales derechos para wall-following
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
# CARGA DEL MODELO CIL (Keras .h5)
# =============================================================================

CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(CTRL_DIR, "..", "..", "models", "cil_model_equipo25.h5"))

cil_model = None
if os.path.exists(MODEL_PATH):
    try:
        # compile=False: no necesitamos la función de loss para inferencia
        cil_model = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"[CIL] Modelo Keras cargado: {MODEL_PATH}")
        print(f"[CIL] Entradas: {[i.name for i in cil_model.inputs]}")
    except Exception as e:
        print(f"[CIL] Error cargando modelo: {e}")
else:
    print(f"[CIL] ADVERTENCIA — modelo no encontrado: {MODEL_PATH}")
    print("[CIL]   Entrena primero con code/train_cil.ipynb")


def cil_predict(bgr_frame, nav_cmd, speed_kmh):
    """
    Ejecuta inferencia CIL en un frame.
    Retorna steering en radianes [-MAX_ANGLE, +MAX_ANGLE].
    """
    if cil_model is None:
        return 0.0

    # Preprocesamiento idéntico al entrenamiento (train_cil.ipynb cell-4)
    rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(rgb, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    img  = (img - MEAN) / STD
    img  = img[np.newaxis, ...]                          # (1, H, W, 3)

    spd  = np.array([[speed_kmh / 30.0]], dtype=np.float32)
    cmd  = np.array([nav_cmd],            dtype=np.int32)

    # Llamada directa al modelo (más rápida que model.predict() en tiempo real)
    pred = cil_model({'image': img, 'speed': spd, 'command': cmd}, training=False)
    steer_norm = float(pred.numpy()[0, 0])               # normalizado [-1, 1]
    return float(np.clip(steer_norm * MAX_ANGLE, -MAX_ANGLE, MAX_ANGLE))


# =============================================================================
# FUNCIONES AUXILIARES DE SENSORES
# =============================================================================

def get_lidar_front_dist():
    """Distancia mínima en el cono frontal del LiDAR."""
    ranges = lidar.getRangeImage()
    if not ranges:
        return 999.0
    c    = LIDAR_RAYS // 2
    half = int(LIDAR_FOV_DEG * LIDAR_RAYS / 180)
    cone = ranges[max(0, c - half): c + half]
    valid = [r for r in cone if 0.1 < r < 80.0]
    return min(valid) if valid else 999.0


def get_radar_min_dist():
    """Distancia al objeto más cercano detectado por radar."""
    tgts = radar.getTargets()
    return min((t.distance for t in tgts), default=999.0)


def check_pedestrian():
    """Detecta peatones usando el nodo de reconocimiento de la cámara."""
    for obj in camera.getRecognitionObjects():
        name = obj.getModel() if hasattr(obj, "getModel") else ""
        if any(kw in name for kw in PEDESTRIAN_KEYWORDS):
            return True
    return False


def wall_follow_step(ds_m):
    """Steering proporcional para mantener WALL_TARGET metros del lado derecho."""
    return float(np.clip(KP_WALL * (WALL_TARGET - ds_m), -MAX_ANGLE, MAX_ANGLE))


def apply_rate_limit(current, target, limit):
    """Suaviza la transición de steering evitando cambios bruscos."""
    return current + float(np.clip(target - current, -limit, limit))


def draw_hud(state, nav_cmd, steer, radar_d, ped, lidar_d):
    """HUD en el display del vehículo: estado, comando, sensores."""
    bg = {STATE_CIL: 0x002200, STATE_PED: 0x440000,
          STATE_EVADE: 0x002244, STATE_REOR: 0x222200}
    display.setColor(bg.get(state, 0x000000))
    display.fillRectangle(0, 0, DW, DH)
    display.setColor(0xFFFFFF)
    display.drawText(f"{state}", 2, 2)
    display.drawText(f"NAV:{CMD_LABEL[nav_cmd]}", 2, 14)
    display.drawText(f"St:{steer:+.3f}r", 2, 26)
    display.drawText(f"Radar:{radar_d:.1f}m", 2, 38)
    display.drawText(f"LiDAR:{lidar_d:.1f}m", 2, 50)
    display.setColor(0xFF4444 if ped else 0x44FF44)
    display.drawText(f"Ped:{'DET' if ped else 'OK'}", 2, 62)
    display.setColor(0xFFFF00 if cil_model is None else 0x44FF44)
    display.drawText(f"CIL:{'NO MODEL' if cil_model is None else 'OK'}", 2, 74)


# =============================================================================
# ESTADO INICIAL
# =============================================================================

state          = STATE_CIL
nav_cmd        = CMD_CONTINUE
current_steer  = 0.0

ped_pos_streak = 0
ped_neg_streak = 0
ped_hold_count = 0
heading_accum  = 0.0
heading_ref    = 0.0
frame_count    = 0

print("=" * 60)
print("[AUTO] CIL puro — Equipo 25 — MR4010 Proyecto Final")
print(f"[AUTO] Modelo: {'CARGADO' if cil_model else 'NO ENCONTRADO'}")
print(f"[AUTO] Radar stop={RADAR_STOP_M}m  LiDAR evasión={OBS_LIDAR_THRESH}m")
print("  s=CONTINUE  w=RECTO  a=IZQ  d=DER  q=detener")
print("=" * 60)

driver.setCruisingSpeed(CRUISE_SPEED)

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

while driver.step() != -1:
    frame_count += 1

    # ── Teclado: comandos de navegación ───────────────────────────────────────
    key = keyboard.getKey()
    while key > 0:
        if   key in (ord('S'), ord('s')):
            nav_cmd = CMD_CONTINUE; print("[NAV] CONTINUE")
        elif key in (ord('W'), ord('w')):
            nav_cmd = CMD_STRAIGHT; print("[NAV] RECTO")
        elif key in (ord('A'), ord('a')):
            nav_cmd = CMD_LEFT;     print("[NAV] IZQUIERDA")
        elif key in (ord('D'), ord('d')):
            nav_cmd = CMD_RIGHT;    print("[NAV] DERECHA")
        elif key in (ord('Q'), ord('q')):
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
        key = keyboard.getKey()

    # ── Captura de imagen ─────────────────────────────────────────────────────
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Lectura de sensores ───────────────────────────────────────────────────
    radar_d      = get_radar_min_dist()
    lidar_d      = get_lidar_front_dist()
    ds_m_val     = ds_rm.getValue()
    ds_f_val     = ds_rf.getValue()
    ds_r_val     = ds_rr.getValue()
    gyro_vals    = gyro.getValues()
    heading_accum += gyro_vals[2] * timestep / 1000.0

    speed_now = driver.getCurrentSpeed()
    if math.isnan(speed_now) or speed_now < 0:
        speed_now = 0.0

    # Reconocimiento de peatones (costoso, cada DETECT_EVERY frames)
    ped_detected = False
    if frame_count % DETECT_EVERY == 0:
        ped_detected = check_pedestrian()

    # ── Máquina de estados ────────────────────────────────────────────────────

    if state == STATE_CIL:
        # Actualizar contador de detección de peatón
        if ped_detected:
            ped_pos_streak += 1; ped_neg_streak = 0
        else:
            ped_neg_streak += 1; ped_pos_streak = 0

        if ped_pos_streak >= PED_CONFIRM_N:
            state = STATE_PED
            ped_hold_count = PED_HOLD_F
            print("[AUTO] PEATÓN detectado → BRAKE_PED")

        elif lidar_d < OBS_LIDAR_THRESH:
            state = STATE_EVADE
            heading_ref = heading_accum
            print(f"[AUTO] Obstáculo a {lidar_d:.1f}m → EVADE_OBS")

        else:
            # Control de velocidad por radar (distancia al vehículo delante)
            if radar_d < RADAR_STOP_M:
                driver.setCruisingSpeed(0)
                driver.setBrakeIntensity(0.5)
            elif radar_d < RADAR_SAFE_M:
                driver.setCruisingSpeed(CRUISE_SPEED * radar_d / RADAR_SAFE_M)
                driver.setBrakeIntensity(0.0)
            else:
                driver.setCruisingSpeed(CRUISE_SPEED)
                driver.setBrakeIntensity(0.0)

            # CIL predice el steering para el comando activo
            target = cil_predict(bgr, nav_cmd, speed_now)
            current_steer = apply_rate_limit(current_steer, target, STEER_RATE_LIMIT)
            driver.setSteeringAngle(current_steer)

    elif state == STATE_PED:
        # Freno total mientras el peatón está en escena
        driver.setCruisingSpeed(0)
        driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)
        ped_hold_count -= 1
        if not ped_detected:
            ped_neg_streak += 1
        else:
            ped_neg_streak = 0
        if ped_hold_count <= 0:
            if ped_neg_streak >= PED_RELEASE_N:
                state = STATE_CIL
                driver.setBrakeIntensity(0.0)
                ped_pos_streak = ped_neg_streak = 0
                print("[AUTO] Peatón despejado → CIL_DRIVE")
            else:
                ped_hold_count = 20

    elif state == STATE_EVADE:
        # Wall-following: mantenerse a WALL_TARGET metros del lateral derecho
        driver.setCruisingSpeed(SPEED_EVADE)
        driver.setBrakeIntensity(0.0)
        steer = wall_follow_step(ds_m_val)
        current_steer = apply_rate_limit(current_steer, steer, STEER_RATE_LIMIT)
        driver.setSteeringAngle(current_steer)
        # Salir cuando el obstáculo queda atrás y el frente está libre
        if ds_r_val > DS_CLEAR_DIST and ds_f_val < DS_ENGAGE_DIST:
            state = STATE_REOR
            heading_ref = heading_accum
            print("[AUTO] Obstáculo superado → REORIENT")

    elif state == STATE_REOR:
        # Corrección de heading usando gyro para volver al carril
        driver.setCruisingSpeed(SPEED_REORIENT)
        heading_error = heading_ref - heading_accum
        steer_corr    = float(np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE))
        current_steer = apply_rate_limit(current_steer, steer_corr, STEER_RATE_LIMIT)
        driver.setSteeringAngle(current_steer)
        if abs(heading_error) < HEADING_TOL:
            state = STATE_CIL
            heading_accum = 0.0
            driver.setCruisingSpeed(CRUISE_SPEED)
            print("[AUTO] Heading corregido → CIL_DRIVE")

    # ── Actualizar HUD ────────────────────────────────────────────────────────
    draw_hud(state, nav_cmd, current_steer, radar_d, ped_detected, lidar_d)
