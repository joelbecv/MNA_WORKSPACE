# =============================================================================
# autonomous_cil.py — Controlador autónomo CIL + módulos de seguridad
# Proyecto Final MR4010 — Equipo 25
# =============================================================================
#
# ARQUITECTURA (máquina de estados):
#   CIL_DRIVE  — Inferencia normal: imagen + nav_cmd → CNN → steering
#   BRAKE_PED  — Freno de emergencia por peatón detectado (cámara recognition)
#   EVADE_OBS  — Wall-following derecha (evasión de vehículo/obstáculo)
#   REORIENT   — Recuperación de heading con giroscopio post-evasión
#
# SENSORES (todos en city_traffic_2025_02.wbt):
#   camera     — 320×160, FOV=1 rad  → input CIL + recognition de peatones
#   radar      — range 1–50 m        → distancia al vehículo más próximo
#   Sick LMS 291 — 180° horizontal   → detección de obstáculos (evasión)
#   ds_right_*   — 3 sensores lat.   → wall-following durante evasión
#   Gyro         — eje Z             → integración de heading
#
# CONTROLES (teclado):
#   A  — fijar comando IZQUIERDA en siguiente intersección
#   W  — fijar comando RECTO
#   D  — fijar comando DERECHA
#   M  — forzar estado CIL_DRIVE (debug)
#
# NOTAS CRÍTICAS:
#   - timestep = int() SIN multiplicador (freeze)
#   - engineSound "" aplicado en .wbt (fix principal del freeze macOS)
#   - lidar.enable() sin enablePointCloud() (enablePointCloud → freeze)
#   - recognition: camera.recognitionEnable(ts) — NO enableRecognition (R2023b)
#   - Radar: getTargets() → lista de RadarTarget con .distance y .speed
#   - Modelo: exportar desde Colab como cil_model_equipo25.h5 en models/
#
# PROTOCOLO PREVIO AL RUN:
#   1. python -m py_compile autonomous_cil.py && echo OK
#   2. Copiar cil_model_equipo25.h5 a ../../models/
#   3. Abrir city_traffic_2025_02.wbt → Play → verificar consola sin errores
#   4. Primera corrida: solo prints de sensores, sin control (MODE_SENSOR_ONLY)
# =============================================================================

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os

# Importar TensorFlow/Keras para el modelo CIL
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[CIL] ⚠️  TensorFlow no disponible — instalar con: pip install tensorflow")

# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

CRUISE_SPEED  = 30       # km/h durante conducción autónoma normal
MAX_ANGLE     = 0.5      # rad — límite del volante
MAX_STEER_RATE = 0.03    # rad/frame — rate limiter (suaviza el volante)

# Comandos de navegación CIL
CMD_STRAIGHT = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_LABEL    = {0: "RECTO", 1: "IZQUIERDA", 2: "DERECHA"}

# =============================================================================
# PARÁMETROS — RADAR (distancia a vehículo frontal)
# =============================================================================

RADAR_SAFE_M  = 15.0     # m: distancia de seguridad — se anuncia en el video
RADAR_STOP_M  = 5.0      # m: detención completa
# Velocidad proporcional a distancia: v = CRUISE × min(1, dist/SAFE_M)
# Negocio: evita colisiones traseras en tráfico denso (SUMO)

# =============================================================================
# PARÁMETROS — RECOGNITION / FRENO POR PEATÓN (reutilizado de Act 3.1)
# =============================================================================

PED_CONFIRM_N  = 2        # scans positivos consecutivos para confirmar peatón
PED_RELEASE_N  = 4        # scans negativos para liberar el freno
PED_HOLD_F     = 100      # frames de freno garantizados (~1.6 s a 16 ms/frame)
DETECT_EVERY   = 10       # ejecutar recognition cada N frames (no cada frame)

# Modelos de peatones en Webots (reconocidos por recognition por nombre de modelo)
PEDESTRIAN_KEYWORDS = ["pedestrian", "Pedestrian", "human", "person"]

# =============================================================================
# PARÁMETROS — LIDAR / EVASIÓN (reutilizados de Act 4.2)
# =============================================================================

OBS_LIDAR_THRESH = 14.0  # m: iniciar evasión con margen suficiente
LIDAR_FOV_DEG    = 20    # grados a cada lado del centro para medir frente

SPEED_EVADE   = 15       # km/h durante la evasión
WALL_TARGET   = 2.9      # m: distancia objetivo al obstáculo lateral
KP_WALL       = 0.10     # ganancia P del controlador de pared
DS_CLEAR_DIST = 4.8      # m: sensor trasero libre → obstáculo superado
DS_ENGAGE_DIST = 4.5     # m: sensor frontal detecta obstáculo lateralmente

# =============================================================================
# PARÁMETROS — REORIENTACIÓN (reutilizados de Act 4.2)
# =============================================================================

SPEED_REORIENT = 20      # km/h durante corrección de heading
KP_HEADING     = 1.0     # ganancia P
HEADING_TOL    = 0.08    # rad: tolerancia para declarar heading recuperado

# =============================================================================
# ESTADOS DE LA MÁQUINA
# =============================================================================

STATE_CIL   = "CIL_DRIVE"
STATE_PED   = "BRAKE_PED"
STATE_EVADE = "EVADE_OBS"
STATE_REOR  = "REORIENT"

# =============================================================================
# INICIALIZACIÓN DE DRIVER Y DISPOSITIVOS
# =============================================================================

driver   = Driver()
timestep = int(driver.getBasicTimeStep())   # 16 ms — NUNCA multiplicar

# ── Cámara ──────────────────────────────────────────────────────────────────
camera = driver.getDevice("camera")
camera.enable(timestep)
# Recognition para detección de peatones (API R2023b: recognitionEnable, NO enableRecognition)
camera.recognitionEnable(timestep * DETECT_EVERY)
CAM_W = camera.getWidth()    # 320
CAM_H = camera.getHeight()   # 160

# ── Radar ────────────────────────────────────────────────────────────────────
radar = driver.getDevice("radar")
radar.enable(timestep)

# ── LiDAR (Sick LMS 291) ────────────────────────────────────────────────────
# IMPORTANTE: solo enable() — enablePointCloud() causa freeze
lidar = driver.getDevice("Sick LMS 291")
lidar.enable(timestep)
# NO llamar lidar.enablePointCloud()
LIDAR_RAYS = lidar.getNumberOfPoints()   # 180 rayos (1°/rayo)

# ── Sensores de distancia laterales (wall-following) ─────────────────────────
ds_rf = driver.getDevice("ds_right_front")
ds_rm = driver.getDevice("ds_right_mid")
ds_rr = driver.getDevice("ds_right_rear")
for ds in [ds_rf, ds_rm, ds_rr]:
    ds.enable(timestep)

# ── Giroscopio (integración de heading) ──────────────────────────────────────
gyro = driver.getDevice("gyro")
gyro.enable(timestep)

# ── Display ──────────────────────────────────────────────────────────────────
display = driver.getDevice("display_image")   # 200×150

# ── Teclado ──────────────────────────────────────────────────────────────────
keyboard = Keyboard()
keyboard.enable(timestep)

# =============================================================================
# CARGA DEL MODELO CIL
# =============================================================================

CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "models", "cil_model_equipo25.h5"))

model = None
if TF_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[CIL] ✓ Modelo cargado: {MODEL_PATH}")
        model.summary()
    else:
        print(f"[CIL] ⚠️  Modelo NO encontrado en: {MODEL_PATH}")
        print("[CIL]    Copiar cil_model_equipo25.h5 exportado de Colab a models/")


def preprocess_image(bgr: np.ndarray, nav_cmd: int) -> np.ndarray:
    """
    Preprocesa la imagen de cámara para la inferencia CIL.
    Entrada:  bgr (CAM_H × CAM_W × 3)
    Salida:   array (1, H_MODEL, W_MODEL, 3+1) o según arquitectura del modelo

    AJUSTAR según la arquitectura definida en el notebook de Colab:
      - Si el modelo acepta imagen + comando como un solo tensor, concatenar aquí
      - Si el modelo tiene dos entradas separadas, retornar una tupla
    """
    # Resize al tamaño de entrada del modelo (Codevilla: 88×200, Bojarski: 66×200)
    # Ajustar si se usó otra resolución en el entrenamiento
    img_model = cv2.resize(bgr, (200, 88))
    img_norm  = img_model.astype(np.float32) / 255.0   # normalizar [0, 1]
    img_batch = np.expand_dims(img_norm, axis=0)        # (1, 88, 200, 3)

    # Comando de navegación como one-hot (3 clases)
    cmd_onehot = np.zeros((1, 3), dtype=np.float32)
    cmd_onehot[0, nav_cmd] = 1.0

    # NOTA: adaptar según la firma del modelo:
    #   Modelo 1 entrada (imagen con cmd como canal adicional):
    #     return img_batch, cmd_onehot   ← devolver por separado y manejar en el loop
    #   Modelo 2 entradas (imagen, cmd):
    #     return [img_batch, cmd_onehot]
    return img_batch, cmd_onehot


def cil_predict(bgr: np.ndarray, nav_cmd: int) -> float:
    """
    Ejecuta la inferencia CIL y retorna el ángulo de dirección normalizado.
    Retorna 0.0 si el modelo no está cargado (carro va recto).
    """
    if model is None:
        return 0.0

    img_inp, cmd_inp = preprocess_image(bgr, nav_cmd)

    # AJUSTAR según arquitectura del modelo Colab:
    try:
        # Intento con dos entradas (más común en CIL)
        pred = model.predict([img_inp, cmd_inp], verbose=0)
    except Exception:
        # Fallback: una sola entrada de imagen
        pred = model.predict(img_inp, verbose=0)

    # La salida del modelo es steering normalizado en [-1, 1]
    # Desnormalizar: steering_rad = pred * MAX_ANGLE
    steering_norm = float(pred[0][0])
    return np.clip(steering_norm, -1.0, 1.0) * MAX_ANGLE


# =============================================================================
# FUNCIONES DE SENSORES
# =============================================================================

def get_lidar_front_dist() -> float:
    """Distancia mínima en el cono frontal del LiDAR (±LIDAR_FOV_DEG°)."""
    ranges = lidar.getRangeImage()
    if not ranges:
        return 999.0
    center = LIDAR_RAYS // 2
    half   = int(LIDAR_FOV_DEG * LIDAR_RAYS / 180)
    cone   = ranges[center - half: center + half]
    valid  = [r for r in cone if r > 0.1 and r < 80.0]
    return min(valid) if valid else 999.0


def get_radar_min_dist() -> float:
    """Distancia al objetivo radar más próximo."""
    targets = radar.getTargets()
    if not targets:
        return 999.0
    return min(t.distance for t in targets)


def check_pedestrian() -> bool:
    """
    Detecta peatones vía recognition de cámara.
    Reutilizado de Act 4.2 — API R2023b.
    """
    objs = camera.getRecognitionObjects()
    for obj in objs:
        model_name = obj.getModel() if hasattr(obj, 'getModel') else ""
        if any(kw in model_name for kw in PEDESTRIAN_KEYWORDS):
            return True
    return False


# =============================================================================
# FUNCIONES DE CONTROL
# =============================================================================

def wall_follow_step(ds_f: float, ds_m: float, ds_r: float) -> float:
    """
    Controlador P de seguimiento de pared derecha (copiado de Act 4.2).
    Mantiene distancia WALL_TARGET al obstáculo lateral derecho.
    """
    error  = WALL_TARGET - ds_m
    steer  = KP_WALL * error
    return float(np.clip(steer, -MAX_ANGLE, MAX_ANGLE))


def apply_rate_limit(current: float, target: float) -> float:
    """Limita el cambio de steering a MAX_STEER_RATE rad/frame."""
    delta = target - current
    delta = np.clip(delta, -MAX_STEER_RATE, MAX_STEER_RATE)
    return current + delta


def draw_hud(state: str, nav_cmd: int, steer: float,
             radar_d: float, ped: bool, lidar_d: float) -> None:
    """Dibuja información de estado en el display 200×150."""
    # Fondo según estado
    colors = {
        STATE_CIL:   0x002200,
        STATE_PED:   0x440000,
        STATE_EVADE: 0x002244,
        STATE_REOR:  0x222200,
    }
    display.setColor(colors.get(state, 0x000000))
    display.fillRectangle(0, 0, 200, 150)

    display.setColor(0xFFFFFF)
    display.drawText(f"STATE: {state}", 2, 2)
    display.drawText(f"NAV:   {CMD_LABEL[nav_cmd]}", 2, 14)
    display.drawText(f"Steer: {steer:+.3f}r", 2, 26)
    display.drawText(f"Radar: {radar_d:.1f}m", 2, 38)
    display.drawText(f"LiDAR: {lidar_d:.1f}m", 2, 50)

    color_ped = 0xFF4444 if ped else 0x44FF44
    display.setColor(color_ped)
    display.drawText(f"Ped:   {'DETECTADO' if ped else 'libre'}", 2, 62)


# =============================================================================
# ESTADO INICIAL
# =============================================================================

state         = STATE_CIL
nav_cmd       = CMD_STRAIGHT
current_steer = 0.0

# Variables de estado — freno por peatón
ped_pos_streak = 0
ped_neg_streak = 0
ped_hold_count = 0

# Variables de estado — evasión
heading_ref    = 0.0
heading_accum  = 0.0
prev_time      = 0.0

frame_count    = 0

print("=" * 60)
print("[CIL-AUTO] Controlador autónomo iniciado")
print(f"[CIL-AUTO] Modelo: {'cargado ✓' if model else 'NO cargado ⚠️'}")
print(f"[CIL-AUTO] Distancia radar segura: {RADAR_SAFE_M} m")
print("  CONTROLES: A=izquierda  W=recto  D=derecha  M=forzar CIL")
print("=" * 60)

driver.setCruisingSpeed(CRUISE_SPEED)

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

while driver.step() != -1:
    frame_count += 1

    # ── TECLADO — comandos de navegación ─────────────────────────────────────
    key = keyboard.getKey()
    while key > 0:
        if key == ord('A') or key == ord('a'):
            nav_cmd = CMD_LEFT
            print(f"[NAV] IZQUIERDA")
        elif key == ord('W') or key == ord('w'):
            nav_cmd = CMD_STRAIGHT
            print(f"[NAV] RECTO")
        elif key == ord('D') or key == ord('d'):
            nav_cmd = CMD_RIGHT
            print(f"[NAV] DERECHA")
        elif key == ord('M') or key == ord('m'):
            state = STATE_CIL
            print(f"[DEBUG] Forzado a CIL_DRIVE")
        key = keyboard.getKey()

    # ── LECTURA DE SENSORES ───────────────────────────────────────────────────
    raw     = camera.getImage()
    img     = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr     = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    radar_d = get_radar_min_dist()
    lidar_d = get_lidar_front_dist()

    ds_f_val = ds_rf.getValue()
    ds_m_val = ds_rm.getValue()
    ds_r_val = ds_rr.getValue()

    gyro_vals    = gyro.getValues()
    yaw_rate     = gyro_vals[2]    # eje Z en frame del vehículo (R2023b)
    heading_accum += yaw_rate * timestep / 1000.0   # integrar en radianes

    # Detection de peatón (solo en frames múltiplos de DETECT_EVERY)
    ped_detected = False
    if frame_count % DETECT_EVERY == 0:
        ped_detected = check_pedestrian()

    # ── MÁQUINA DE ESTADOS ────────────────────────────────────────────────────

    if state == STATE_CIL:
        # ── Verificar peatón ──────────────────────────────────────────────────
        if ped_detected:
            ped_pos_streak += 1
            ped_neg_streak  = 0
        else:
            ped_neg_streak  += 1
            ped_pos_streak   = 0

        if ped_pos_streak >= PED_CONFIRM_N:
            state          = STATE_PED
            ped_hold_count = PED_HOLD_F
            print(f"[CIL] PEATÓN detectado → BRAKE_PED")

        # ── Verificar obstáculo con LiDAR ─────────────────────────────────────
        elif lidar_d < OBS_LIDAR_THRESH:
            state       = STATE_EVADE
            heading_ref = heading_accum   # guardar heading actual como referencia
            print(f"[CIL] Obstáculo a {lidar_d:.1f}m → EVADE_OBS")

        else:
            # ── Inferencia CIL ────────────────────────────────────────────────
            target_steer = cil_predict(bgr, nav_cmd)

            # Control de velocidad por radar (nueva funcionalidad Act Final)
            if radar_d < RADAR_STOP_M:
                driver.setCruisingSpeed(0)
                driver.setBrakeIntensity(0.5)
            elif radar_d < RADAR_SAFE_M:
                speed_factor = radar_d / RADAR_SAFE_M
                driver.setCruisingSpeed(CRUISE_SPEED * speed_factor)
                driver.setBrakeIntensity(0.0)
            else:
                driver.setCruisingSpeed(CRUISE_SPEED)
                driver.setBrakeIntensity(0.0)

            # Rate limiter en steering (suaviza cambios abruptos de la CNN)
            current_steer = apply_rate_limit(current_steer, target_steer)
            driver.setSteeringAngle(current_steer)

    elif state == STATE_PED:
        # Freno de emergencia — mantener N frames
        driver.setCruisingSpeed(0)
        driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)

        ped_hold_count -= 1
        if ped_hold_count <= 0:
            if ped_neg_streak >= PED_RELEASE_N:
                state = STATE_CIL
                driver.setBrakeIntensity(0.0)
                ped_pos_streak = 0
                ped_neg_streak = 0
                print("[CIL] Peatón liberado → CIL_DRIVE")
            else:
                ped_hold_count = 20   # extender 20 frames más si sigue detectando

    elif state == STATE_EVADE:
        # Wall-following lateral derecha (Act 4.2 logic)
        driver.setCruisingSpeed(SPEED_EVADE)
        driver.setBrakeIntensity(0.0)

        steer = wall_follow_step(ds_f_val, ds_m_val, ds_r_val)
        current_steer = apply_rate_limit(current_steer, steer)
        driver.setSteeringAngle(current_steer)

        # Obstáculo superado: sensor trasero libre
        if ds_r_val > DS_CLEAR_DIST and ds_f_val < DS_ENGAGE_DIST:
            state = STATE_REOR
            heading_ref = heading_accum   # re-capturar referencia para corrección
            print(f"[EVADE] Obstáculo superado → REORIENT")

    elif state == STATE_REOR:
        # Recuperar heading original con giroscopio
        driver.setCruisingSpeed(SPEED_REORIENT)
        heading_error = heading_ref - heading_accum
        steer_corr    = KP_HEADING * heading_error
        steer_corr    = np.clip(steer_corr, -MAX_ANGLE, MAX_ANGLE)

        current_steer = apply_rate_limit(current_steer, steer_corr)
        driver.setSteeringAngle(current_steer)

        if abs(heading_error) < HEADING_TOL:
            state         = STATE_CIL
            heading_accum = 0.0   # resetear acumulador
            driver.setCruisingSpeed(CRUISE_SPEED)
            print(f"[REOR] Heading recuperado → CIL_DRIVE")

    # ── HUD ───────────────────────────────────────────────────────────────────
    draw_hud(state, nav_cmd, current_steer, radar_d, ped_detected, lidar_d)
