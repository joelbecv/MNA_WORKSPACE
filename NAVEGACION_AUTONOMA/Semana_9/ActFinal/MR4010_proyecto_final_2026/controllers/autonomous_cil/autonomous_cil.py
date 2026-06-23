# =============================================================================
# autonomous_cil.py — Controlador autónomo híbrido PID + CIL
# Proyecto Final MR4010 — Equipo 25
# =============================================================================
#
# ARQUITECTURA:
#   Carretera normal (s/w) → Hough + PID (línea amarilla HSV)
#   Intersección izq/der  (a/d) → CIL (rama entrenada para giros)
#   BRAKE_PED → freno de emergencia por peatón
#   EVADE_OBS → wall-following LiDAR
#   REORIENT  → corrección de heading
#
# CONTROLES:
#   s = CONTINUE  (PID normal en carretera)
#   w = STRAIGHT  (PID en cruce — va recto)
#   a = LEFT      (CIL gira izquierda en cruce)
#   d = RIGHT     (CIL gira derecha en cruce)
#   m = forzar PID
#   q = detener
# =============================================================================

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os
import math
import time

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

CRUISE_SPEED   = 25       # km/h
MAX_ANGLE      = 0.5      # rad

CMD_CONTINUE = 0
CMD_STRAIGHT = 1
CMD_LEFT     = 2
CMD_RIGHT    = 3
CMD_LABEL    = {0: "CONTINUE", 1: "RECTO", 2: "IZQUIERDA", 3: "DERECHA"}

# =============================================================================
# PARÁMETROS — PID LANE FOLLOWING (Act 2.1 H2 — línea amarilla HSV)
# =============================================================================

PID_KP           = 0.28
PID_KI           = 0.01
PID_KD           = 0.005
PID_RATE_LIMIT   = 0.06    # rad/frame
PID_MIN_SLOPE    = 0.4     # rechaza líneas casi horizontales
# HSV amarillo (línea central)
YELLOW_LOW  = np.array([15,  80,  80])
YELLOW_HIGH = np.array([35, 255, 255])

# =============================================================================
# PARÁMETROS — CIL GIROS (solo para CMD_LEFT / CMD_RIGHT)
# =============================================================================

CIL_STEER_GAIN = 4.0       # amplifica salida CNN en giros
CIL_RATE_LIMIT = 0.12      # rad/frame más rápido en giros

# =============================================================================
# PARÁMETROS — RADAR
# =============================================================================

RADAR_SAFE_M = 15.0
RADAR_STOP_M =  5.0

# =============================================================================
# PARÁMETROS — PEATÓN
# =============================================================================

PED_CONFIRM_N = 2
PED_RELEASE_N = 4
PED_HOLD_F    = 100
DETECT_EVERY  = 10
PEDESTRIAN_KEYWORDS = ["pedestrian", "Pedestrian", "human", "person"]

# =============================================================================
# PARÁMETROS — LIDAR / EVASIÓN
# =============================================================================

OBS_LIDAR_THRESH = 14.0
LIDAR_FOV_DEG    = 20
SPEED_EVADE      = 15
WALL_TARGET      = 2.9
KP_WALL          = 0.10
DS_CLEAR_DIST    = 4.8
DS_ENGAGE_DIST   = 4.5

# =============================================================================
# PARÁMETROS — REORIENTACIÓN
# =============================================================================

SPEED_REORIENT = 20
KP_HEADING     = 1.0
HEADING_TOL    = 0.08

# =============================================================================
# ESTADOS
# =============================================================================

STATE_CIL   = "CIL_DRIVE"
STATE_PED   = "BRAKE_PED"
STATE_EVADE = "EVADE_OBS"
STATE_REOR  = "REORIENT"

# =============================================================================
# INICIALIZACIÓN
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
# MODELO CIL (PyTorch — solo para giros en intersección)
# =============================================================================

IMG_W, IMG_H = 200, 88
N_COMMANDS   = 4

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CILModel(nn.Module):
    def __init__(self, n_commands=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.2),
        )
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        branch_input = 512 + 128
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(branch_input, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())
            for _ in range(n_commands)
        ])

    def forward(self, img, speed, cmd):
        feat = self.cnn(img)
        spd  = self.speed_fc(speed)
        x    = torch.cat([feat, spd], dim=1)
        outs = torch.stack([b(x) for b in self.branches], dim=1)
        idx  = cmd.view(-1, 1, 1).expand(-1, 1, 1)
        return outs.gather(1, idx).squeeze(1)


CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(
    os.path.join(CTRL_DIR, "..", "..", "models", "cil_model_equipo25.pt"))

cil_model = None
if os.path.exists(MODEL_PATH):
    try:
        ckpt      = torch.load(MODEL_PATH, map_location="cpu")
        cil_model = CILModel(ckpt.get("n_commands", N_COMMANDS))
        cil_model.load_state_dict(ckpt["model_state_dict"])
        cil_model.eval()
        print(f"[CIL] Modelo cargado epoch={ckpt.get('epoch','?')} "
              f"val={ckpt.get('val_loss', ckpt.get('val_mse','?'))}")
    except Exception as e:
        print(f"[CIL] Error cargando modelo: {e}")
else:
    print(f"[CIL] Modelo NO encontrado: {MODEL_PATH}")


def cil_predict(bgr_frame, nav_cmd, speed_kmh):
    if cil_model is None:
        return 0.0
    img_pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    img_pil = img_pil.resize((IMG_W, IMG_H))
    img_t   = NORMALIZE(img_pil).unsqueeze(0)
    spd_t   = torch.tensor([[speed_kmh / 50.0]], dtype=torch.float32)
    cmd_t   = torch.tensor([nav_cmd], dtype=torch.long)
    with torch.no_grad():
        pred = cil_model(img_t, spd_t, cmd_t)
    raw = float(pred[0, 0])
    return float(np.clip(raw * CIL_STEER_GAIN, -1.0, 1.0) * MAX_ANGLE)


# =============================================================================
# PID LANE FOLLOWING (Hough + línea amarilla HSV — H2 de Act 2.1)
# =============================================================================

pid_integral   = 0.0
pid_prev_error = 0.0
pid_no_line    = 0


def pid_lane_step(bgr_frame, dt):
    """Calcula el ángulo de volante con PID siguiendo la línea amarilla."""
    global pid_integral, pid_prev_error, pid_no_line

    small = cv2.resize(bgr_frame, (DW, DH))
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    edges = cv2.Canny(ymask, 50, 150)

    h, w = edges.shape
    verts = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.6)),
        (int(w * 0.65), int(h * 0.6)),
        (int(w * 0.90), h),
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, verts, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20,
                            minLineLength=20, maxLineGap=15)

    filtered = []
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            if x2 == x1:
                continue
            if abs((y2 - y1) / (x2 - x1)) >= PID_MIN_SLOPE:
                filtered.append(ln)

    lane_x = None
    if filtered:
        left_xs, right_xs = [], []
        for ln in filtered:
            x1, y1, x2, y2 = ln[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_xs.extend([x1, x2])
            else:
                right_xs.extend([x1, x2])
        if left_xs and right_xs:
            lane_x = (np.mean(left_xs) + np.mean(right_xs)) / 2.0
        elif left_xs:
            lane_x = np.mean(left_xs)
        elif right_xs:
            lane_x = np.mean(right_xs)

    if lane_x is not None:
        pid_no_line = 0
        error        = (lane_x - w / 2.0) / (w / 2.0)
        pid_integral = float(np.clip(pid_integral + error * dt, -0.5, 0.5))
        deriv        = (error - pid_prev_error) / dt
        raw          = PID_KP * error + PID_KI * pid_integral + PID_KD * deriv
        pid_prev_error = error
        return float(np.clip(raw, -MAX_ANGLE, MAX_ANGLE)), True
    else:
        pid_no_line  += 1
        pid_integral *= 0.6
        pid_prev_error = 0.0
        return None, False


# =============================================================================
# SENSORES / CONTROL AUXILIAR
# =============================================================================

def get_lidar_front_dist():
    ranges = lidar.getRangeImage()
    if not ranges:
        return 999.0
    c    = LIDAR_RAYS // 2
    half = int(LIDAR_FOV_DEG * LIDAR_RAYS / 180)
    cone = ranges[max(0, c - half): c + half]
    valid = [r for r in cone if 0.1 < r < 80.0]
    return min(valid) if valid else 999.0


def get_radar_min_dist():
    tgts = radar.getTargets()
    return min((t.distance for t in tgts), default=999.0)


def check_pedestrian():
    for obj in camera.getRecognitionObjects():
        name = obj.getModel() if hasattr(obj, "getModel") else ""
        if any(kw in name for kw in PEDESTRIAN_KEYWORDS):
            return True
    return False


def wall_follow_step(ds_m):
    return float(np.clip(KP_WALL * (WALL_TARGET - ds_m), -MAX_ANGLE, MAX_ANGLE))


def apply_rate_limit(current, target, limit):
    return current + float(np.clip(target - current, -limit, limit))


def draw_hud(state, nav_cmd, steer, radar_d, ped, lidar_d, mode):
    bg = {STATE_CIL: 0x002200, STATE_PED: 0x440000,
          STATE_EVADE: 0x002244, STATE_REOR: 0x222200}
    display.setColor(bg.get(state, 0x000000))
    display.fillRectangle(0, 0, DW, DH)
    display.setColor(0xFFFFFF)
    display.drawText(f"{state}", 2, 2)
    display.drawText(f"NAV: {CMD_LABEL[nav_cmd]}", 2, 14)
    display.drawText(f"MODE:{mode}", 2, 26)
    display.drawText(f"St:{steer:+.3f}r", 2, 38)
    display.drawText(f"Radar:{radar_d:.1f}m", 2, 50)
    display.drawText(f"LiDAR:{lidar_d:.1f}m", 2, 62)
    display.setColor(0xFF4444 if ped else 0x44FF44)
    display.drawText(f"Ped:{'DET' if ped else 'OK'}", 2, 74)


# =============================================================================
# ESTADO INICIAL
# =============================================================================

state         = STATE_CIL
nav_cmd       = CMD_CONTINUE
current_steer = 0.0
mode          = "PID"

ped_pos_streak = 0
ped_neg_streak = 0
ped_hold_count = 0
heading_accum  = 0.0
heading_ref    = 0.0
frame_count    = 0

print("=" * 60)
print("[AUTO] Controlador híbrido PID+CIL — Equipo 25")
print(f"[AUTO] CIL: {'CARGADO' if cil_model else 'NO CARGADO'}")
print(f"[AUTO] Radar seguro={RADAR_SAFE_M}m  stop={RADAR_STOP_M}m")
print("  s=CONTINUE(PID)  w=RECTO(PID)  a=IZQ(CIL)  d=DER(CIL)  q=stop")
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
        if   key in (ord('S'), ord('s')):
            nav_cmd = CMD_CONTINUE; print("[NAV] CONTINUE (PID)")
        elif key in (ord('W'), ord('w')):
            nav_cmd = CMD_STRAIGHT; print("[NAV] RECTO (PID)")
        elif key in (ord('A'), ord('a')):
            nav_cmd = CMD_LEFT;     print("[NAV] IZQUIERDA (CIL)")
        elif key in (ord('D'), ord('d')):
            nav_cmd = CMD_RIGHT;    print("[NAV] DERECHA (CIL)")
        elif key in (ord('M'), ord('m')):
            state = STATE_CIL;      print("[DBG] Forzado CIL_DRIVE")
        elif key in (ord('Q'), ord('q')):
            driver.setCruisingSpeed(0); driver.setBrakeIntensity(1.0)
        key = keyboard.getKey()

    # ── Imagen ────────────────────────────────────────────────────────────────
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # ── Sensores ──────────────────────────────────────────────────────────────
    radar_d  = get_radar_min_dist()
    lidar_d  = get_lidar_front_dist()
    ds_m_val = ds_rm.getValue()
    ds_f_val = ds_rf.getValue()
    ds_r_val = ds_rr.getValue()

    gyro_vals     = gyro.getValues()
    heading_accum += gyro_vals[2] * timestep / 1000.0

    speed_now = driver.getCurrentSpeed()
    if math.isnan(speed_now) or speed_now < 0:
        speed_now = 0.0

    ped_detected = False
    if frame_count % DETECT_EVERY == 0:
        ped_detected = check_pedestrian()

    # ── Máquina de estados ────────────────────────────────────────────────────

    if state == STATE_CIL:
        # Detección de peatón
        if ped_detected:
            ped_pos_streak += 1; ped_neg_streak = 0
        else:
            ped_neg_streak += 1; ped_pos_streak = 0

        if ped_pos_streak >= PED_CONFIRM_N:
            state = STATE_PED; ped_hold_count = PED_HOLD_F
            print("[AUTO] PEATÓN → BRAKE_PED")

        # Detección de obstáculo
        elif lidar_d < OBS_LIDAR_THRESH:
            state = STATE_EVADE; heading_ref = heading_accum
            print(f"[AUTO] Obstáculo {lidar_d:.1f}m → EVADE_OBS")

        else:
            # ── Control de velocidad por radar ───────────────────────────────
            if radar_d < RADAR_STOP_M:
                driver.setCruisingSpeed(0); driver.setBrakeIntensity(0.5)
            elif radar_d < RADAR_SAFE_M:
                driver.setCruisingSpeed(CRUISE_SPEED * radar_d / RADAR_SAFE_M)
                driver.setBrakeIntensity(0.0)
            else:
                driver.setCruisingSpeed(CRUISE_SPEED); driver.setBrakeIntensity(0.0)

            # ── Dirección: PID o CIL según comando ───────────────────────────
            dt_sim = timestep / 1000.0   # dt fijo de simulación

            if nav_cmd in (CMD_LEFT, CMD_RIGHT):
                target = cil_predict(bgr, nav_cmd, speed_now)
                current_steer = apply_rate_limit(current_steer, target, CIL_RATE_LIMIT)
                mode = "CIL"
            else:
                pid_raw, detected = pid_lane_step(bgr, dt_sim)
                if detected:
                    current_steer = apply_rate_limit(current_steer, pid_raw, PID_RATE_LIMIT)
                else:
                    if pid_no_line > 10:
                        current_steer *= 0.95
                mode = f"PID{'!' if not detected else ''}"

            driver.setSteeringAngle(current_steer)

    elif state == STATE_PED:
        driver.setCruisingSpeed(0); driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)
        ped_hold_count -= 1
        if not ped_detected:
            ped_neg_streak += 1
        else:
            ped_neg_streak = 0
        if ped_hold_count <= 0:
            if ped_neg_streak >= PED_RELEASE_N:
                state = STATE_CIL; driver.setBrakeIntensity(0.0)
                ped_pos_streak = ped_neg_streak = 0
                print("[AUTO] Peatón despejado → CIL_DRIVE")
            else:
                ped_hold_count = 20
        mode = "BRAKE"

    elif state == STATE_EVADE:
        driver.setCruisingSpeed(SPEED_EVADE); driver.setBrakeIntensity(0.0)
        steer = wall_follow_step(ds_m_val)
        current_steer = apply_rate_limit(current_steer, steer, PID_RATE_LIMIT)
        driver.setSteeringAngle(current_steer)
        if ds_r_val > DS_CLEAR_DIST and ds_f_val < DS_ENGAGE_DIST:
            state = STATE_REOR; heading_ref = heading_accum
            print("[AUTO] Obstáculo superado → REORIENT")
        mode = "EVADE"

    elif state == STATE_REOR:
        driver.setCruisingSpeed(SPEED_REORIENT)
        heading_error = heading_ref - heading_accum
        steer_corr    = float(np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE))
        current_steer = apply_rate_limit(current_steer, steer_corr, PID_RATE_LIMIT)
        driver.setSteeringAngle(current_steer)
        if abs(heading_error) < HEADING_TOL:
            state = STATE_CIL; heading_accum = 0.0
            driver.setCruisingSpeed(CRUISE_SPEED)
            print("[AUTO] Heading OK → CIL_DRIVE")
        mode = "REOR"

    # ── HUD ───────────────────────────────────────────────────────────────────
    draw_hud(state, nav_cmd, current_steer, radar_d, ped_detected, lidar_d, mode)
