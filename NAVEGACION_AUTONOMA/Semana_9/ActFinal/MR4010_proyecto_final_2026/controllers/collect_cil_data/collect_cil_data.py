# =============================================================================
# collect_cil_data.py — Recolección de datos CIL (Codevilla 2017)
# Proyecto Final MR4010 — Equipo 25
# =============================================================================
#
# CONTROLES (ventana 3D de Webots):
#   ← / →  : volante izquierda / derecha (mantener presionado)
#   i / k   : velocidad +5 / -5 km/h
#   s       : CMD_CONTINUE — carretera normal (DEFAULT)
#   w       : CMD_STRAIGHT — recto en intersección
#   a       : CMD_LEFT     — izquierda en intersección
#   d       : CMD_RIGHT    — derecha en intersección
#   m       : marcar cruce en GPS
#   q       : salir
# =============================================================================

from controller import Keyboard
from vehicle import Car, Driver
import numpy as np, cv2, os, csv, json, math

# =============================================================================
# PARÁMETROS
# =============================================================================

CRUISE_SPEED  = 40
SPEED_MIN, SPEED_MAX, SPEED_STEP = 10, 50, 5
MAX_ANGLE     = 0.5
STEER_STEP    = 0.015
CENTER_DECAY  = 0.85
CAPTURE_EVERY      = 5
CAPTURE_EVERY_CONT = 30
INTER_RADIUS  = 80

# Parámetros del asistente de carril automático (tecla F)
LANE_KP       = 0.40    # ganancia: error de posición → steering
LANE_EMA      = 0.65    # suavizado exponencial del steering
LANE_RATE_MAX = 0.06    # rad/frame máximo cambio
LANE_ROI_TOP  = 0.45    # ignorar el % superior de la imagen
MIN_VERT_PX   = 8       # mínimo píxeles verticales para aceptar línea
YELLOW_TARGET = 0.22    # fracción del ancho donde debe quedar la línea amarilla
                        # (~22% desde izquierda = carril derecho correcto)

CMD_CONTINUE, CMD_STRAIGHT, CMD_LEFT, CMD_RIGHT = 0, 1, 2, 3
CMD_FULL   = {0:"CONTINUE", 1:"RECTO", 2:"IZQUIERDA", 3:"DERECHA"}
CMD_TARGET = {0:500, 1:150, 2:175, 3:175}
CMD_COLOR  = {0:0x003300, 1:0x003366, 2:0x553300, 3:0x440044}
CMD_TCOL   = {0:0x00FF44, 1:0x44AAFF, 2:0xFFAA00, 3:0xFF66FF}

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

# =============================================================================
# DATOS
# =============================================================================

CTRL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "images"))
CSV_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "dataset.csv"))
os.makedirs(DATA_DIR, exist_ok=True)

csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
csv_file   = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow(["image_path", "steering_angle", "speed_kmh", "nav_command"])
    csv_file.flush()

INTER_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "intersections.json"))
intersections = []
if os.path.exists(INTER_PATH):
    with open(INTER_PATH) as f:
        intersections = json.load(f)
    print(f"[CIL] {len(intersections)} intersecciones cargadas")
else:
    print("[CIL] Sin intersecciones — presiona M al pasar por cada cruce")

def save_intersections():
    with open(INTER_PATH, "w") as f:
        json.dump(intersections, f, indent=2)

def dist_to_nearest(pos):
    if not intersections or pos is None:
        return None
    gx, gy = pos[0], pos[1]
    return min(math.sqrt((gx - ix)**2 + (gy - iy)**2) for ix, iy in intersections)

existing   = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
img_count  = len(existing)
cmd_counts = {0:0, 1:0, 2:0, 3:0}
if csv_exists:
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            try: cmd_counts[int(row["nav_command"])] += 1
            except: pass

print("=" * 60)
print("[CIL] Controlador iniciado — controles en ventana 3D Webots")
print(f"[CIL] Imágenes previas: {img_count}")
print("  f = AUTO/MANUAL  |  ←/→ volante (MANUAL)  |  i/k velocidad")
print("  s/w/a/d nav      |  m marcar cruce         |  q salir")
print("=" * 60)

# =============================================================================
# ASISTENTE DE CARRIL (modo AUTO)
# =============================================================================

def lane_follow_steer(bgr):
    """
    Calcula steering para mantenerse en el carril derecho.
    Rastrea la línea amarilla central al 22% del ancho (YELLOW_TARGET).
    Fallback a centrado general si no hay amarillo.
    Devuelve rad en [-MAX_ANGLE, MAX_ANGLE] o None si no detecta líneas.
    """
    h = bgr.shape[0]
    roi   = bgr[int(h * LANE_ROI_TOP):, :]
    roi_w = roi.shape[1]

    hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ymask   = cv2.inRange(hsv, np.array([18, 70, 70]), np.array([38, 255, 255]))
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.bitwise_or(cv2.Canny(blur, 40, 120), cv2.Canny(ymask, 50, 150))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=25, minLineLength=15, maxLineGap=12)
    if lines is None:
        return None

    yellow_xs, all_xs = [], []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        if abs(y2 - y1) < MIN_VERT_PX:
            continue
        cx = (x1 + x2) / 2.0
        all_xs.append(cx)
        mx, my = int(cx), int((y1 + y2) / 2)
        if 0 <= my < ymask.shape[0] and 0 <= mx < ymask.shape[1]:
            if ymask[my, mx] > 0:
                yellow_xs.append(cx)

    if not all_xs:
        return None

    if yellow_xs:
        # Amarillo detectado: mantenerlo en YELLOW_TARGET del ancho → carril derecho
        ycx   = sum(yellow_xs) / len(yellow_xs)
        error = (ycx - YELLOW_TARGET * roi_w) / (roi_w / 2.0)
    else:
        # Sin amarillo: centrar en todas las líneas (curva o zona sin pintura)
        error = (sum(all_xs) / len(all_xs) - roi_w / 2.0) / (roi_w / 2.0)

    return float(np.clip(LANE_KP * error, -MAX_ANGLE, MAX_ANGLE))

# =============================================================================
# HUD
# =============================================================================

def draw_hud(cmd, steer, speed, total, counts, dist_m, auto_mode=False):
    n = max(1, sum(counts.values()))
    display.setColor(CMD_COLOR[cmd])
    display.fillRectangle(0, 0, DW, DH)
    display.setColor(0x00CCFF if auto_mode else 0xFFCC00)
    display.drawText("AUTO" if auto_mode else "MAN", 2, 2)
    display.setColor(CMD_TCOL[cmd])
    display.drawText(f"CMD:{CMD_FULL[cmd]}", 36, 2)
    display.setColor(0xFFFFFF)
    display.drawText(f"St:{steer:+.3f} {speed:.0f}km/h", 2, 14)
    display.drawText(f"TOTAL:{total:,}", 2, 26)

    if dist_m is None:
        display.setColor(0x888888)
        display.drawText("CRUCE: sin datos", 2, 38)
    elif dist_m < 15:
        display.setColor(0xFF0000)
        display.drawText(f"[EN CRUCE] {dist_m:.0f}m  W/A/D", 2, 38)
    elif dist_m < INTER_RADIUS:
        display.setColor(0xFF8800)
        display.drawText(f"CRUCE: {dist_m:.0f}m  PREPARA", 2, 38)
    elif dist_m < 120:
        display.setColor(0xFFFF00)
        display.drawText(f"CRUCE: {dist_m:.0f}m  pronto", 2, 38)
    else:
        display.setColor(0x00FF44)
        display.drawText(f"CRUCE: {dist_m:.0f}m", 2, 38)

    if dist_m is not None:
        prox = max(0.0, min(1.0, 1.0 - dist_m / 120.0))
        bw   = max(1, int(prox * (DW - 4)))
        col  = 0x00FF44 if prox < 0.5 else (0xFF8800 if prox < 0.85 else 0xFF0000)
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
        bw = max(1, int(min(1.0, ct / CMD_TARGET[c]) * (DW - 4)))
        rem = DW - 4 - bw
        display.setColor(CMD_TCOL[c] if c == cmd else 0xAAAAAA)
        display.drawText(f"{lb}:{ct:4d}({ct/n*100:.0f}%)", 2, y)
        display.setColor(CMD_TCOL[c] if c == cmd else 0x444444)
        display.fillRectangle(2, y + 11, bw, 4)
        if rem > 0:
            display.setColor(0x222222)
            display.fillRectangle(2 + bw, y + 11, rem, 4)

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

steer        = 0.0
nav_cmd      = CMD_CONTINUE
cruise_speed = CRUISE_SPEED
frame        = 0
gps_pos      = None
gps_printed  = False
auto_mode    = False
auto_steer   = 0.0
driver.setCruisingSpeed(cruise_speed)

while robot.step() != -1:
    frame += 1

    gps_vals = gps.getValues()
    if gps_vals and not any(math.isnan(v) for v in gps_vals):
        gps_pos = gps_vals
        if not gps_printed:
            d = dist_to_nearest(gps_pos)
            print(f"[GPS] x={gps_pos[0]:.1f} y={gps_pos[1]:.1f} z={gps_pos[2]:.1f}")
            print(f"[GPS] Cruce más cercano: {d:.1f}m" if d else "[GPS] Sin intersecciones")
            gps_printed = True

    dist_m = dist_to_nearest(gps_pos)

    steer_pressed = False
    key = keyboard.getKey()
    while key > 0:
        if key == Keyboard.LEFT:
            steer = max(-MAX_ANGLE, steer - STEER_STEP)
            steer_pressed = True
        elif key == Keyboard.RIGHT:
            steer = min(MAX_ANGLE, steer + STEER_STEP)
            steer_pressed = True
        elif key == ord('I') or key == ord('i'):
            cruise_speed = min(SPEED_MAX, cruise_speed + SPEED_STEP)
            driver.setCruisingSpeed(cruise_speed)
            print(f"[SPD] {cruise_speed} km/h")
        elif key == ord('K') or key == ord('k'):
            cruise_speed = max(SPEED_MIN, cruise_speed - SPEED_STEP)
            driver.setCruisingSpeed(cruise_speed)
            print(f"[SPD] {cruise_speed} km/h")
        elif key == ord('S') or key == ord('s'):
            nav_cmd = CMD_CONTINUE; print("[NAV] CONTINUE")
        elif key == ord('W') or key == ord('w'):
            nav_cmd = CMD_STRAIGHT; print("[NAV] RECTO")
        elif key == ord('A') or key == ord('a'):
            nav_cmd = CMD_LEFT;     print("[NAV] IZQUIERDA")
        elif key == ord('D') or key == ord('d'):
            nav_cmd = CMD_RIGHT;    print("[NAV] DERECHA")
        elif key == ord('M') or key == ord('m'):
            if gps_pos:
                pt = (gps_pos[0], gps_pos[1])
                too_close = any(
                    math.sqrt((pt[0]-ix)**2 + (pt[1]-iy)**2) < 30
                    for ix, iy in intersections
                )
                if too_close:
                    print("[GPS] Punto ya marcado cerca — ignorado")
                else:
                    intersections.append(list(pt))
                    save_intersections()
                    print(f"[GPS] Cruce #{len(intersections)}: x={pt[0]:.1f} y={pt[1]:.1f}")
            else:
                print("[GPS] Sin señal GPS")
        elif key == ord('F') or key == ord('f'):
            auto_mode = not auto_mode
            steer = 0.0; auto_steer = 0.0
            print(f"[MODE] {'AUTO-FOLLOW' if auto_mode else 'MANUAL'}")
        elif key == ord('Q') or key == ord('q'):
            csv_file.close()
            print(f"[CIL] Fin. {img_count} imágenes")
            driver.setCruisingSpeed(0)
            break
        key = keyboard.getKey()

    capture_rate = CAPTURE_EVERY_CONT if nav_cmd == CMD_CONTINUE else CAPTURE_EVERY
    if frame % capture_rate == 0:
        raw = camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if auto_mode:
            raw_s = lane_follow_steer(bgr)
            if raw_s is not None:
                auto_steer = auto_steer * LANE_EMA + raw_s * (1.0 - LANE_EMA)
                delta = float(np.clip(auto_steer - steer, -LANE_RATE_MAX, LANE_RATE_MAX))
                steer = float(np.clip(steer + delta, -MAX_ANGLE, MAX_ANGLE))
            else:
                steer *= 0.97
        else:
            if not steer_pressed:
                steer *= CENTER_DECAY
                if abs(steer) < 0.003:
                    steer = 0.0
        driver.setSteeringAngle(steer)

        name = f"img_{img_count:06d}.jpg"
        cv2.imwrite(os.path.join(DATA_DIR, name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        spd = driver.getCurrentSpeed()
        csv_writer.writerow([f"data/images/{name}", round(steer, 5), round(spd, 2), nav_cmd])
        csv_file.flush()
        cmd_counts[nav_cmd] += 1
        img_count += 1
        draw_hud(nav_cmd, steer, spd, img_count, cmd_counts, dist_m, auto_mode)
        if img_count % 100 == 0:
            print(f"[CIL] {img_count} | S:{cmd_counts[0]} W:{cmd_counts[1]} A:{cmd_counts[2]} D:{cmd_counts[3]} | {'AUTO' if auto_mode else 'MAN'}")
    else:
        if not auto_mode and not steer_pressed:
            steer *= CENTER_DECAY
            if abs(steer) < 0.003:
                steer = 0.0
        driver.setSteeringAngle(steer)

csv_file.close()
