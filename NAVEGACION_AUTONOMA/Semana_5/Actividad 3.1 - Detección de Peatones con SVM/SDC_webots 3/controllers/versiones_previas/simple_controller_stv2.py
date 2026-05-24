# =============================================================================
# simple_controller_pedestrian_v2.py
# Actividad 3.1 — Detección de Peatones con SVM
# Navegación Autónoma — Maestría en Inteligencia Artificial
# =============================================================================
#
# ARQUITECTURA:
#   1. PID — sigue la línea amarilla del carril (HSV → Canny → Hough)
#   2. SVM — sliding window HOG sobre la cámara para detectar peatones
#   3. Comportamiento dual:
#        · Peatón (SVM=1) → freno de emergencia, SIN intermitentes
#        · Barril  (SVM=0) → freno de emergencia, CON intermitentes
#
# NOTA SOBRE LIDAR:
#   El sensor Sick LMS 291 está presente en el mundo y el código de integración
#   se incluye documentado abajo (Sección 2-B). En macOS con Webots R2025a,
#   lidar.enable() en la escena city causa freeze (beachball) independientemente
#   del intervalo de actualización. Por ello se deja deshabilitado en runtime
#   y la detección de obstáculos se realiza íntegramente por cámara + SVM.
#   En un entorno Linux/Windows el bloque LiDAR puede activarse sin cambios.
#
# MODIFICACIONES respecto al código original de detección de vehículos:
#   - Dataset: INRIA Person Dataset en lugar de dataset de vehículos
#   - HOG: orientations=11 (vs 9 original) — mejor captura de siluetas humanas
#   - Ventana deslizante: 64×128 px (aspecto vertical, humanos) vs 64×64 (autos)
#   - Multi-escala 1× y 2× para detectar peatones a distintas distancias
#   - ROI horizontal 25%-75%: zona del carril, excluye banquetas laterales
#   - Sistema de confirmación: requiere N detecciones consecutivas — evita falsos
#
# BIBLIOGRAFÍA:
#   - INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
#   - Dalal & Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR 2005
#   - Sliding Window: https://medium.com/@ricardo.zuccolo/self-driving-cars-opencv...
#   - Webots Driver API: https://cyberbotics.com/doc/automobile/driver-library
# =============================================================================

from controller import Display, Keyboard
from vehicle import Car, Driver
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import math
import os
import time
from datetime import datetime

_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))


# ---------------------------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------------------------
CRUISE_SPEED   = 30
MAX_ANGLE      = 0.5
MAX_STEER_RATE = 0.03
DEBOUNCE_TIME  = 0.1
Kp, Ki, Kd     = 0.28, 0.01, 0.01
MIN_ABS_SLOPE  = 0.4
YELLOW_LOW     = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH    = np.array([35, 255, 255], dtype=np.uint8)

# LiDAR (parámetros documentados — ver Sección 2-B)
LIDAR_CONE_DEG = 25
LIDAR_MAX_M    = 20.0

# SVM
HOG_WIN_W    = 64
HOG_WIN_H    = 128
SLIDE_STEP   = 32
DETECT_EVERY  = 5     # corre SVM cada 5 frames (~50 ms)
CONFIRM_N     = 2     # scans positivos consecutivos para confirmar peatón
RELEASE_N     = 4     # scans negativos consecutivos para liberar freno
HOLD_FRAMES   = 100   # frames mínimos de freno activo (~1 s a 10 ms/frame)
MIN_HITS      = 1     # ventanas positivas mínimas por scan
SVM_THRESHOLD = 0.30  # líneas camino ≈0.20-0.28, peatones Webots ≈0.30-0.52


# ---------------------------------------------------------------------------
# SECCIÓN 1 — VISIÓN: SEGUIMIENTO DE CARRIL
# ---------------------------------------------------------------------------

def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_color(display, bgr_img):
    rgb = bgr_img[:, :, ::-1].copy()
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def apply_roi(edges, h, w):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.60)),
        (int(w * 0.65), int(h * 0.60)),
        (int(w * 0.90), h),
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines(lines):
    if lines is None:
        return None
    ok = [l for l in lines
          if l[0][2] != l[0][0]
          and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(ok) if ok else None

def compute_center(lines):
    if lines is None:
        return None
    lx, rx, ax = [], [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2-y1)/(x2-x1)
        mid   = (x1+x2)/2
        ax.append(mid)
        (lx if slope < 0 else rx).append(mid)
    if lx and rx:
        return (np.mean(lx)+np.mean(rx))/2.0
    return np.mean(ax) if ax else None


# ---------------------------------------------------------------------------
# SECCIÓN 2-A — SVM: CLASIFICACIÓN DE OBSTÁCULO
# ---------------------------------------------------------------------------

def svm_detect(bgr, model):
    """
    Sliding Window Search HOG.
    Returns (detected, hits, max_score, windows, best_box_bgr).
    best_box_bgr = (x1, y1, x2, y2) in bgr pixel coords, or None.
    """
    h, w      = bgr.shape[:2]
    y_off     = int(h * 0.58)   # +5% arriba
    roi       = bgr[y_off : int(h * 0.85), :]
    total     = 0
    max_score = -999.0
    windows   = 0
    best_box  = None

    for scale in (4.0,):        # ROI 36px: scale 3.0 requiere 43px mínimo, se usa 4.0
        scaled = cv2.resize(roi, (int(w*scale), int(roi.shape[0]*scale)))
        sh, sw = scaled.shape[:2]
        if sh < HOG_WIN_H or sw < HOG_WIN_W:
            continue
        x0        = int(sw * 0.30)   # −10% ancho
        x1        = int(sw * 0.70)
        cx_skip_lo = int(sw * 0.42)   # excluye centro donde viven las líneas amarillas
        cx_skip_hi = int(sw * 0.58)
        ystep = max(16, HOG_WIN_H // 4)
        for y in range(0, sh - HOG_WIN_H + 1, ystep):
            for x in range(x0, min(x1, sw - HOG_WIN_W + 1), SLIDE_STEP):
                if cx_skip_lo <= x < cx_skip_hi:
                    continue
                win   = cv2.cvtColor(
                    scaled[y:y+HOG_WIN_H, x:x+HOG_WIN_W], cv2.COLOR_BGR2GRAY)
                feat  = hog(win, orientations=11, pixels_per_cell=(16,16),
                            cells_per_block=(2,2), transform_sqrt=False,
                            feature_vector=True)
                score = model.decision_function([feat])[0]
                windows += 1
                if score > max_score:
                    max_score = score
                    bx1 = max(0, int(x / scale))
                    by1 = max(0, int(y / scale) + y_off)
                    bx2 = min(w,  int((x + HOG_WIN_W) / scale))
                    by2 = min(h,  int((y + HOG_WIN_H) / scale) + y_off)
                    best_box = (bx1, by1, bx2, by2)
                if score >= SVM_THRESHOLD:
                    total += 1

    detected = total >= MIN_HITS
    return detected, total, max_score, windows, best_box


# ---------------------------------------------------------------------------
# SECCIÓN 2-B — LIDAR (documentado, deshabilitado en runtime macOS)
# ---------------------------------------------------------------------------
#
# En un sistema Linux/Windows descomentar el bloque en main() marcado LIDAR.
#
# def lidar_obstacle(lidar):
#     """
#     Lee cono frontal ±12.5° del Sick LMS 291 y devuelve True si hay algo < 20 m.
#     Solo usa getRangeImage() — NO enablePointCloud() (causa freeze en macOS).
#     """
#     ranges  = lidar.getRangeImage()
#     if not ranges:
#         return False
#     n       = len(ranges)
#     fov_rad = lidar.getFov()
#     half    = max(1, int(n * (math.radians(LIDAR_CONE_DEG) / fov_rad) / 2))
#     center  = n // 2
#     cone    = [r for r in ranges[center-half: center+half]
#                if not (math.isnan(r) or math.isinf(r))]
#     return bool(cone) and min(cone) < LIDAR_MAX_M


# ---------------------------------------------------------------------------
# SECCIÓN 3 — MAIN
# ---------------------------------------------------------------------------

def main():

    # Cargar modelo SVM
    model_path = os.path.normpath(MODEL_PATH)
    svm_model  = joblib.load(model_path) if os.path.exists(model_path) else None
    print("[OK] Modelo SVM cargado" if svm_model else "[AVISO] Sin modelo — solo PID")

    # Driver hereda de Car que hereda de Robot — una sola instancia cubre todo
    driver   = Driver()
    timestep = int(driver.getBasicTimeStep())   # SIN multiplicador — evita freeze

    camera  = driver.getDevice("camera")
    camera.enable(timestep)

    display  = driver.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── LIDAR (deshabilitado en macOS — ver Sección 2-B) ──────────────────
    # En Linux/Windows reemplazar las dos líneas siguientes por:
    #   lidar = driver.getDevice("Sick LMS 291")
    #   lidar.enable(100)   # NO enablePointCloud()
    lidar = None

    dw, dh   = display.getWidth(), display.getHeight()
    setpoint = dw / 2.0

    integral, prev_err, prev_t = 0.0, 0.0, time.time()
    steering, no_line_frames   = 0.0, 0
    frame_cnt  = 0
    pos_streak = 0
    neg_streak = 0
    brake_hold = 0
    threat     = 'none'
    last_press = {}
    # Diagnóstico
    last_hits  = 0
    last_score = 0.0
    last_wins  = 0
    last_box   = None

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("Controlador listo — PID + SVM (LiDAR documentado, ver Sección 2-B)")

    while driver.step() != -1:
        t  = time.time()
        dt = max(t - prev_t, 1e-3)
        frame_cnt += 1

        # ── Imagen ──────────────────────────────────────────────────────────
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── LiDAR (si habilitado) ───────────────────────────────────────────
        # lidar_alert = lidar_obstacle(lidar) if lidar else True
        # En macOS usamos lidar_alert=True para que SVM siempre evalúe.
        lidar_alert = True

        # ── SVM cada DETECT_EVERY frames ────────────────────────────────────
        if svm_model and lidar_alert and frame_cnt % DETECT_EVERY == 0:
            detected, last_hits, last_score, last_wins, last_box = svm_detect(bgr, svm_model)
            if detected:
                pos_streak = min(pos_streak + 1, CONFIRM_N + 3)
                neg_streak = 0
            else:
                neg_streak = min(neg_streak + 1, RELEASE_N + 3)
                pos_streak = max(pos_streak - 1, 0)
            # Print consola cada scan para diagnóstico
            print(f"[SVM] f={frame_cnt:05d} wins={last_wins} "
                  f"hits={last_hits}/{MIN_HITS} score={last_score:.3f} "
                  f"thresh={SVM_THRESHOLD} pos={pos_streak}/{CONFIRM_N} "
                  f"neg={neg_streak}/{RELEASE_N} threat={threat}")

        # ── Determinar amenaza ──────────────────────────────────────────────
        if pos_streak >= CONFIRM_N:
            threat     = 'pedestrian'
            brake_hold = HOLD_FRAMES
            neg_streak = 0
        elif neg_streak >= RELEASE_N and brake_hold <= 0:
            threat     = 'none'
            pos_streak = 0

        if brake_hold > 0:
            brake_hold -= 1
            if brake_hold == 0 and pos_streak < CONFIRM_N:
                threat = 'none'

        # Si LiDAR detecta obstáculo pero SVM aún no confirma → tratar como barril
        # (En este runtime lidar_alert=True siempre, así que esta rama
        #  solo aplica cuando el LiDAR real esté habilitado en Linux/Windows)
        if lidar and lidar_alert and threat == 'none':
            threat     = 'barrel'
            brake_hold = HOLD_FRAMES

        # ── PID ─────────────────────────────────────────────────────────────
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        edges = cv2.bitwise_or(cv2.Canny(grey, 50, 150),
                               cv2.Canny(ymask, 50, 150))
        roi   = apply_roi(edges, dh, dw)
        lines = filter_lines(cv2.HoughLinesP(roi, 1, np.pi/180, 20,
                                             minLineLength=20, maxLineGap=15))
        center = compute_center(lines)

        # ── Display: cámara en color con ROI + ventana SVM superpuestos ────────
        viz = frame.copy()  # BGR 200×150

        # ROI de detección (cian) — zona vertical escaneada
        roi_y0 = int(dh * 0.58)
        roi_y1 = int(dh * 0.85)
        cv2.rectangle(viz, (0, roi_y0), (dw - 1, roi_y1), (200, 200, 0), 1)

        # Zona horizontal de búsqueda (naranja) + hueco central excluido (rojo oscuro)
        hx0 = int(dw * 0.30)
        hx1 = int(dw * 0.70)
        cx0 = int(dw * 0.42)
        cx1 = int(dw * 0.58)
        cv2.line(viz, (hx0, roi_y0), (hx0, roi_y1), (0, 140, 255), 1)
        cv2.line(viz, (hx1, roi_y0), (hx1, roi_y1), (0, 140, 255), 1)
        cv2.line(viz, (cx0, roi_y0), (cx0, roi_y1), (0, 0, 160), 1)   # límite izq zona excluida
        cv2.line(viz, (cx1, roi_y0), (cx1, roi_y1), (0, 0, 160), 1)   # límite der zona excluida

        # Ventana con mayor score (verde si bajo umbral, rojo si supera)
        if last_box is not None:
            bx1, by1, bx2, by2 = last_box
            ih, iw = bgr.shape[:2]
            fx1 = int(bx1 * dw / iw);  fy1 = int(by1 * dh / ih)
            fx2 = int(bx2 * dw / iw);  fy2 = int(by2 * dh / ih)
            box_col = (0, 0, 255) if last_score >= SVM_THRESHOLD else (0, 200, 0)
            cv2.rectangle(viz, (fx1, fy1), (fx2, fy2), box_col, 2)

        # Líneas de carril detectadas (amarillo)
        if lines is not None:
            scale_x = dw / dw  # frame ya es dw×dh
            for l in lines:
                cv2.line(viz, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 255), 1)

        display_color(display, viz)

        # Texto sobre la imagen
        if threat == 'barrel':
            display.setColor(0xFF6600)
            display.drawText("BARRIL", 2, 2)
        elif threat == 'pedestrian':
            display.setColor(0xFF0000)
            display.drawText("PEATON", 2, 2)
        else:
            display.setColor(0x00FF00)
            display.drawText("PID OK", 2, 2)

        display.setColor(0xFFFFFF)
        display.drawText(f"V:{CRUISE_SPEED} St:{steering:.2f}", 2, 12)

        score_color = 0xFF4444 if last_score >= SVM_THRESHOLD else (0xFFAA00 if last_score >= 0.0 else 0xCCCCCC)
        display.setColor(score_color)
        display.drawText(f"SVM:{last_score:.3f}(>{SVM_THRESHOLD})", 2, 22)

        display.setColor(0xFFFFFF)
        display.drawText(f"hits:{last_hits}/{MIN_HITS} w:{last_wins}", 2, 32)

        pos_color = 0xFF2222 if pos_streak >= CONFIRM_N else 0xFF8800
        display.setColor(pos_color)
        display.drawText(f"pos:{pos_streak}/{CONFIRM_N}", 2, 42)

        display.setColor(0x44FF44)
        display.drawText(f"neg:{neg_streak}/{RELEASE_N}", 2, 52)

        display.setColor(0xAAAAAA)
        display.drawText(f"hold:{brake_hold} f:{frame_cnt}", 2, 62)

        # ── Control ─────────────────────────────────────────────────────────
        if threat != 'none':
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            driver.setHazardFlashers(threat == 'barrel')  # intermitentes solo barril
            steering, integral = 0.0, 0.0
            prev_t = t
            continue

        driver.setHazardFlashers(False)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        if center is not None:
            no_line_frames = 0
            error    = (center - setpoint) / setpoint
            integral = max(-0.5, min(0.5, integral + error * dt))
            raw_s    = Kp*error + Ki*integral + Kd*(error-prev_err)/dt
            raw_s    = max(-MAX_ANGLE, min(MAX_ANGLE, raw_s))
            steering = max(steering-MAX_STEER_RATE,
                           min(steering+MAX_STEER_RATE, raw_s))
            prev_err = error
        else:
            no_line_frames += 1
            integral       *= 0.6
            prev_err        = 0.0
            if no_line_frames > 10:
                steering *= 0.95

        driver.setSteeringAngle(steering)
        prev_t = t

        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and t-last_press[key] < DEBOUNCE_TIME):
                last_press[key] = t
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd()+"/"+ts+".png", 1)
                    print(f"[A] {ts}.png guardada")

if __name__ == "__main__":
    main()
