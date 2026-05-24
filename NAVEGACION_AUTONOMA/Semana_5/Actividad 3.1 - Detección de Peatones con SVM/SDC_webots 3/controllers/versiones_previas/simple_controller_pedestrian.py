# Controlador Actividad 3.1 — PID lane following + SVM pedestrian detection
from controller import Display, Keyboard
from vehicle import Car, Driver
from skimage.feature import hog
import numpy as np
import cv2
import joblib
import os
import time
import threading
from datetime import datetime

_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))

HOG_WIN_W      = 64
HOG_WIN_H      = 128
SLIDE_STEP     = 32
DETECT_EVERY_N = 10

_ped_lock      = threading.Lock()
_ped_found     = False
_ped_busy      = False
_ped_result_id = 0

def _scan_scale(bgr_scaled, model):
    h, w = bgr_scaled.shape[:2]
    if h < HOG_WIN_H or w < HOG_WIN_W:
        return False
    x0    = int(w * 0.35)
    x1    = int(w * 0.65)
    ystep = max(16, HOG_WIN_H // 4)
    for y in range(0, h - HOG_WIN_H + 1, ystep):
        for x in range(x0, min(x1, w - HOG_WIN_W + 1), SLIDE_STEP):
            win  = cv2.cvtColor(bgr_scaled[y:y+HOG_WIN_H, x:x+HOG_WIN_W],
                                cv2.COLOR_BGR2GRAY)
            feat = hog(win, orientations=11, pixels_per_cell=(16,16),
                       cells_per_block=(2,2), transform_sqrt=False,
                       feature_vector=True)
            if model.predict([feat])[0] == 1:
                return True
    return False

def _svm_worker(bgr, model):
    global _ped_found, _ped_busy, _ped_result_id
    h, w = bgr.shape[:2]
    roi  = bgr[int(h * 0.40):int(h * 0.85), :]
    found = False
    for scale in (2.0, 1.0):
        scaled = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)))
        if _scan_scale(scaled, model):
            found = True
            break
    print(f"[SVM] found={found}")
    with _ped_lock:
        _ped_found     = found
        _ped_busy      = False
        _ped_result_id += 1


BRAKE_ON_DETECT = False   # True = frena al detectar peatón / False = solo muestra label

CRUISE_SPEED   = 30
MAX_ANGLE      = 0.5
DEBOUNCE_TIME  = 0.1
MIN_ABS_SLOPE  = 0.6
MAX_STEER_RATE = 0.03

Kp = 0.28
Ki = 0.01
Kd = 0.01

YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_gray(display, gray):
    rgb = np.dstack((gray, gray, gray))
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def apply_roi(edges, h, w):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), int(h * 0.90)),
        (int(w * 0.35), int(h * 0.60)),
        (int(w * 0.65), int(h * 0.60)),
        (int(w * 0.90), int(h * 0.90))
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines_by_slope(lines):
    if lines is None:
        return None
    filtered = [l for l in lines
                if l[0][2] != l[0][0]
                and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(filtered) if filtered else None

def compute_lane_center(lines):
    if lines is None:
        return None
    left_x, right_x, all_x = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid   = (x1 + x2) / 2
        all_x.append(mid)
        (left_x if slope < 0 else right_x).append(mid)
    if left_x and right_x:
        return (np.mean(left_x) + np.mean(right_x)) / 2.0
    return np.mean(all_x) if all_x else None

def main():
    global _ped_found, _ped_busy, _ped_result_id

    svm_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    if svm_model:
        print("[OK] Modelo SVM cargado")
    else:
        print("[AVISO] Modelo no encontrado — solo PID")

    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera  = robot.getDevice("camera")
    camera.enable(timestep)

    display = robot.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    dw, dh   = display.getWidth(), display.getHeight()
    setpoint = dw / 2.0

    integral        = 0.0
    previous_error  = 0.0
    previous_time   = time.time()
    steering        = 0.0
    no_line_frames  = 0
    frame_count     = 0
    ped_score       = 0
    last_result_id  = 0
    last_press      = {}

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("PID + SVM activo")

    while robot.step() != -1:
        t  = time.time()
        dt = max(t - previous_time, 1e-3)
        frame_count += 1

        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))
        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        edges = cv2.bitwise_or(cv2.Canny(grey, 50, 150),
                               cv2.Canny(ymask, 50, 150))
        roi   = apply_roi(edges, dh, dw)
        lines = filter_lines_by_slope(
                    cv2.HoughLinesP(roi, 1, np.pi/180, 20,
                                    minLineLength=20, maxLineGap=15))
        lane_center = compute_lane_center(lines)

        if svm_model and frame_count % DETECT_EVERY_N == 0:
            with _ped_lock:
                busy = _ped_busy
            if not busy:
                with _ped_lock:
                    _ped_busy = True
                threading.Thread(target=_svm_worker,
                                 args=(frame.copy(), svm_model),
                                 daemon=True).start()

        with _ped_lock:
            cur_id        = _ped_result_id
            raw_detection = _ped_found

        if cur_id != last_result_id:
            last_result_id = cur_id
            if raw_detection:
                ped_score = min(ped_score + 1, 3)
            else:
                ped_score = 0

        peaton = ped_score >= 3

        viz = grey.copy()
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(viz, (x1, y1), (x2, y2), 255, 2)
        display_gray(display, viz)
        if peaton:
            display.setColor(0xFF0000)
            display.drawText("PEATON", 2, 2)
        display.setColor(0xFFFFFF)
        display.drawText(f"V:{CRUISE_SPEED}km/h", 2, 14)
        display.drawText(f"St:{steering:.3f}",    2, 24)

        if peaton and BRAKE_ON_DETECT:
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            driver.setHazardFlashers(True)
            steering = 0.0
            integral = 0.0
            previous_time = t
            continue

        driver.setHazardFlashers(peaton)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        if lane_center is not None:
            no_line_frames = 0
            error     = (lane_center - setpoint) / setpoint
            integral += error * dt
            integral  = max(-0.5, min(0.5, integral))
            raw_steer = Kp*error + Ki*integral + Kd*(error - previous_error) / dt
            raw_steer = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steer))
            steering  = max(steering - MAX_STEER_RATE,
                            min(steering + MAX_STEER_RATE, raw_steer))
            previous_error = error
        else:
            no_line_frames += 1
            integral       *= 0.6
            previous_error  = 0.0
            if no_line_frames > 10:
                steering *= 0.95

        driver.setSteeringAngle(steering)
        previous_time = t

        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and t - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = t
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)

if __name__ == "__main__":
    main()
