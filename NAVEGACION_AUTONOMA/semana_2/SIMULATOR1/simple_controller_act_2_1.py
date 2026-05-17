# Lane detection controller with PID steering
# Pipeline: Camera -> Grayscale -> Canny -> ROI -> HoughLinesP -> PID -> SteeringAngle

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# ── Vehicle configuration ──────────────────────────────────────────────────────
SPEED         = 50        # constant speed in km/h (minimum required by activity)
MAX_ANGLE     = 0.5       # maximum steering angle in radians
DEFAULT_ANGLE = 0.0       # go straight when no lane lines are detected
SPEED_INCR    = 5         # manual speed increment
ANGLE_INCR    = 0.05      # manual angle increment
DEBOUNCE_TIME = 0.1       # seconds between key repeats
MANUAL_MODE   = False     # False = PID autopilot on start, M to toggle manual

# ── Canny edge detection parameters ───────────────────────────────────────────
CANNY_LOW   = 50        # lower threshold: edges with gradient below this are rejected
CANNY_HIGH  = 150       # upper threshold: edges above this are always accepted

# ── Hough transform parameters (HoughLinesP) ──────────────────────────────────
# HoughLinesP detects line segments (not infinite lines like HoughLines)
HOUGH_RHO        = 1           # distance resolution in pixels
HOUGH_THETA      = np.pi / 180 # angular resolution: 1 degree
HOUGH_THRESHOLD  = 30          # minimum votes to consider a line
HOUGH_MIN_LENGTH = 30          # minimum segment length in pixels — filters short noise segments
HOUGH_MAX_GAP    = 100         # maximum gap between segments to be joined into one line

# ── Horizontal line filter ─────────────────────────────────────────────────────
# Lines where |y2 - y1| < MIN_VERT are mostly horizontal (road markings at intersections)
# and would give wrong error readings — we skip them
MIN_VERT_DIFF = 10

# ── PID controller gains ───────────────────────────────────────────────────────
# Kp: proportional — how strongly to react to current error
# Ki: integral     — corrects accumulated drift over time
# Kd: derivative   — dampens oscillations by reacting to rate of change
Kp = 0.003
Ki = 0.0001
Kd = 0.001


def get_image(camera):
    """Read raw camera image and convert to numpy BGRA array."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def preprocess(image):
    """Convert BGRA to grayscale and apply Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    return edges


def apply_roi(edges, height, width):
    """
    Define and apply a Region of Interest (ROI) to focus on the road ahead.

    The ROI is a trapezoid covering the lower 40% of the image:
      - Bottom-left  (0, height)
      - Top-left     (0, height * 0.6)
      - Top-right    (width, height * 0.6)
      - Bottom-right (width, height)

    Pixels outside the ROI are set to black so Hough only sees road area.
    fillPoly draws the trapezoid filled with white on a black mask,
    then we AND it with the edge image.
    """
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0,     height),
        (0,     int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(roi_edges):
    """
    Apply HoughLinesP to the ROI edge image.

    Returns a list of line segments [x1, y1, x2, y2], or empty list if none found.
    Each segment needs at least HOUGH_MIN_LENGTH pixels and gaps ≤ HOUGH_MAX_GAP
    are merged into a single segment.
    """
    lines = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )
    if lines is None:
        return []
    return lines.reshape(-1, 4)  # flatten to list of [x1, y1, x2, y2]


def compute_error(lines, setpoint):
    """
    Calculate the PID error from detected lane lines.

    For each line:
      1. Skip lines that are mostly horizontal (|y2 - y1| < MIN_VERT_DIFF)
         — these appear at intersections and don't represent lane markings
      2. Compute the horizontal midpoint: mid_x = (x1 + x2) / 2
      3. Compute distance to setpoint: dist = mid_x - setpoint

    The error used by PID is the one with smallest absolute distance to setpoint
    (i.e., the line closest to the center — most likely the yellow lane line).

    Returns the error value, or None if no valid lines found.
    """
    best_error = None
    for x1, y1, x2, y2 in lines:
        if abs(y2 - y1) < MIN_VERT_DIFF:
            continue  # skip horizontal lines
        mid_x = (x1 + x2) / 2
        error = mid_x - setpoint
        if best_error is None or abs(error) < abs(best_error):
            best_error = error
    return best_error


def display_image(display, image):
    """Show grayscale image on Webots onboard display."""
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


def main():
    robot  = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Image dimensions — used for setpoint and ROI
    width    = camera.getWidth()
    height   = camera.getHeight()
    setpoint = width / 2  # target: keep lane line at horizontal center

    # PID state variables
    integral      = 0.0
    prev_error    = 0.0
    prev_time     = time.time()

    # Manual control state
    speed      = 10
    angle      = 0.0
    last_press = {}
    manual     = MANUAL_MODE

    driver.setCruisingSpeed(speed)
    print("Modo PID AUTOPILOT activo. Presiona M para cambiar a manual.")
    print("↑↓ velocidad | ←→ dirección | A captura imagen | M cambia modo")

    while robot.step() != -1:
        current_time = time.time()
        dt = current_time - prev_time
        if dt <= 0:
            dt = 1e-6
        prev_time = current_time

        # ── Image pipeline (siempre corre para mostrar cámara) ─────────────────
        image = get_image(camera)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        display_image(display_img, gray)

        # ── Keyboard ───────────────────────────────────────────────────────────
        key = keyboard.getKey()
        if key != -1:
            if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
                pass
            else:
                last_press[key] = current_time

                if key == ord('M'):
                    manual = not manual
                    print(f"Modo: {'MANUAL' if manual else 'PID AUTOPILOT'}")

                elif key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"Imagen guardada: {ts}.png")

                elif manual:
                    if key == keyboard.UP:
                        speed = min(speed + SPEED_INCR, 250)
                        print(f"Velocidad: {speed}")
                    elif key == keyboard.DOWN:
                        speed = max(speed - SPEED_INCR, 0)
                        print(f"Velocidad: {speed}")
                    elif key == keyboard.RIGHT:
                        angle = min(angle + ANGLE_INCR, MAX_ANGLE)
                    elif key == keyboard.LEFT:
                        angle = max(angle - ANGLE_INCR, -MAX_ANGLE)

        # ── Control output ─────────────────────────────────────────────────────
        if manual:
            driver.setCruisingSpeed(speed)
            driver.setSteeringAngle(angle)
        else:
            # PID autopilot
            edges   = preprocess(image)
            roi     = apply_roi(edges, height, width)
            lines   = detect_lines(roi)
            error   = compute_error(lines, setpoint)

            if error is None:
                steering = DEFAULT_ANGLE
                integral = 0.0
            else:
                integral  += error * dt
                derivative = (error - prev_error) / dt
                steering   = Kp * error + Ki * integral + Kd * derivative
                prev_error = error

            steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
            driver.setCruisingSpeed(SPEED)
            driver.setSteeringAngle(steering)


if __name__ == "__main__":
    main()
