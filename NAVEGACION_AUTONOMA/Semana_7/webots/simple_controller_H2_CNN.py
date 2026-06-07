# Controlador Webots — Actividad 4.2
# Base: simple_controller_H2.py (Semana 2, lane-following con PID + HSV amarillo)
# Extensión: detección de señales de tráfico con CNN entrenada en GTSRB
#
# Pipeline de detección:
#   1. Cada DETECT_EVERY frames: buscar regiones de color rojo/azul en el frame
#   2. Filtrar contornos por área y aspecto → candidatos a señal
#   3. Recortar, redimensionar a 32×32 y normalizar → predecir con CNN
#   4. Si confianza > CONF_THRESHOLD: mostrar etiqueta en pantalla

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

# ── Intentar importar TensorFlow para la CNN ──────────────────────────────────
try:
    import tensorflow as tf
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("[CNN] TensorFlow no disponible — detección de señales deshabilitada")

# =============================================================================
# CONSTANTES DE CONTROL (sin cambios respecto a simple_controller_H2.py)
# =============================================================================
DEBOUNCE_TIME  = 0.1    # segundos anti-rebote de teclado
MAX_ANGLE      = 0.5    # ángulo máximo de giro (rad)
MAX_SPEED      = 250    # velocidad máxima (km/h)
SPEED_INCR     = 5
ANGLE_INCR     = 0.05
MIN_ABS_SLOPE  = 0.4    # umbral de pendiente para filtrar cebras

# =============================================================================
# CONSTANTES DE DETECCIÓN CNN
# =============================================================================
IMG_SIZE       = 32     # tamaño de entrada del modelo GTSRB
CONF_THRESHOLD = 0.70   # confianza mínima para reportar una detección
DETECT_EVERY   = 10     # ejecutar CNN cada N frames (evita carga excesiva)
MIN_AREA       = 150    # área mínima (px²) de un contorno candidato a señal

# Ruta al modelo exportado desde el notebook (relativa al controlador)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "modelo_gtsrb.keras"
)

# Nombres de las 43 clases GTSRB (índice = ClassId)
CLASS_NAMES = {
     0: "Lim. 20km/h",      1: "Lim. 30km/h",       2: "Lim. 50km/h",
     3: "Lim. 60km/h",      4: "Lim. 70km/h",        5: "Lim. 80km/h",
     6: "Fin lim. 80",       7: "Lim. 100km/h",       8: "Lim. 120km/h",
     9: "No adelantar",     10: "No adelant. >3.5t", 11: "Ceder derecho",
    12: "Prioridad",        13: "Ceder paso",         14: "STOP",
    15: "Sin vehículos",    16: "Sin >3.5t",          17: "No entrar",
    18: "Precaución",       19: "Curva izq.",         20: "Curva der.",
    21: "Doble curva",      22: "Bache",              23: "Pav. resbaloso",
    24: "Ancho reduce",     25: "Obras",              26: "Semáforo",
    27: "Peatones",         28: "Niños",              29: "Bicicletas",
    30: "Hielo/Nieve",      31: "Animales",           32: "Fin restricciones",
    33: "Girar derecha",    34: "Girar izquierda",    35: "Solo adelante",
    36: "Ade. o derecha",   37: "Ade. o izquierda",  38: "Derecha",
    39: "Izquierda",        40: "Glorieta",           41: "Fin no adelantar",
    42: "Fin no adel. >3.5t"
}

# =============================================================================
# FUNCIONES DE LANE-FOLLOWING (sin modificaciones desde Semana 2)
# =============================================================================

def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )

def display_image(display, image):
    display.setColor(0x000000)
    display.fillRectangle(0, 0, display.getWidth(), display.getHeight())
    image_rgb = np.dstack((image, image, image))
    ref = display.imageNew(image_rgb.tobytes(), Display.RGB,
                           width=image_rgb.shape[1], height=image_rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def region_of_interest(edges):
    h, w = edges.shape
    vertices = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.6)),
        (int(w * 0.65), int(h * 0.6)),
        (int(w * 0.90), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)

def hough_lines(roi_edges):
    return cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 20,
                            minLineLength=20, maxLineGap=15)

def filter_lines_by_slope(lines, min_abs_slope=MIN_ABS_SLOPE):
    if lines is None:
        return None
    filtered = [l for l in lines
                if l[0][2] != l[0][0]
                and abs((l[0][3] - l[0][1]) / (l[0][2] - l[0][0])) >= min_abs_slope]
    return np.array(filtered) if filtered else None

def draw_lines(image, lines):
    img = np.zeros_like(image)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return img

def compute_lane_center(lines):
    if lines is None:
        return None
    left_pts, right_pts, all_pts = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        all_pts.extend([x1, x2])
        (left_pts if slope < 0 else right_pts).extend([x1, x2])
    if left_pts and right_pts:
        return (np.mean(left_pts) + np.mean(right_pts)) / 2.0
    return np.mean(all_pts) if all_pts else None

# =============================================================================
# FUNCIONES DE DETECCIÓN DE SEÑALES (nuevas en Actividad 4.2)
# =============================================================================

def cargar_modelo_cnn(model_path):
    """Carga el modelo Keras entrenado en GTSRB. Devuelve None si falla."""
    if not CNN_AVAILABLE:
        return None
    abs_path = os.path.abspath(model_path)
    if not os.path.exists(abs_path):
        print(f"[CNN] Modelo no encontrado en: {abs_path}")
        return None
    try:
        model = tf.keras.models.load_model(abs_path)
        print(f"[CNN] Modelo cargado: {abs_path}")
        return model
    except Exception as e:
        print(f"[CNN] Error al cargar modelo: {e}")
        return None


def detectar_regiones_senales(bgr_img):
    """Segmentación por color para encontrar candidatos a señales de tráfico.

    Busca píxeles rojos (señales de velocidad, stop, precaución) y azules
    (señales de obligación). Opera sobre el 65 % superior de la imagen
    (la mitad inferior es calzada, sin señales relevantes).

    Devuelve lista de bounding-boxes (x1, y1, x2, y2) en coordenadas del
    frame original.
    """
    h, w = bgr_img.shape[:2]
    roi_h = int(h * 0.65)         # solo zona donde aparecen señales
    roi   = bgr_img[:roi_h, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Rojo: el canal H envuelve en HSV → dos rangos
    red_lo = cv2.inRange(hsv, np.array([0,  120, 70]), np.array([10,  255, 255]))
    red_hi = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([180, 255, 255]))
    red_mask  = cv2.bitwise_or(red_lo, red_hi)

    # Azul: señales de obligación (girar, glorieta, etc.)
    blue_mask = cv2.inRange(hsv, np.array([100, 120, 70]), np.array([130, 255, 255]))

    combined = cv2.bitwise_or(red_mask, blue_mask)

    # Morfología: cerrar huecos y eliminar ruido pequeño
    k = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)

    contornos, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contornos:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / float(ch) if ch > 0 else 0
        if aspect < 0.3 or aspect > 3.0:    # descartar formas muy elongadas
            continue
        pad = 4
        bboxes.append((
            max(0, x - pad), max(0, y - pad),
            min(w, x + cw + pad), min(roi_h, y + ch + pad)
        ))
    return bboxes


def clasificar_region(bgr_img, bbox, model):
    """Recorta la región, la preprocesa para la CNN y devuelve (clase, confianza)."""
    x1, y1, x2, y2 = bbox
    crop = bgr_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    img_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_res  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_res.astype(np.float32) / 255.0
    batch    = np.expand_dims(img_norm, axis=0)

    probs      = model.predict(batch, verbose=0)[0]
    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id])
    return class_id, confidence


def detectar_senales(bgr_img, model):
    """Pipeline completo: segmentación → recorte → CNN → filtro por confianza.

    Devuelve la detección más confiada del frame actual, o None.
    """
    if model is None:
        return None

    bboxes = detectar_regiones_senales(bgr_img)
    mejor  = None

    for bbox in bboxes:
        class_id, conf = clasificar_region(bgr_img, bbox, model)
        if class_id is None:
            continue
        if conf >= CONF_THRESHOLD:
            if mejor is None or conf > mejor["confidence"]:
                mejor = {
                    "class_id":   class_id,
                    "label":      CLASS_NAMES.get(class_id, f"Clase {class_id}"),
                    "confidence": conf,
                    "bbox":       bbox
                }
    return mejor


def mostrar_deteccion(display, deteccion, frame_count):
    """Escribe la etiqueta de la señal detectada en el display de Webots."""
    if deteccion is None:
        return
    label = f"{deteccion['label']} ({deteccion['confidence']*100:.0f}%)"
    display.setColor(0x00FF00)     # verde
    display.setFont("Arial", 8, True)
    display.drawText(label, 2, 2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Parámetros PID (idénticos a simple_controller_H2.py) ─────────────────
    speed          = 50
    kp             = 0.28
    ki             = 0.01
    kd             = 0.01
    integral       = 0.0
    previous_error = 0.0
    previous_time  = time.time()
    steering       = 0.0
    no_line_frames = 0
    MAX_STEER_RATE = 0.03
    last_press     = {}
    frame_count    = 0

    # ── Detección de señales: estado persistente entre frames ─────────────────
    ultima_deteccion   = None   # última detección válida
    frames_sin_deteccion = 0    # para limpiar la etiqueta tras N frames

    # ── Inicialización de Webots ──────────────────────────────────────────────
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera     = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = robot.getDevice("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── Carga de la CNN ───────────────────────────────────────────────────────
    model_cnn = cargar_modelo_cnn(MODEL_PATH)

    # ── Bucle principal ───────────────────────────────────────────────────────
    while robot.step() != -1:
        frame_count  += 1
        image         = get_image(camera)
        display_w     = display_img.getWidth()
        display_h     = display_img.getHeight()

        # Conversión BGRA → BGR y redimensión al display
        bgr_image   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        resized_bgr = cv2.resize(bgr_image, (display_w, display_h))

        # ── Lane-following (pipeline HSV amarillo) ────────────────────────────
        hsv         = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv,
                                  np.array([15, 80, 80]),
                                  np.array([35, 255, 255]))
        canny        = cv2.Canny(yellow_mask, 50, 150)
        roi_edges    = region_of_interest(canny)
        lines        = filter_lines_by_slope(hough_lines(roi_edges))
        lane_center  = compute_lane_center(lines)

        current_time = time.time()
        dt = max(current_time - previous_time, 1e-3)

        if lane_center is not None:
            no_line_frames = 0
            error      = (lane_center - display_w / 2.0) / (display_w / 2.0)
            integral   = max(-0.5, min(0.5, integral + error * dt))
            derivative = (error - previous_error) / dt
            raw_steer  = kp * error + ki * integral + kd * derivative
            raw_steer  = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steer))
            steering   = max(steering - MAX_STEER_RATE,
                             min(steering + MAX_STEER_RATE, raw_steer))
            previous_error = error
        else:
            no_line_frames += 1
            integral       *= 0.6
            previous_error  = 0.0
            if no_line_frames > 10:
                steering *= 0.95

        previous_time = current_time

        # ── Detección de señales cada DETECT_EVERY frames ─────────────────────
        if model_cnn is not None and frame_count % DETECT_EVERY == 0:
            det = detectar_senales(resized_bgr, model_cnn)
            if det is not None:
                ultima_deteccion     = det
                frames_sin_deteccion = 0
                print(f"[Señal] {det['label']}  conf={det['confidence']:.2f}")
            else:
                frames_sin_deteccion += 1
                if frames_sin_deteccion > 30:   # ~30 × DETECT_EVERY frames sin ver nada
                    ultima_deteccion = None

        # ── Display: debug de carriles + etiqueta de señal ───────────────────
        line_img      = draw_lines(np.zeros((display_h, display_w, 3), np.uint8), lines)
        line_gray     = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        debug_view    = cv2.addWeighted(roi_edges, 0.7, line_gray, 1.0, 0)
        display_image(display_img, debug_view)
        mostrar_deteccion(display_img, ultima_deteccion, frame_count)

        # ── Teclado ───────────────────────────────────────────────────────────
        key = keyboard.getKey()
        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue
        last_press[key] = current_time

        if key == ord('A'):
            ts   = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            path = os.getcwd() + f"/{ts}.png"
            camera.saveImage(path, 1)
            print("Imagen guardada:", path)

        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
