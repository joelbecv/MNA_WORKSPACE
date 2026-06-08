# =============================================================================
# Actividad 4.2 — Detección de señales de tráfico USA con CNN en Webots
# Maestría en Inteligencia Artificial Aplicada — Tec de Monterrey
# Autor: Joel Arturo Becerril Balderas  A01797427
#
# RESUMEN DEL SISTEMA:
#   1. Cámara 128×128 captura el mundo Webots cada frame
#   2. Se aplica segmentación HSV para encontrar regiones con colores
#      característicos de señales (rojo, amarillo, blanco)
#   3. Cada región candidata se recorta (crop) y se pasa a la CNN
#   4. La CNN (modelo_us_webots.keras, 11 clases) devuelve la señal detectada
#   5. Un LiDAR (Sick LMS 291) guía la dirección siguiendo la pared derecha
#   6. El teclado permite override manual temporal
# =============================================================================

from controller import Display, Keyboard, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import json

# Carpeta del script: todos los archivos de debug se guardan aquí
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_DIR   = os.path.join(SCRIPT_DIR, "debug_crops")
# Limpiar crops previos al arrancar para no mezclar corridas
import shutil
if os.path.exists(CROP_DIR):
    shutil.rmtree(CROP_DIR)
os.makedirs(CROP_DIR)
_crop_n = 0   # contador de crops guardados (máx 200 por corrida)
_mask_n = 0   # contador para diagnóstico de máscaras HSV

try:
    import tensorflow as tf
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("[CNN] TensorFlow no disponible — solo modo LiDAR")

# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================

MAX_ANGLE      = 0.5    # ángulo máximo de dirección en radianes (~28°)
SPEED_INCR     = 2.0    # km/h que sube/baja cada tecla ↑↓
ANGLE_INCR     = 0.012  # radianes por pulso de dirección (teclas ←→)
STEER_DEBOUNCE = 0.07   # segundos mínimos entre pulsos — evita rebote de tecla
DEFAULT_SPEED  = 10.0   # velocidad de arranque en km/h

IMG_SIZE       = 32     # tamaño de entrada de la CNN (32×32 px, igual que GTSRB)
CONF_THRESHOLD = 0.45   # confianza mínima para contar una señal (no se usa actualmente)
CONFIRM_STREAK = 2      # veces consecutivas que debe verse la misma señal para confirmarla
DETECT_EVERY   = 3      # CNN corre cada 3 frames (no cada frame) para ahorrar CPU
MIN_AREA       = 18     # área mínima de contorno HSV en px² (filtra ruido)

# ── Parámetros del LiDAR ──────────────────────────────────────────────────────
# El Sick LMS 291 devuelve 181 distancias en un arco de 180°.
# Con el carro apuntando al oeste:
#   índice 0   = izquierda (sur)
#   índice 90  = frente (oeste)
#   índice 180 = derecha (norte) ← bordillo del carril derecho
WALL_TARGET  = 2.5   # distancia objetivo al bordillo derecho en metros
WALL_NO_WALL = 8.0   # si la pared está más lejos de esto, se considera "sin pared"
KP_WALL      = 0.15  # ganancia proporcional del controlador P de pared
FRONT_STOP   = 0.9   # distancia frontal para parar completamente (m)
FRONT_SLOW   = 1.8   # distancia frontal para reducir velocidad (m)
MANUAL_TICKS = 120   # frames que dura el override manual antes de volver a LiDAR
DETECT_GRACE = 60    # frames iniciales sin CNN (el modelo tarda en iniciar)
CROSS_HOLD   = 20    # frames que mantiene el último ángulo al perder la pared
CROSS_DECAY  = 0.90  # factor de decaimiento del ángulo cuando no hay pared

# ── Ruta del modelo CNN ───────────────────────────────────────────────────────
# Busca primero el modelo US (11 clases de señales americanas).
# Si no existe, cae al modelo GTSRB genérico (43 clases europeas).
_US_MODEL  = os.path.join(os.path.dirname(__file__), "..", "modelo_us_webots.keras")
_OLD_MODEL = os.path.join(os.path.dirname(__file__), "..", "modelo_gtsrb.keras")
MODEL_PATH = _US_MODEL if os.path.exists(_US_MODEL) else _OLD_MODEL
MAP_PATH   = os.path.join(os.path.dirname(__file__), "..", "us_class_map.json")

# ── Diccionario de nombres de señales (por ID GTSRB) ─────────────────────────
# El modelo US tiene 11 salidas (índices 0-10).
# us_class_map.json convierte índice_modelo → ID_GTSRB.
# Este diccionario convierte ID_GTSRB → nombre legible.
# IMPORTANTE: Las etiquetas de velocidad usan mph (señales USA de Webots).
CLASS_NAMES = {
     0: "Lim. 20km/h",      1: "Lim. 30km/h",       2: "Lim. 50km/h",
     3: "Lim. 55 mph",      4: "Lim. 65 mph",        5: "Lim. 80km/h",
     6: "Fin lim. 80",       7: "Lim. 100km/h",       8: "Lim. 120km/h",
     9: "No adelantar",     10: "No adel. >3.5t",    11: "Ceder derecho",
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
    42: "Fin no adel.>3.5t"
}

# =============================================================================
# FUNCIONES DE CÁMARA Y DISPLAY
# =============================================================================

def get_image(camera):
    """Lee la imagen cruda de la cámara Webots y la convierte a array NumPy BGRA."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))


def render_roi_overlay(display_cam, display_info, bgr_img, bboxes, deteccion, total):
    """
    Dibuja la imagen de cámara con overlays en display_cam
    y el panel de detección (señal actual + conteo) en display_info.

    bboxes: lista de (x1,y1,x2,y2) — regiones candidatas (magenta)
    deteccion: dict con 'label' y 'confidence' de la última señal confirmada
    total: número de señales únicas detectadas hasta ahora
    """
    dw = display_cam.getWidth()
    dh = display_cam.getHeight()

    # Escalar imagen al tamaño del display y dibujar ROI (amarillo) y candidatos (magenta)
    vis = cv2.resize(bgr_img, (dw, dh))
    rx     = int(dw * 0.25)
    ry_top = int(dh * 0.05)
    ry_bot = int(dh * 0.70)
    cv2.rectangle(vis, (rx, ry_top), (dw - 1, ry_bot), (0, 255, 255), 3)
    cv2.putText(vis, "ROI", (rx + 2, ry_top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    sx = dw / bgr_img.shape[1]
    sy = dh / bgr_img.shape[0]
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(vis,
                      (int(x1 * sx), int(y1 * sy)),
                      (int(x2 * sx), int(y2 * sy)),
                      (255, 0, 255), 2)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ir = display_cam.imageNew(vis_rgb.tobytes(), Display.RGB, width=dw, height=dh)
    display_cam.imagePaste(ir, 0, 0, False)

    # Panel de detección: fondo oscuro + conteo + señal actual
    dw2 = display_info.getWidth()
    dh2 = display_info.getHeight()
    display_info.setColor(0x0A0A1A)
    display_info.fillRectangle(0, 0, dw2, dh2)
    display_info.setColor(0x2255AA)
    display_info.fillRectangle(0, 0, dw2, 16)
    display_info.setColor(0xFFFFFF)
    display_info.drawText("CNN US Signs  Act. 4.2", 4, 4)

    cnt_c = 0x00FF88 if total >= 8 else 0xFFEE00  # verde si ≥8, amarillo si no
    display_info.setColor(0x111111)
    display_info.fillRectangle(0, 20, dw2, 24)
    display_info.setColor(cnt_c)
    display_info.drawText(f"Senales unicas: {total} / 16", 4, 28)

    if deteccion is not None:
        pct = int(deteccion["confidence"] * 100)
        display_info.setColor(0x004400)
        display_info.fillRectangle(0, 50, dw2, dh2 - 50)
        display_info.setColor(0x00FF44)
        display_info.drawText("DETECTADA:", 4, 56)
        display_info.setColor(0xFFFFFF)
        display_info.drawText(deteccion["label"], 4, 72)
        display_info.drawText(f"Conf: {pct}%", 4, 88)
        display_info.setColor(0x006600)
        display_info.fillRectangle(4, 100, 150, 10)
        display_info.setColor(0x00FF44)
        display_info.fillRectangle(4, 100, int(150 * pct / 100), 10)
    else:
        display_info.setColor(0x555566)
        display_info.drawText("(buscando senales...)", 4, 72)


# =============================================================================
# CNN — CARGA DEL MODELO
# =============================================================================

def cargar_modelo(path):
    """
    Carga modelo Keras y el mapeo índice→GTSRB desde us_class_map.json.

    El modelo tiene 11 salidas (0-10). El JSON convierte esos índices a
    IDs GTSRB para poder usar CLASS_NAMES. Sin el JSON, los índices se
    usan directamente (fallback para modelo GTSRB genérico).
    """
    if not CNN_AVAILABLE:
        return None, {}
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        print(f"[CNN] Modelo no encontrado: {abs_path}")
        return None, {}
    try:
        model = tf.keras.models.load_model(abs_path)
        print(f"[CNN] Modelo cargado: {abs_path}")
    except Exception as e:
        print(f"[CNN] Error: {e}")
        return None, {}

    idx_to_gtsrb = {}
    map_abs = os.path.abspath(MAP_PATH)
    if os.path.exists(map_abs):
        with open(map_abs) as f:
            idx_to_gtsrb = {int(k): int(v) for k, v in json.load(f).items()}
        print(f"[CNN] Mapeo US cargado: {len(idx_to_gtsrb)} clases")
    else:
        print("[CNN] Sin mapeo US — usando índices GTSRB directos")
    return model, idx_to_gtsrb


# =============================================================================
# DETECCIÓN HSV — BÚSQUEDA DE REGIONES CANDIDATAS
# =============================================================================

def detectar_regiones(bgr_img):
    """
    Segmenta la imagen por color HSV para encontrar regiones que podrían
    ser señales de tráfico. Devuelve lista de bboxes (x1,y1,x2,y2).

    PIPELINE:
    1. Se recorta una ROI (Region of Interest):
       - x: 25% derecho de la imagen (las señales están al costado)
       - y: 5% a 70% (elimina cielo y asfalto)
    2. Se calculan máscaras HSV para rojo, amarillo y blanco
    3. Se aplican "floor cuts" para eliminar marcas de piso
    4. Se encuentran contornos y se filtran por área y aspect ratio
    5. Adicionalmente, se buscan contornos amarillos y blancos de forma
       independiente para no perder señales pequeñas

    PROBLEMA HISTÓRICO — domain gap:
    Las señales en Webots aparecen más oscuras y menos saturadas que en
    los datasets de entrenamiento. Por eso los rangos HSV son amplios
    (especialmente S mínimo bajo).
    """
    global _mask_n
    h, w = bgr_img.shape[:2]

    # ── ROI ──────────────────────────────────────────────────────────────────
    y_min = int(h * 0.05)
    y_max = int(h * 0.70)
    x_min = int(w * 0.25)
    roi   = bgr_img[y_min:y_max, x_min:]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Máscaras de color ─────────────────────────────────────────────────────
    # Rojo: aparece en dos rangos HSV (0-12° y 155-180°) porque el hue es circular
    # S>=35: Webots desatura los colores vs escenas reales
    red1   = cv2.inRange(hsv, np.array([0,   35, 45]), np.array([12,  255, 255]))
    red2   = cv2.inRange(hsv, np.array([155, 35, 45]), np.array([180, 255, 255]))
    blue   = cv2.inRange(hsv, np.array([95,  50, 50]), np.array([135, 255, 255]))

    # Amarillo: tres rangos para cubrir sombra, luz directa y sobreexposición
    # Las señales de precaución (diamantes) son amarillas
    yel1   = cv2.inRange(hsv, np.array([15,  15, 40]),  np.array([45, 255, 255]))
    yel2   = cv2.inRange(hsv, np.array([18,  10, 25]),  np.array([40, 180, 120]))
    yel3   = cv2.inRange(hsv, np.array([20,   8, 150]), np.array([50,  60, 255]))
    yellow = yel1 | yel2 | yel3

    # Blanco: baja saturación, alto valor (señales de velocidad y One Way)
    white  = cv2.inRange(hsv, np.array([0,   0,  120]), np.array([180, 130, 255]))

    # ── Floor cuts — eliminar marcas de piso ──────────────────────────────────
    # El asfalto tiene marcas blancas (líneas de carretera) y reflejos amarillos.
    # Cortamos la parte inferior del ROI para ignorarlos.
    floor_cut_w = int(roi.shape[0] * 0.90)  # blanco: cortar último 10%
    floor_cut_y = int(roi.shape[0] * 0.75)  # amarillo: cortar último 25%
    white[floor_cut_w:, :] = 0
    yellow[floor_cut_y:, :] = 0

    # Máscara combinada y morfología (cierra huecos pequeños, elimina puntitos)
    mask = red1 | red2 | blue | yellow | white
    k    = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ── Diagnóstico periódico (cada 5 llamadas) ───────────────────────────────
    _mask_n += 1
    if _mask_n % 5 == 0:
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_roi_color.png"), roi)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_white.png"),  white)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_yellow.png"), yellow)
        any_hue_yellow = cv2.inRange(hsv, np.array([10, 0, 0]), np.array([50, 255, 255]))
        any_px = cv2.countNonZero(any_hue_yellow)
        if any_px > 0:
            ys = np.where(any_hue_yellow > 0)
            sv = hsv[ys[0], ys[1], 1]
            vv = hsv[ys[0], ys[1], 2]
            print(f"[YEL] H=10-50 px={any_px} S={sv.min()}-{sv.max()}(med={int(np.median(sv))}) "
                  f"V={vv.min()}-{vv.max()}(med={int(np.median(vv))}) -> filtered_y={cv2.countNonZero(yellow)}")
        else:
            print(f"[YEL] CERO pixels con H=10-50 en ROI. w={cv2.countNonZero(white)} cnts={len(contornos)}")
        w_px = cv2.countNonZero(white)
        white_cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w_areas = sorted([int(cv2.contourArea(c)) for c in white_cnts if cv2.contourArea(c) >= 5], reverse=True)[:5]
        print(f"[WHT] px={w_px}  contornos≥5px={len(w_areas)}  top_areas={w_areas}")

    # ── Contornos de la máscara combinada ────────────────────────────────────
    bboxes = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 25:
            continue  # ruido
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        # Señales son aproximadamente cuadradas o verticales (asp 0.4-2.0)
        # Descarta marcas de piso (muy anchas, asp > 2)
        if 0.4 <= asp <= 2.0:
            pad = 4
            bboxes.append((max(0, x + x_min - pad),
                           max(0, y + y_min - pad),
                           min(w, x + x_min + cw + pad),
                           min(y_max, y + y_min + ch + pad)))

    # ── Detección amarilla independiente ─────────────────────────────────────
    # Las señales amarillas pequeñas (lejanas) se pueden perder en la máscara
    # combinada si la apertura morfológica las elimina. Se buscan por separado.
    y_cnts, _ = cv2.findContours(yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in y_cnts:
        area = cv2.contourArea(cnt)
        if area < 18 or area > 600:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        if 0.4 <= asp <= 1.8:
            pad = 5
            new_bbox = (max(0, x + x_min - pad),
                        max(0, y + y_min - pad),
                        min(w, x + x_min + cw + pad),
                        min(y_max, y + y_min + ch + pad))
            if not any(abs(b[0]-new_bbox[0]) < 8 and abs(b[1]-new_bbox[1]) < 8 for b in bboxes):
                bboxes.append(new_bbox)

    # ── Detección blanca independiente ───────────────────────────────────────
    # PROBLEMA HISTÓRICO: el guardarriel (área ~1900px) se fusionaba con la señal
    # de velocidad (área ~300-600px) en la máscara combinada. Al buscarlos por
    # separado y filtrar por área, se puede aislar solo la señal.
    w_cnts, _ = cv2.findContours(white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in w_cnts:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 900:  # <60=ruido/poste; >900=guardarriel
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        if 0.2 <= asp <= 1.4:
            pad = 4
            new_bbox = (max(0, x + x_min - pad),
                        max(0, y + y_min - pad),
                        min(w, x + x_min + cw + pad),
                        min(y_max, y + y_min + ch + pad))
            if not any(abs(b[0]-new_bbox[0]) < 8 and abs(b[1]-new_bbox[1]) < 8 for b in bboxes):
                bboxes.append(new_bbox)
                print(f"[WHT+] area={int(area)} asp={asp:.2f}")

    return bboxes


# =============================================================================
# CLASIFICACIÓN — COLOR DOMINANTE Y CNN
# =============================================================================

# Mapeo de color detectado → clases GTSRB válidas para ese color.
# Evita que una señal amarilla se confunda con una señal blanca y viceversa.
COLOR_CLASSES = {
    'yellow': {11, 19, 20, 22},  # señales de precaución: cruce, curva izq/der, bache
    'red':    {13, 14, 15, 17},  # señales regulatorias: stop, ceder, no-peatones, no-girar
    'white':  {3, 4, 34},        # velocidad 55/65 mph, un sentido
}


def color_region(bgr_crop):
    """
    Determina el color dominante de un crop de señal.
    Devuelve 'yellow', 'red', 'white' u 'other'.

    Los umbrales de fracción (y>0.02, r>0.04, w>0.10) se calibraron
    empiricamente para el nivel de saturación de Webots.
    """
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    n   = hsv.shape[0] * hsv.shape[1]
    y   = cv2.inRange(hsv, (15, 15, 30),  (40,  255, 255)).sum() / 255 / n
    r1  = cv2.inRange(hsv, (0,  35, 45),  (12,  255, 255)).sum() / 255 / n
    r2  = cv2.inRange(hsv, (155,35, 45),  (180, 255, 255)).sum() / 255 / n
    w   = cv2.inRange(hsv, (0,  0,  100), (180, 150, 255)).sum() / 255 / n
    if y  > 0.02: return 'yellow'
    if r1 + r2 > 0.04: return 'red'
    if w  > 0.10: return 'white'
    print(f"    [OTHER] y={y:.3f} r={r1+r2:.3f} w={w:.3f}  sz={bgr_crop.shape[:2]}")
    return 'other'


def clasificar(bgr_img, bbox, model, idx_to_gtsrb):
    """
    Clasifica una región candidata con la CNN.

    PIPELINE POR COLOR:

    BLANCAS (señales de velocidad y Un Sentido):
      1. Filtro de tamaño: regiones >20% del área total son edificios/paredes
      2. Filtro de textura: varianza Laplaciana <15 → guardarriel/pared uniforme
      3. Filtro de aspect ratio: portrait (asp<1.1) → 55/65 mph
                                 landscape (asp>1.4) → Un Sentido
      4. CNN decide entre candidatos filtrados

    AMARILLAS y ROJAS:
      1. Filtro anti-desierto: el desierto de Webots tiene pixeles amarillos pero
         con saturación baja (S_med < 22). Las señales reales tienen S_med > 22.
      2. CLAHE: normaliza el contraste del crop antes de pasarlo a la CNN.
         CRÍTICO para cerrar el domain gap (crops oscuros en Webots vs
         imágenes brillantes en el dataset de entrenamiento).
      3. CNN con máscara de clases: solo se consideran clases del color correcto

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
      Redistribuye el histograma de luminosidad en bloques locales (4×4 tiles).
      Esto hace que un crop oscuro de Webots se parezca más a las imágenes
      bien iluminadas con las que fue entrenada la CNN.
    """
    x1, y1, x2, y2 = bbox
    crop = bgr_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    color = color_region(crop)
    bh, bw = crop.shape[:2]

    # ── Señales BLANCAS ───────────────────────────────────────────────────────
    if color == 'white':
        asp = bw / float(bh) if bh > 0 else 1.0

        if bh * bw > int(bgr_img.shape[0] * bgr_img.shape[1] * 0.20):
            return None, 0.0   # región demasiado grande → edificio

        gray_c  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_c, cv2.CV_64F).var()
        print(f"    [BLANCA] asp={asp:.2f}  lap={lap_var:.0f}  thresh_lap=15")
        if lap_var < 15:
            return None, 0.0   # guardarriel/pared/cielo → textura uniforme

        if asp < 1.1:
            candidates = [k for k, v in idx_to_gtsrb.items() if v in {3, 4}]
        elif asp > 1.4:
            candidates = [k for k, v in idx_to_gtsrb.items() if v in {34}]
        else:
            candidates = [k for k, v in idx_to_gtsrb.items() if v in {3, 4, 34}]

        if not candidates:
            return None, 0.0

        img_norm = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        probs = model.predict(np.expand_dims(img_norm, 0), verbose=0)[0]
        best  = max(candidates, key=lambda i: probs[i])
        cid   = idx_to_gtsrb[best]
        raw_conf = float(probs[best])
        print(f"           gtsrb={cid}  cnn_conf={raw_conf:.2f}")
        if raw_conf < 0.10:
            return None, 0.0
        return cid, 0.80

    if color == 'other':
        return None, 0.0

    # ── Filtro anti-desierto (solo amarillo) ──────────────────────────────────
    if color == 'yellow':
        hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yel_mask = cv2.inRange(hsv_c, (15, 10, 30), (45, 255, 255))
        yel_count = cv2.countNonZero(yel_mask)
        if yel_count > 0:
            s_vals = hsv_c[:, :, 1][yel_mask > 0]
            s_med  = int(np.median(s_vals))
            if s_med < 22:
                print(f"    [SKIP] crop amarillo descartado S_med={s_med}<22 (desierto)")
                return None, 0.0
        else:
            return None, 0.0

    # ── CLAHE + CNN (amarillo y rojo) ─────────────────────────────────────────
    valid_gtsrb = COLOR_CLASSES[color]

    # CLAHE normaliza el contraste localmente (4×4 bloques, clipLimit=2.5)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    lab     = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    crop_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img_norm = cv2.resize(
        cv2.cvtColor(crop_eq, cv2.COLOR_BGR2RGB),
        (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    probs = model.predict(np.expand_dims(img_norm, 0), verbose=0)[0]

    # Poner a cero las clases que no corresponden al color detectado
    probs_f = probs.copy()
    for local_i, gtsrb_c in idx_to_gtsrb.items():
        if gtsrb_c not in valid_gtsrb:
            probs_f[local_i] = 0.0

    local_id = int(np.argmax(probs_f))
    cid      = idx_to_gtsrb.get(local_id, local_id)
    conf     = float(probs_f[local_id])
    label    = CLASS_NAMES.get(cid, f"c{cid}")

    top2 = sorted([(idx_to_gtsrb.get(i, i), float(probs_f[i]))
                   for i in range(len(probs_f)) if probs_f[i] > 0.01],
                  key=lambda x: -x[1])[:2]
    top2_str = "  ".join(f"g{g}={v:.2f}" for g, v in top2)
    print(f"    [{color.upper()}] {top2_str}  -> {label}  thresh=0.25")

    if conf < 0.25:
        return None, 0.0
    return cid, conf


def detectar_senales(bgr_img, model, idx_to_gtsrb):
    """
    Pipeline completo: HSV → candidatos → CNN → mejor detección.
    También guarda crops en disco para diagnóstico visual.
    Devuelve (mejor_deteccion, bboxes_candidatos).
    """
    global _crop_n
    if model is None:
        return None, []
    bboxes = detectar_regiones(bgr_img)
    if bboxes:
        print(f"[CNN] {len(bboxes)} candidato(s)")
    mejor = None
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        crop = bgr_img[y1:y2, x1:x2]
        if crop.size > 0 and _crop_n < 200:
            col   = color_region(crop)
            fname = os.path.join(CROP_DIR, f"{_crop_n:04d}_col{col}_b{i}.png")
            cv2.imwrite(fname, crop)
            _crop_n += 1
        cid, conf = clasificar(bgr_img, bbox, model, idx_to_gtsrb)
        if cid is None:
            continue
        label = CLASS_NAMES.get(cid, f"Clase {cid}")
        print(f"  ✓ {label}  {conf:.2f}")
        if mejor is None or conf > mejor["confidence"]:
            mejor = {"class_id": cid, "label": label, "confidence": conf}
    return mejor, bboxes


# =============================================================================
# LIDAR — SEGUIMIENTO DE PARED
# =============================================================================

def lidar_steering(ranges):
    """
    Calcula el ángulo de dirección para mantener distancia al bordillo derecho.

    El sensor Sick LMS 291 entrega 181 distancias en 180°.
    Con el carro orientado al oeste:
      - Índices 135-165 apuntan a la pared DERECHA (norte)
      - Índices 105-135 apuntan al frente-derecha

    Si la pared derecha desaparece (cruce, rampa), el main loop intenta
    usar la pared izquierda o mantiene el último ángulo con decaimiento.
    """
    n = len(ranges)
    right_vals  = ranges[n * 135 // 180 : n * 165 // 180]
    fr_vals     = ranges[n * 105 // 180 : n * 135 // 180]
    right       = min(right_vals) if right_vals else 99.0
    front_right = min(fr_vals)    if fr_vals    else 99.0
    hay_pared   = right < WALL_NO_WALL

    if hay_pared:
        error = right - WALL_TARGET     # positivo = muy lejos → girar derecha
        steer = KP_WALL * error         # P puro: simple y estable a 10km/h
    else:
        steer = 0.0  # señal "sin pared" — el main loop decide qué hacer

    return float(np.clip(steer, -MAX_ANGLE, MAX_ANGLE)), right, front_right, hay_pared


# =============================================================================
# MAIN — LOOP PRINCIPAL
# =============================================================================

def main():
    # ── Estado de control ────────────────────────────────────────────────────
    target_speed     = DEFAULT_SPEED
    speed            = DEFAULT_SPEED
    steering         = 0.0
    frame_count      = 0
    manual_countdown = 0    # frames restantes de override manual
    blend_countdown  = 0    # frames de transición suave manual→LiDAR
    BLEND_TICKS      = 20
    last_wall_steer  = 0.0
    no_wall_frames   = 0

    last_steer_time      = 0.0
    lidar_enabled        = True
    ultima_deteccion     = None
    frames_sin_deteccion = 0
    senales_detectadas   = set()   # conjunto de IDs únicos confirmados
    pending_class        = None    # clase esperando confirmación
    last_bboxes          = []
    pending_streak       = 0       # frames consecutivos con la misma clase

    # ── Inicializar robot y periféricos ─────────────────────────────────────
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img  = robot.getDevice("display_image")   # cámara + ROI
    display_img2 = robot.getDevice("display_image2")  # panel de detección

    keyboard = Keyboard()
    keyboard.enable(timestep)

    sick = robot.getDevice("Sick LMS 291")
    sick.enable(timestep)

    model_cnn, idx_to_gtsrb = cargar_modelo(MODEL_PATH)

    print("=" * 55)
    print("  MODO AUTOMÁTICO (LiDAR) — pared derecha")
    print("  ↑↓  velocidad    ESPACIO stop")
    print("  ←→  override manual (vuelve a LiDAR solo)")
    print("  A   capturar foto    L  toggle LiDAR")
    print("=" * 55)

    while robot.step() != -1:
        frame_count += 1

        # ── Imagen de cámara ─────────────────────────────────────────────────
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # ── LiDAR: dirección y velocidad ─────────────────────────────────────
        ranges = sick.getRangeImage()
        if ranges:
            n = len(ranges)

            # Obstáculo frontal: índices 70-110 (±20° del frente)
            front_vals = ranges[n*70//180 : n*110//180]
            frente     = min(front_vals) if front_vals else 99.0

            if frente < FRONT_STOP:
                speed = 0.0
                print(f"[LIDAR] Obstáculo a {frente:.1f}m — STOP")
            elif frente < FRONT_SLOW:
                speed = 6.0
            else:
                speed = target_speed

            # Dirección automática (solo si LiDAR activo y sin override manual)
            if lidar_enabled and manual_countdown <= 0:
                auto_steer, dist_right, dist_fr, hay_pared = lidar_steering(ranges)

                if hay_pared:
                    target_steer    = auto_steer
                    last_wall_steer = auto_steer
                    no_wall_frames  = 0
                else:
                    # Intentar pared izquierda como fallback (índices 15-45)
                    left_vals = ranges[n * 15 // 180 : n * 45 // 180]
                    dist_left = min(left_vals) if left_vals else 99.0
                    if dist_left < WALL_NO_WALL:
                        error_left   = WALL_TARGET - dist_left
                        target_steer = float(np.clip(-KP_WALL * error_left, -MAX_ANGLE, MAX_ANGLE))
                        last_wall_steer = target_steer
                        no_wall_frames  = 0
                    else:
                        # Sin ninguna pared: hold → decay para no girar en seco
                        no_wall_frames += 1
                        if no_wall_frames <= CROSS_HOLD:
                            target_steer = last_wall_steer
                        else:
                            last_wall_steer *= CROSS_DECAY
                            target_steer     = last_wall_steer

                # Blend suave al volver de override manual
                if blend_countdown > 0:
                    alpha    = 1.0 - blend_countdown / BLEND_TICKS
                    steering = steering * (1.0 - alpha) + target_steer * alpha
                    blend_countdown -= 1
                else:
                    steering = target_steer
            else:
                manual_countdown -= 1

        # ── CNN (cada DETECT_EVERY frames, después de la gracia inicial) ─────
        if model_cnn is not None and frame_count > DETECT_GRACE and frame_count % DETECT_EVERY == 0:
            det, last_bboxes = detectar_senales(bgr, model_cnn, idx_to_gtsrb)
            if det is not None:
                cid = det["class_id"]
                if cid == pending_class:
                    pending_streak += 1
                else:
                    pending_class  = cid
                    pending_streak = 1

                # Confirmar solo si se vio CONFIRM_STREAK veces seguidas
                # Reduce falsos positivos de un solo frame
                if pending_streak >= CONFIRM_STREAK:
                    es_nueva = cid not in senales_detectadas
                    senales_detectadas.add(cid)
                    ultima_deteccion     = det
                    frames_sin_deteccion = 0
                    if es_nueva:
                        print(f"[Señal CONFIRMADA] {det['label']}  conf={det['confidence']:.2f}"
                              f"  ({len(senales_detectadas)}/16 únicas)")
                else:
                    print(f"[Señal pendiente {pending_streak}/{CONFIRM_STREAK}]"
                          f" {det['label']}  conf={det['confidence']:.2f}")
            else:
                pending_class  = None
                pending_streak = 0
                frames_sin_deteccion += 1
                if frames_sin_deteccion > 50:
                    ultima_deteccion = None

        # ── Displays ─────────────────────────────────────────────────────────
        render_roi_overlay(display_img, display_img2,
                           bgr, last_bboxes,
                           ultima_deteccion, len(senales_detectadas))

        # Guardar frame anotado cada 30 frames para debug
        if frame_count % 30 == 0:
            dw = display_img.getWidth()
            dh = display_img.getHeight()
            vis = cv2.resize(bgr, (dw, dh))
            rx  = int(dw * 0.25)
            cv2.rectangle(vis, (rx, int(dh*0.05)), (dw-1, int(dh*0.85)), (0, 255, 255), 2)
            sx = dw / bgr.shape[1]; sy = dh / bgr.shape[0]
            for x1, y1, x2, y2 in last_bboxes:
                cv2.rectangle(vis, (int(x1*sx), int(y1*sy)),
                              (int(x2*sx), int(y2*sy)), (255, 0, 255), 2)
            cv2.imwrite(os.path.join(SCRIPT_DIR, "debug_roi.png"), vis)

        # ── Teclado ───────────────────────────────────────────────────────────
        key = keyboard.getKey()
        if key != -1:
            if key == ord('L'):
                lidar_enabled = not lidar_enabled
                print(f"[Modo] {'MANUAL' if not lidar_enabled else 'LIDAR'}  (L para cambiar)")
            elif key == ord(' '):
                speed = 0.0
                target_speed = 0.0
            elif key == ord('A'):
                ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path = os.path.join(SCRIPT_DIR, f"foto_{ts}.png")
                camera.saveImage(path, 1)
                print(f"[Foto] {path}")
            elif key == Keyboard.UP:
                target_speed = min(target_speed + SPEED_INCR, 80.0)
                speed = target_speed
            elif key == Keyboard.DOWN:
                target_speed = max(target_speed - SPEED_INCR, 0.0)
                speed = target_speed
            elif key in (Keyboard.LEFT, Keyboard.RIGHT):
                # Debounce: evita que la tecla mantenida presionada dispare
                # demasiado rápido y haga el volante incontrolable
                now = time.time()
                if now - last_steer_time >= STEER_DEBOUNCE:
                    last_steer_time = now
                    if key == Keyboard.LEFT:
                        steering = max(steering - ANGLE_INCR, -MAX_ANGLE)
                    else:
                        steering = min(steering + ANGLE_INCR, MAX_ANGLE)
                    if manual_countdown <= 0:
                        blend_countdown = BLEND_TICKS
                    manual_countdown = MANUAL_TICKS

        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
