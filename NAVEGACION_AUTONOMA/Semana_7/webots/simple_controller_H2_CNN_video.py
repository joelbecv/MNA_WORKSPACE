# Actividad 4.2 — Detección de señales de tráfico USA con CNN en Webots
# MNA Tec de Monterrey — Joel Arturo Becerril Balderas  A01797427

# Importamos las librerías de Webots para controlar el carro y el display,
# OpenCV para procesar imágenes y TensorFlow para correr la red neuronal.
from controller import Display, Keyboard, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os, time, json, shutil

# Al arrancar borramos la carpeta de crops del run anterior y la recreamos.
# Así solo guardamos los recortes de la corrida actual para diagnóstico.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_DIR   = os.path.join(SCRIPT_DIR, "debug_crops")
if os.path.exists(CROP_DIR):
    shutil.rmtree(CROP_DIR)
os.makedirs(CROP_DIR)
_crop_n = 0
_mask_n = 0

try:
    import tensorflow as tf
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False


# =============================================================================
# PARÁMETROS DE CONDUCCIÓN
# Aquí concentramos todos los valores ajustables del sistema.
# Separar parámetros del código es buena práctica: si algo falla,
# solo cambiamos un número aquí en vez de buscar en todo el código.
# =============================================================================
MAX_ANGLE      = 0.5
SPEED_INCR     = 2.0
ANGLE_INCR     = 0.012
STEER_DEBOUNCE = 0.07
DEFAULT_SPEED  = 10.0

IMG_SIZE       = 32
CONFIRM_STREAK = 2
DETECT_EVERY   = 3

# El LiDAR mide distancias en metros. WALL_TARGET es cuánto queremos
# mantenernos del bordillo. KP_WALL es la ganancia del controlador P:
# qué tan fuerte reacciona el volante al error de distancia.
WALL_TARGET  = 2.5
WALL_NO_WALL = 8.0
KP_WALL      = 0.15
FRONT_STOP   = 0.9
FRONT_SLOW   = 1.8
MANUAL_TICKS = 120
DETECT_GRACE = 60
CROSS_HOLD   = 20
CROSS_DECAY  = 0.90

_US_MODEL  = os.path.join(os.path.dirname(__file__), "..", "modelo_us_webots.keras")
_OLD_MODEL = os.path.join(os.path.dirname(__file__), "..", "modelo_gtsrb.keras")
MODEL_PATH = _US_MODEL if os.path.exists(_US_MODEL) else _OLD_MODEL
MAP_PATH   = os.path.join(os.path.dirname(__file__), "..", "us_class_map.json")

# Diccionario que convierte el ID numérico de clase en un nombre legible.
# La CNN solo devuelve un número (ej: 14). Este dict lo traduce a "STOP".
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
# CÁMARA Y DISPLAY
# Lee la imagen cruda de la cámara Webots y la convierte a un array NumPy
# en formato BGR que OpenCV puede procesar directamente.
# =============================================================================
def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))


# Dibuja la imagen de cámara con el recuadro ROI (amarillo) y los candidatos
# detectados (magenta) en el display izquierdo. En el display derecho muestra
# el nombre de la señal, confianza y el contador de señales únicas detectadas.
def render_roi_overlay(display_cam, display_info, bgr_img, bboxes, deteccion, total):
    dw = display_cam.getWidth()
    dh = display_cam.getHeight()
    vis = cv2.resize(bgr_img, (dw, dh))
    rx, ry_top, ry_bot = int(dw*0.25), int(dh*0.05), int(dh*0.70)
    cv2.rectangle(vis, (rx, ry_top), (dw-1, ry_bot), (0, 255, 255), 3)
    cv2.putText(vis, "ROI", (rx+2, ry_top+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
    sx = dw / bgr_img.shape[1]; sy = dh / bgr_img.shape[0]
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(vis, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), (255,0,255), 2)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    ir = display_cam.imageNew(vis_rgb.tobytes(), Display.RGB, width=dw, height=dh)
    display_cam.imagePaste(ir, 0, 0, False)

    dw2, dh2 = display_info.getWidth(), display_info.getHeight()
    display_info.setColor(0x0A0A1A); display_info.fillRectangle(0, 0, dw2, dh2)
    display_info.setColor(0x2255AA); display_info.fillRectangle(0, 0, dw2, 16)
    display_info.setColor(0xFFFFFF); display_info.drawText("CNN US Signs  Act. 4.2", 4, 4)
    display_info.setColor(0x111111); display_info.fillRectangle(0, 20, dw2, 24)
    display_info.setColor(0x00FF88 if total >= 8 else 0xFFEE00)
    display_info.drawText(f"Senales unicas: {total} / 16", 4, 28)
    if deteccion is not None:
        pct = int(deteccion["confidence"] * 100)
        display_info.setColor(0x004400); display_info.fillRectangle(0, 50, dw2, dh2-50)
        display_info.setColor(0x00FF44); display_info.drawText("DETECTADA:", 4, 56)
        display_info.setColor(0xFFFFFF)
        display_info.drawText(deteccion["label"], 4, 72)
        display_info.drawText(f"Conf: {pct}%", 4, 88)
        display_info.setColor(0x006600); display_info.fillRectangle(4, 100, 150, 10)
        display_info.setColor(0x00FF44); display_info.fillRectangle(4, 100, int(150*pct/100), 10)
    else:
        display_info.setColor(0x555566); display_info.drawText("(buscando senales...)", 4, 72)


# =============================================================================
# CARGA DEL MODELO CNN
# Cargamos el modelo Keras desde disco. El archivo JSON convierte el índice
# de salida del modelo (0 a 10) al ID de clase del dataset original.
# Si el modelo no existe, el sistema sigue funcionando solo con LiDAR.
# =============================================================================
def cargar_modelo(path):
    if not CNN_AVAILABLE:
        return None, {}
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        return None, {}
    try:
        model = tf.keras.models.load_model(abs_path)
        print(f"[CNN] Modelo cargado: {abs_path}")
    except Exception as e:
        print(f"[CNN] Error: {e}"); return None, {}
    idx_to_gtsrb = {}
    map_abs = os.path.abspath(MAP_PATH)
    if os.path.exists(map_abs):
        with open(map_abs) as f:
            idx_to_gtsrb = {int(k): int(v) for k, v in json.load(f).items()}
    return model, idx_to_gtsrb


# =============================================================================
# DETECCIÓN HSV — BÚSQUEDA DE REGIONES CANDIDATAS
# Buscamos en la imagen regiones con colores de señal (rojo, amarillo, blanco).
# El espacio HSV separa el tono de la intensidad, lo que hace el filtrado
# más robusto a cambios de iluminación que trabajar directo en RGB.
# =============================================================================
def detectar_regiones(bgr_img):
    global _mask_n
    h, w = bgr_img.shape[:2]

    # Solo miramos el 75% derecho de la imagen y entre 5% y 70% de altura.
    # Ahí aparecen las señales. El cielo y el asfalto quedan fuera del ROI.
    y_min = int(h * 0.05); y_max = int(h * 0.70); x_min = int(w * 0.25)
    roi = bgr_img[y_min:y_max, x_min:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Rojo aparece en dos rangos HSV porque el tono es circular (0-12° y 155-180°).
    # Bajamos la saturación mínima a 35 porque Webots desatura los colores.
    red1 = cv2.inRange(hsv, np.array([0,   35, 45]), np.array([12,  255, 255]))
    red2 = cv2.inRange(hsv, np.array([155, 35, 45]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([95,  50, 50]), np.array([135, 255, 255]))

    # Amarillo usa tres rangos para cubrir sombra, luz directa y sobreexposición.
    yel1 = cv2.inRange(hsv, np.array([15, 15, 40]),  np.array([45, 255, 255]))
    yel2 = cv2.inRange(hsv, np.array([18, 10, 25]),  np.array([40, 180, 120]))
    yel3 = cv2.inRange(hsv, np.array([20,  8,150]),  np.array([50,  60, 255]))
    yellow = yel1 | yel2 | yel3

    # Blanco: baja saturación y alto valor. Detecta señales de velocidad y Un Sentido.
    white = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 130, 255]))

    # Cortamos la parte inferior del ROI para ignorar marcas blancas del asfalto
    # y reflejos amarillos de las líneas del carril que confundirían el detector.
    white[int(roi.shape[0] * 0.90):, :] = 0
    yellow[int(roi.shape[0] * 0.75):, :] = 0

    mask = red1 | red2 | blue | yellow | white
    k    = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _mask_n += 1
    if _mask_n % 5 == 0:
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_roi_color.png"), roi)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_white.png"),  white)
        cv2.imwrite(os.path.join(SCRIPT_DIR, "dbg_yellow.png"), yellow)
        any_hue_yellow = cv2.inRange(hsv, np.array([10,0,0]), np.array([50,255,255]))
        any_px = cv2.countNonZero(any_hue_yellow)
        if any_px > 0:
            ys = np.where(any_hue_yellow > 0)
            sv = hsv[ys[0], ys[1], 1]; vv = hsv[ys[0], ys[1], 2]
            print(f"[YEL] px={any_px} S={sv.min()}-{sv.max()}(med={int(np.median(sv))}) "
                  f"V={vv.min()}-{vv.max()}(med={int(np.median(vv))}) filtered={cv2.countNonZero(yellow)}")
        white_cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w_areas = sorted([int(cv2.contourArea(c)) for c in white_cnts if cv2.contourArea(c)>=5], reverse=True)[:5]
        print(f"[WHT] px={cv2.countNonZero(white)} cnts={len(w_areas)} top={w_areas}")

    # Filtramos contornos por área mínima y aspect ratio.
    # Las señales son aproximadamente cuadradas (asp 0.4–2.0).
    # Marcas de piso y desierto suelen ser muy anchas (asp > 2).
    bboxes = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 25: continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        if 0.4 <= asp <= 2.0:
            pad = 4
            bboxes.append((max(0, x+x_min-pad), max(0, y+y_min-pad),
                           min(w, x+x_min+cw+pad), min(y_max, y+y_min+ch+pad)))

    # Buscamos contornos amarillos por separado para no perder señales pequeñas
    # que la operación morfológica combinada podría eliminar.
    y_cnts, _ = cv2.findContours(yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in y_cnts:
        area = cv2.contourArea(cnt)
        if area < 18 or area > 600: continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        if 0.4 <= asp <= 1.8:
            pad = 5
            nb = (max(0,x+x_min-pad), max(0,y+y_min-pad),
                  min(w,x+x_min+cw+pad), min(y_max,y+y_min+ch+pad))
            if not any(abs(b[0]-nb[0])<8 and abs(b[1]-nb[1])<8 for b in bboxes):
                bboxes.append(nb)

    # Para blanco usamos área 60–900. El guardarriel tiene >1600 px y queda fuera.
    # Esto resolvió el problema donde la señal de 55 mph no se detectaba porque
    # se fusionaba con el guardarriel en un único contorno enorme.
    w_cnts, _ = cv2.findContours(white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in w_cnts:
        area = cv2.contourArea(cnt)
        if area < 60 or area > 900: continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        asp = cw / float(ch) if ch > 0 else 0
        if 0.2 <= asp <= 1.4:
            pad = 4
            nb = (max(0,x+x_min-pad), max(0,y+y_min-pad),
                  min(w,x+x_min+cw+pad), min(y_max,y+y_min+ch+pad))
            if not any(abs(b[0]-nb[0])<8 and abs(b[1]-nb[1])<8 for b in bboxes):
                bboxes.append(nb)
                print(f"[WHT+] area={int(area)} asp={asp:.2f}")
    return bboxes


# =============================================================================
# COLOR DOMINANTE DEL CROP
# Antes de pasar el recorte a la CNN, determinamos su color dominante.
# Esto nos permite limitar la CNN solo a las clases del color correcto:
# un recorte amarillo nunca puede ser una señal de velocidad blanca.
# =============================================================================
COLOR_CLASSES = {
    'yellow': {11, 19, 20, 22},
    'red':    {13, 14, 15, 17},
    'white':  {3, 4, 34},
}

def color_region(bgr_crop):
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    n   = hsv.shape[0] * hsv.shape[1]
    y   = cv2.inRange(hsv, (15,15,30),  (40, 255,255)).sum()/255/n
    r1  = cv2.inRange(hsv, (0, 35,45),  (12, 255,255)).sum()/255/n
    r2  = cv2.inRange(hsv, (155,35,45), (180,255,255)).sum()/255/n
    w   = cv2.inRange(hsv, (0,  0,100), (180,150,255)).sum()/255/n
    if y > 0.02: return 'yellow'
    if r1+r2 > 0.04: return 'red'
    if w > 0.10: return 'white'
    print(f"    [OTHER] y={y:.3f} r={r1+r2:.3f} w={w:.3f}  sz={bgr_crop.shape[:2]}")
    return 'other'


# =============================================================================
# CLASIFICACIÓN CON CNN
# Aquí ocurre la detección real. Para señales blancas aplicamos filtros de
# tamaño y textura antes de la CNN. Para amarillas y rojas aplicamos CLAHE,
# que normaliza el contraste y cierra el domain gap entre el dataset y Webots.
# =============================================================================
def clasificar(bgr_img, bbox, model, idx_to_gtsrb):
    x1, y1, x2, y2 = bbox
    crop = bgr_img[y1:y2, x1:x2]
    if crop.size == 0: return None, 0.0

    color = color_region(crop)
    bh, bw = crop.shape[:2]

    if color == 'white':
        asp = bw / float(bh) if bh > 0 else 1.0
        # Regiones mayores al 20% del frame son paredes o edificios, no señales.
        if bh * bw > int(bgr_img.shape[0] * bgr_img.shape[1] * 0.20):
            return None, 0.0
        # La varianza Laplaciana mide textura. Un guardarriel es uniforme (var~0).
        # Una señal con texto o número tiene varianza alta (>15).
        gray_c  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray_c, cv2.CV_64F).var()
        print(f"    [BLANCA] asp={asp:.2f}  lap={lap_var:.0f}  thresh_lap=15")
        if lap_var < 15: return None, 0.0
        if asp < 1.1:
            candidates = [k for k,v in idx_to_gtsrb.items() if v in {3,4}]
        elif asp > 1.4:
            candidates = [k for k,v in idx_to_gtsrb.items() if v in {34}]
        else:
            candidates = [k for k,v in idx_to_gtsrb.items() if v in {3,4,34}]
        if not candidates: return None, 0.0
        img_norm = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                              (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
        probs    = model.predict(np.expand_dims(img_norm,0), verbose=0)[0]
        best     = max(candidates, key=lambda i: probs[i])
        cid      = idx_to_gtsrb[best]
        raw_conf = float(probs[best])
        print(f"           gtsrb={cid}  cnn_conf={raw_conf:.2f}")
        if raw_conf < 0.10: return None, 0.0
        return cid, 0.80

    if color == 'other': return None, 0.0

    # El desierto de Webots tiene píxeles amarillos con saturación muy baja.
    # Si la saturación mediana del crop es menor a 22, es desierto, no señal.
    if color == 'yellow':
        hsv_c    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        yel_mask = cv2.inRange(hsv_c, (15,10,30), (45,255,255))
        yel_count = cv2.countNonZero(yel_mask)
        if yel_count > 0:
            s_med = int(np.median(hsv_c[:,:,1][yel_mask > 0]))
            if s_med < 22:
                print(f"    [SKIP] S_med={s_med}<22 (desierto)")
                return None, 0.0
        else:
            return None, 0.0

    # CLAHE normaliza el contraste antes de pasar el crop a la CNN.
    # El dataset de entrenamiento tiene brillo ~210/255; los crops de Webots ~99/255.
    # Sin esta corrección, la red ve señales que no reconoce y falla.
    valid_gtsrb = COLOR_CLASSES[color]
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
    lab     = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    crop_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img_norm = cv2.resize(cv2.cvtColor(crop_eq, cv2.COLOR_BGR2RGB),
                          (IMG_SIZE, IMG_SIZE)).astype(np.float32)/255.0
    probs = model.predict(np.expand_dims(img_norm,0), verbose=0)[0]

    # Ponemos a cero las clases que no corresponden al color del recorte.
    # Luego tomamos el índice con mayor probabilidad como la señal detectada.
    probs_f = probs.copy()
    for local_i, gtsrb_c in idx_to_gtsrb.items():
        if gtsrb_c not in valid_gtsrb: probs_f[local_i] = 0.0
    local_id = int(np.argmax(probs_f))
    cid      = idx_to_gtsrb.get(local_id, local_id)
    conf     = float(probs_f[local_id])
    label    = CLASS_NAMES.get(cid, f"c{cid}")
    top2 = sorted([(idx_to_gtsrb.get(i,i), float(probs_f[i]))
                   for i in range(len(probs_f)) if probs_f[i]>0.01],
                  key=lambda x:-x[1])[:2]
    print(f"    [{color.upper()}] {'  '.join(f'g{g}={v:.2f}' for g,v in top2)} -> {label}")
    if conf < 0.25: return None, 0.0
    return cid, conf


# Pipeline completo por frame: obtiene candidatos HSV, guarda crops para
# diagnóstico, pasa cada uno a clasificar() y devuelve la mejor detección.
def detectar_senales(bgr_img, model, idx_to_gtsrb):
    global _crop_n
    if model is None: return None, []
    bboxes = detectar_regiones(bgr_img)
    if bboxes: print(f"[CNN] {len(bboxes)} candidato(s)")
    mejor = None
    for i, bbox in enumerate(bboxes):
        x1,y1,x2,y2 = bbox
        crop = bgr_img[y1:y2, x1:x2]
        if crop.size > 0 and _crop_n < 200:
            col = color_region(crop)
            cv2.imwrite(os.path.join(CROP_DIR, f"{_crop_n:04d}_col{col}_b{i}.png"), crop)
            _crop_n += 1
        cid, conf = clasificar(bgr_img, bbox, model, idx_to_gtsrb)
        if cid is None: continue
        label = CLASS_NAMES.get(cid, f"Clase {cid}")
        print(f"  ✓ {label}  {conf:.2f}")
        if mejor is None or conf > mejor["confidence"]:
            mejor = {"class_id": cid, "label": label, "confidence": conf}
    return mejor, bboxes


# =============================================================================
# LIDAR — SEGUIMIENTO DE PARED
# El LiDAR devuelve 181 distancias en 180°. Tomamos los índices 135–165
# que apuntan a la pared derecha. El controlador P calcula el ángulo de giro
# proporcional al error entre la distancia actual y los 2.5 m objetivo.
# =============================================================================
def lidar_steering(ranges):
    n = len(ranges)
    right_vals = ranges[n*135//180 : n*165//180]
    fr_vals    = ranges[n*105//180 : n*135//180]
    right      = min(right_vals) if right_vals else 99.0
    front_right= min(fr_vals)    if fr_vals    else 99.0
    hay_pared  = right < WALL_NO_WALL
    steer = KP_WALL * (right - WALL_TARGET) if hay_pared else 0.0
    return float(np.clip(steer, -MAX_ANGLE, MAX_ANGLE)), right, front_right, hay_pared


# =============================================================================
# LOOP PRINCIPAL
# Cada frame: capturamos la imagen, leemos el LiDAR para conducir,
# corremos la CNN cada 3 frames para detectar señales, y actualizamos
# los displays. El teclado permite override manual temporal.
# =============================================================================
def main():
    target_speed=DEFAULT_SPEED; speed=DEFAULT_SPEED; steering=0.0
    frame_count=0; manual_countdown=0; blend_countdown=0; BLEND_TICKS=20
    last_wall_steer=0.0; no_wall_frames=0
    sin_pared_activo=False; pared_vista=False; NO_WALL_TRIGGER=8

    last_steer_time=0.0; lidar_enabled=True
    ultima_deteccion=None; frames_sin_deteccion=0
    senales_detectadas=set(); pending_class=None
    last_bboxes=[]; pending_streak=0

    robot=Car(); driver=Driver()
    timestep=int(robot.getBasicTimeStep())
    camera=robot.getDevice("camera"); camera.enable(timestep)
    display_img =robot.getDevice("display_image")
    display_img2=robot.getDevice("display_image2")
    keyboard=Keyboard(); keyboard.enable(timestep)
    sick=robot.getDevice("Sick LMS 291"); sick.enable(timestep)
    model_cnn, idx_to_gtsrb = cargar_modelo(MODEL_PATH)

    print("="*55)
    print("  MODO AUTOMÁTICO (LiDAR) — pared derecha")
    print("  ↑↓ velocidad  ←→ manual  ESPACIO stop  A foto  L lidar")
    print("="*55)

    while robot.step() != -1:
        frame_count += 1
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        ranges = sick.getRangeImage()
        if ranges:
            n = len(ranges)
            front_vals = ranges[n*70//180 : n*110//180]
            frente     = min(front_vals) if front_vals else 99.0

            # Control de velocidad: frena por obstáculo frontal o por pérdida de barra.
            if frente < FRONT_STOP: speed = 0.0
            elif frente < FRONT_SLOW: speed = 6.0
            elif sin_pared_activo: speed = 4.0
            else: speed = target_speed

            # La detección de pared corre siempre — incluso en modo manual —
            # para poder recuperar el LiDAR automáticamente cuando vuelve la barra.
            if lidar_enabled:
                auto_steer, dist_right, dist_fr, hay_pared = lidar_steering(ranges)
                left_vals = ranges[n*15//180 : n*45//180]
                dist_left = min(left_vals) if left_vals else 99.0
                hay_pared_izq = dist_left < WALL_NO_WALL

                if hay_pared or hay_pared_izq:
                    pared_vista = True

                if sin_pared_activo and (hay_pared or hay_pared_izq):
                    sin_pared_activo = False; manual_countdown = 0
                    print("[PARED] Barra recuperada — LiDAR activo")

                if not hay_pared and not hay_pared_izq:
                    no_wall_frames += 1
                    if pared_vista and no_wall_frames == NO_WALL_TRIGGER and not sin_pared_activo:
                        sin_pared_activo = True; manual_countdown = 99999; speed = 4.0
                        print("[PARED] Barra perdida — frenando, modo manual")
                else:
                    no_wall_frames = 0

            # Steering automático: solo si LiDAR activo y sin override manual.
            # Prioridad: pared derecha > pared izquierda > hold/decay.
            if lidar_enabled and manual_countdown <= 0:
                if hay_pared:
                    target_steer = auto_steer; last_wall_steer = auto_steer
                elif hay_pared_izq:
                    error_left   = WALL_TARGET - dist_left
                    target_steer = float(np.clip(-KP_WALL*error_left, -MAX_ANGLE, MAX_ANGLE))
                    last_wall_steer = target_steer
                else:
                    if no_wall_frames <= CROSS_HOLD: target_steer = last_wall_steer
                    else: last_wall_steer *= CROSS_DECAY; target_steer = last_wall_steer

                if blend_countdown > 0:
                    alpha = 1.0 - blend_countdown/BLEND_TICKS
                    steering = steering*(1.0-alpha) + target_steer*alpha
                    blend_countdown -= 1
                else:
                    steering = target_steer
            else:
                manual_countdown -= 1

        # CNN: exige CONFIRM_STREAK detecciones consecutivas de la misma señal
        # antes de confirmarla. Reduce drásticamente los falsos positivos.
        if model_cnn is not None and frame_count > DETECT_GRACE and frame_count % DETECT_EVERY == 0:
            det, last_bboxes = detectar_senales(bgr, model_cnn, idx_to_gtsrb)
            if det is not None:
                cid = det["class_id"]
                if cid == pending_class: pending_streak += 1
                else: pending_class = cid; pending_streak = 1
                if pending_streak >= CONFIRM_STREAK:
                    es_nueva = cid not in senales_detectadas
                    senales_detectadas.add(cid)
                    ultima_deteccion = det; frames_sin_deteccion = 0
                    if es_nueva:
                        print(f"[CONFIRMADA] {det['label']}  conf={det['confidence']:.2f}"
                              f"  ({len(senales_detectadas)}/16)")
                else:
                    print(f"[pendiente {pending_streak}/{CONFIRM_STREAK}] {det['label']}")
            else:
                pending_class=None; pending_streak=0; frames_sin_deteccion+=1
                if frames_sin_deteccion > 50: ultima_deteccion = None

        render_roi_overlay(display_img, display_img2, bgr, last_bboxes,
                           ultima_deteccion, len(senales_detectadas))

        if frame_count % 30 == 0:
            dw=display_img.getWidth(); dh=display_img.getHeight()
            vis=cv2.resize(bgr,(dw,dh))
            cv2.rectangle(vis,(int(dw*0.25),int(dh*0.05)),(dw-1,int(dh*0.85)),(0,255,255),2)
            sx=dw/bgr.shape[1]; sy=dh/bgr.shape[0]
            for x1,y1,x2,y2 in last_bboxes:
                cv2.rectangle(vis,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(255,0,255),2)
            cv2.imwrite(os.path.join(SCRIPT_DIR,"debug_roi.png"),vis)

        key = keyboard.getKey()
        if key != -1:
            if key == ord('L'):
                lidar_enabled = not lidar_enabled
                print(f"[Modo] {'MANUAL' if not lidar_enabled else 'LIDAR'}")
            elif key == ord(' '):
                speed=0.0; target_speed=0.0
            elif key == ord('A'):
                ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path=os.path.join(SCRIPT_DIR,f"foto_{ts}.png")
                camera.saveImage(path,1); print(f"[Foto] {path}")
            elif key == Keyboard.UP:
                target_speed=min(target_speed+SPEED_INCR,80.0); speed=target_speed
            elif key == Keyboard.DOWN:
                target_speed=max(target_speed-SPEED_INCR,0.0); speed=target_speed
            elif key in (Keyboard.LEFT, Keyboard.RIGHT):
                now=time.time()
                if now-last_steer_time >= STEER_DEBOUNCE:
                    last_steer_time=now
                    if key==Keyboard.LEFT: steering=max(steering-ANGLE_INCR,-MAX_ANGLE)
                    else: steering=min(steering+ANGLE_INCR,MAX_ANGLE)
                    if manual_countdown<=0: blend_countdown=BLEND_TICKS
                    manual_countdown=MANUAL_TICKS

        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
