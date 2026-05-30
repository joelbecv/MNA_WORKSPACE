# =============================================================================
# simple_controller_stv3.py
# Actividad 3.1 — Sistema de Detección de Peatones para Vehículo Autónomo
# Navegación Autónoma — Maestría en Inteligencia Artificial
# =============================================================================
#
# CONVENCIÓN DE COMENTARIOS
# ─────────────────────────
#   # * Narrativa del sistema  →  para leer en el video (resaltado con Better Comments)
#   # Detalle técnico          →  notas de implementación para el equipo
#   Instalar "Better Comments" en VS Code: aaron-bond.better-comments
#
# =============================================================================
# * Este controlador convierte un auto BMW simulado en un vehículo autónomo
# * que circula por una ciudad, sigue su carril y se detiene antes de atropellar
# * a nadie. El reto es moverse de forma eficiente sin comprometer la seguridad
# * de los peatones que cruzan la calle.
#
# * Para resolverlo usamos tres sistemas trabajando al mismo tiempo. El primero
# * es la navegación: el auto lee las líneas amarillas del carril con su cámara
# * y ajusta el volante constantemente para mantenerse centrado, como el control
# * de crucero de un auto real pero que además toma las curvas solo.
#
# * El segundo sistema busca personas en la imagen. Una ventana pequeña recorre
# * la cámara comparando cada zona con lo que aprendió de miles de fotos de
# * personas reales. Si encuentra una silueta humana dos veces seguidas activa
# * el freno de emergencia.
#
# * El tercer sistema es un sensor láser que barre 180 grados frente al auto.
# * No necesita reconocer qué hay en el camino, solo mide la distancia al
# * objeto más cercano. Reacciona más rápido que la cámara y funciona incluso
# * en condiciones donde la imagen no es confiable.
#
# * La lógica de seguridad combina los tres: el láser frena primero si algo
# * está muy cerca, la cámara confirma si ese algo es una persona, y el PID
# * no interfiere con la detección porque solo analiza el color amarillo.
# =============================================================================
#
# REFERENCIAS
#   - INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
#   - Dalal & Triggs, "HOG for Human Detection", CVPR 2005
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
import threading
import queue
from datetime import datetime

_CTRL_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(_CTRL_DIR, '..', '..', 'pedestrian_svm.joblib'))

#! HUGO INICIA
# =============================================================================
# PARÁMETROS DEL SISTEMA
# =============================================================================

# ── Navegación ───────────────────────────────────────────────────────────────
CRUISE_SPEED   = 30       # * velocidad de crucero del auto en km/h
MAX_ANGLE      = 0.5      # ángulo máximo del volante (radianes ≈ 28°)
MAX_STEER_RATE = 0.03     # suavizado: máximo cambio de volante por frame
DEBOUNCE_TIME  = 0.1      # pausa mínima entre pulsaciones de teclado (s)

# * Las tres ganancias del PID controlan cómo reacciona el volante. Kp corrige
# * el error del momento, Ki elimina desviaciones que se acumulan con el tiempo
# * y Kd amortigua el temblor cuando el auto oscila. Calibrados para 30 km/h.
Kp, Ki, Kd     = 0.28, 0.01, 0.01

# * El carril tiene líneas diagonales pero las rayas de los cruces peatonales
# * son casi horizontales. Este umbral descarta todo lo demasiado plano para
# * ser una línea de carril y evita que las cebras confundan al volante.
MIN_ABS_SLOPE  = 0.5      # pendiente mínima aceptada (0=horizontal, 1=45°)

# Rango de tono HSV que corresponde al amarillo de las líneas del carril
YELLOW_LOW     = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH    = np.array([35, 255, 255], dtype=np.uint8)

# ── Sensor LiDAR Sick LMS 291 ────────────────────────────────────────────────
# * El sensor dispara 180 rayos láser en abanico de 180 grados. Solo usamos
# * los rayos del cono central de 61 grados que apuntan hacia donde el auto va,
# * lo que equivale a cubrir aproximadamente 30 grados hacia cada lado.
LIDAR_CONE_DEG    = 61.0  # ancho del cono frontal activo (grados totales)
LIDAR_MAX_M       = 8.0   # * distancia a la que el láser activa la alerta (metros)
LIDAR_EMERGENCY_M = 8.0   # freno inmediato si la cámara no ha actuado aún
LIDAR_OVERRIDE_M  = 5.0   # * a menos de 5m el láser frena sin importar nada más
LIDAR_EVERY       = 3     # leer el sensor cada 3 frames (~30ms de intervalo)
LIDAR_CONFIRM     = 1     # una sola lectura positiva ya activa la alerta
STATIC_TIMEOUT    = 250   # * frames bloqueado sin persona → obstáculo fijo (~2.5s)
LIDAR_PAUSE_F     = 180   # * frames que el auto avanza ignorando el obstáculo fijo (~1.8s)

# ── Detector de personas SVM + HOG ───────────────────────────────────────────
# * El detector analiza ventanas de 64x128 píxeles sobre la imagen de la cámara.
# * La zona de interés es pequeña, así que la ampliamos 4 veces antes de buscar
# * personas para que las figuras tengan el tamaño mínimo que el algoritmo necesita.
HOG_WIN_W     = 64        # ancho de la ventana de búsqueda (píxeles)
HOG_WIN_H     = 128       # alto de la ventana de búsqueda (píxeles)
SLIDE_STEP    = 32        # cuántos píxeles avanza la ventana en cada paso
DETECT_EVERY  = 10        # * analizar la imagen cada 10 frames (~100ms)
DISPLAY_EVERY = 3         # actualizar la pantalla cada 3 frames (~30ms)
CONFIRM_N     = 2         # * detecciones seguidas necesarias para confirmar peatón
RELEASE_N     = 4         # detecciones negativas seguidas para liberar el freno
HOLD_FRAMES   = 100       # * mínimo de frames frenado tras una detección (~1 segundo)
MIN_HITS      = 1         # ventanas positivas mínimas por pasada para contar detección
SVM_THRESHOLD = 0.25      # * score mínimo de la SVM para considerar que hay una persona
                          # El modelo INRIA da scores >1.0; en Webots da 0.06-0.70
                          # porque las imágenes simuladas se ven diferente a fotos reales


# =============================================================================
# MÓDULO 1 — NAVEGACIÓN: SEGUIMIENTO DE CARRIL CON PID
# =============================================================================
# * El auto navega mirando únicamente el color amarillo de las líneas del carril.
# * Filtrar el resto de la imagen hace que peatones, cruces o sombras no puedan
# * confundir al volante ni provocar giros incorrectos.

def get_image(camera):
    """Convierte la imagen BGRA de Webots a array NumPy BGR."""
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4))

def display_color(display, bgr_img):
    """Envía una imagen BGR al display de Webots (convierte a RGB internamente)."""
    rgb = bgr_img[:, :, ::-1].copy()
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

def apply_roi(edges, h, w):
    """
    * Recorta una zona en forma de trapecio sobre la imagen de bordes.
    * Solo conserva la franja donde aparecen las líneas del carril, desde el
    * capó del auto hasta el horizonte próximo. La forma trapezoidal imita cómo
    * se ve la carretera en perspectiva desde la cámara del vehículo.

    Vértices del trapecio (origen arriba-izquierda):
      Inf-izq=(10%, 100%)  Sup-izq=(35%, 60%)
      Sup-der=(65%, 60%)   Inf-der=(90%, 100%)
    """
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([[
        (int(w * 0.10), h),
        (int(w * 0.35), int(h * 0.60)),
        (int(w * 0.65), int(h * 0.60)),
        (int(w * 0.90), h),
    ]], dtype=np.int32), 255)
    return cv2.bitwise_and(edges, mask)

def filter_lines(lines):
    """
    * Descarta las líneas demasiado horizontales como las rayas de los cruces
    * peatonales. Solo acepta líneas con suficiente inclinación, que son las
    * que corresponden a los bordes del carril vistos en perspectiva.
    """
    if lines is None:
        return None
    ok = [l for l in lines
          if l[0][2] != l[0][0]
          and abs((l[0][3]-l[0][1])/(l[0][2]-l[0][0])) >= MIN_ABS_SLOPE]
    return np.array(ok) if ok else None

def compute_center(lines):
    """
    * Calcula dónde está el centro del carril a partir de las líneas detectadas.
    * Las líneas que inclinan hacia la izquierda son el borde izquierdo del carril
    * y las que inclinan hacia la derecha son el borde derecho. El centro es el
    * punto medio entre ambos bordes. Si solo se ve uno de los dos lados, como
    * ocurre en las curvas cerradas donde el otro borde sale del encuadre, se usa
    * ese borde como referencia parcial para no perder el control del volante.
    """
    if lines is None:
        return None
    lx, rx, ax = [], [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        mid   = (x1 + x2) / 2
        ax.append(mid)
        (lx if slope < 0 else rx).append(mid)
    if lx and rx:
        return (np.mean(lx) + np.mean(rx)) / 2.0
    return np.mean(ax) if ax else None


#! ALBERTO
# =============================================================================
# MÓDULO 2A — DETECCIÓN DE PERSONAS: SVM + HOG
# =============================================================================
# * La cámara analiza una franja horizontal de la escena entre el 59% y el 85%
# * de la altura de la imagen. Esa es la zona donde aparecen peatones a distancia
# * de frenado: por encima solo hay cielo y edificios, por debajo está el capó.
# * La franja se amplía 4 veces y una ventana recorre la imagen buscando siluetas.

def svm_detect(bgr, model):
    """
    * Busca personas en la imagen deslizando una ventana de 64x128 píxeles
    * sobre la zona de interés. Cada posición de la ventana se convierte en
    * 924 números que describen los bordes de lo que hay dentro. La SVM
    * clasifica esos números y decide si la forma se parece a una persona.

    Retorna: (detected, hits, max_score, windows, best_box)
      detected   → True si al menos una ventana superó el umbral de la SVM
      hits       → número de ventanas positivas en esta pasada
      max_score  → score más alto (útil para calibrar el umbral)
      windows    → total de posiciones evaluadas
      best_box   → coordenadas (x1,y1,x2,y2) de la ventana con mayor score
    """
    h, w      = bgr.shape[:2]

    # * Solo analizamos entre el 59% y el 85% de la altura porque es donde
    # * aparecen peatones a distancia útil. Analizar toda la imagen triplicaría
    # * el tiempo de cómputo sin añadir detecciones que ayuden a frenar a tiempo.
    y_off = int(h * 0.59)
    roi   = bgr[y_off : int(h * 0.85), :]

    total     = 0
    max_score = -999.0
    windows   = 0
    best_box  = None

    for scale in (4.0,):
        # * La franja tiene apenas unos 34 píxeles de alto pero el detector
        # * necesita mínimo 128 para describir bien una figura humana. Ampliar
        # * 4 veces da 136 píxeles, suficiente para que el algoritmo funcione.
        scaled = cv2.resize(roi, (int(w * scale), int(roi.shape[0] * scale)))
        sh, sw = scaled.shape[:2]
        if sh < HOG_WIN_H or sw < HOG_WIN_W:
            continue

        # * Solo barremos entre el 30% y el 70% del ancho de la imagen porque
        # * los peatones peligrosos están en el carril, no en las banquetas.
        # * Esto reduce a la mitad el número de posiciones a evaluar.
        x0    = int(sw * 0.30)
        x1    = int(sw * 0.70)
        ystep = max(16, HOG_WIN_H // 4)

        for y in range(0, sh - HOG_WIN_H + 1, ystep):
            for x in range(x0, min(x1, sw - HOG_WIN_W + 1), SLIDE_STEP):

                # * Si la ventana tiene más de 6% de píxeles amarillos es muy probable
                # * que sea una línea de carril, no una persona. La saltamos para
                # * reducir falsos positivos y ahorrar tiempo de cómputo.
                win_bgr = scaled[y:y + HOG_WIN_H, x:x + HOG_WIN_W]
                win_hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
                if cv2.inRange(win_hsv, YELLOW_LOW, YELLOW_HIGH).mean() > 15:
                    continue

                # * HOG divide la ventana en bloques y mide en qué dirección apuntan
                # * los bordes de la imagen. El resultado son 924 números que
                # * describen la forma de lo que hay en esa posición de la imagen.
                win  = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2GRAY)
                feat = hog(win, orientations=11, pixels_per_cell=(16, 16),
                           cells_per_block=(2, 2), transform_sqrt=False,
                           feature_vector=True)

                # * La SVM devuelve un número positivo si la forma se parece a una
                # * persona y negativo si parece fondo. El umbral de 0.25 es más
                # * bajo que en imágenes reales porque la simulación se ve diferente
                # * a las fotos con las que se entrenó el modelo.
                score   = model.decision_function([feat])[0]
                windows += 1
                if score > max_score:
                    max_score = score
                    bx1 = max(0, int(x / scale))
                    by1 = max(0, int(y / scale) + y_off)
                    bx2 = min(w, int((x + HOG_WIN_W) / scale))
                    by2 = min(h, int((y + HOG_WIN_H) / scale) + y_off)
                    best_box = (bx1, by1, bx2, by2)
                if score >= SVM_THRESHOLD:
                    total += 1

    detected = total >= MIN_HITS
    return detected, total, max_score, windows, best_box

#! RAUL
# =============================================================================
# MÓDULO 2B — DETECCIÓN DE OBSTÁCULOS: LiDAR
# =============================================================================
# * El láser no identifica qué hay en el camino, solo mide a qué distancia está.
# * Es más rápido que la cámara y detecta cualquier objeto físico. Actúa como
# * primera línea de defensa mientras la cámara confirma si ese objeto es una persona.

def lidar_read(lidar, fov_rad, n_rays):
    """
    * Lee cuál es el objeto más cercano dentro del cono frontal del auto.
    * El sensor dispara 180 rayos en 180 grados. Solo usamos los del cono
    * central de 61 grados que apuntan hacia donde el auto se dirige.
    * Los rayos que no encuentran nada se ignoran porque representan espacio libre.

    Retorna: (alert, dist_m, dist_str)
      alert    → True si hay algo a menos de LIDAR_MAX_M metros
      dist_m   → distancia al objeto más cercano en metros (None si no hay)
      dist_str → texto para mostrar en pantalla

    getRangeImage() debe llamarse dentro del step() de Webots o retorna vacío.
    """
    ranges = lidar.getRangeImage()
    if not ranges:
        return False, None, '---'

    center = n_rays // 2
    half   = max(1, int(n_rays * (math.radians(LIDAR_CONE_DEG) / fov_rad) / 2))
    cone   = [r for r in ranges[center - half : center + half]
              if not (math.isnan(r) or math.isinf(r))]
    if not cone:
        return False, None, '---'

    d = min(cone)
    return d < LIDAR_MAX_M, d, f"{d:.1f}m"


#! JOEL
# =============================================================================
# MÓDULO 3 — LOOP PRINCIPAL
# =============================================================================

def main():

    # * Cargamos el modelo de inteligencia artificial entrenado con miles de fotos
    # * de personas reales del dataset INRIA. Incluye un normalizador de datos y
    # * la SVM ya entrenada. Si no se encuentra el archivo, el auto corre solo
    # * con navegación PID y sensor láser, sin detección de personas por cámara.
    model_path = os.path.normpath(MODEL_PATH)
    svm_model  = joblib.load(model_path) if os.path.exists(model_path) else None
    print("[OK] Modelo SVM cargado" if svm_model else "[AVISO] Sin modelo — solo PID")

    # * Inicializamos el auto y sus sensores. Driver() es la interfaz principal
    # * que combina movimiento y sensores en una sola API. El timestep define
    # * cada cuánto se actualiza la simulación, en este caso cada 10 milisegundos.
    # No multiplicar el timestep: causa freeze en macOS Apple Silicon con Rosetta.
    driver   = Driver()
    timestep = int(driver.getBasicTimeStep())   # 10ms por step de simulación

    camera  = driver.getDevice("camera")
    camera.enable(timestep)

    display  = driver.getDevice("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # * El sensor láser requiere que Webots corra bajo Rosetta 2 en Mac con chip
    # * M1, M2 o M3. Sin Rosetta el sensor falla sin dar ningún mensaje de error.
    # * Para activarlo: click derecho en Webots.app, Información, Abrir con Rosetta.
    lidar   = driver.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    _fov    = lidar.getFov()                    # campo visual: 3.14 rad = 180°
    _n_rays = lidar.getHorizontalResolution()   # 180 rayos en total
    lidar.enablePointCloud()                    # muestra los rayos en la vista 3D de Webots

    _half_rays = max(1, int(_n_rays * (math.radians(LIDAR_CONE_DEG) / _fov) / 2))
    _half_deg  = round(_half_rays * math.degrees(_fov / _n_rays), 1)
    print(f"[LiDAR] FOV={_fov:.2f} rad  rayos={_n_rays}  cono=±{_half_deg}° ({2*_half_rays} rayos activos)  max={LIDAR_MAX_M}m")

    dw, dh   = display.getWidth(), display.getHeight()
    setpoint = dw / 2.0     # el auto debe mantenerse en el centro horizontal del display

    integral, prev_err, prev_t = 0.0, 0.0, time.time()
    steering, no_line_frames   = 0.0, 0
    frame_cnt  = 0

    # * pos_streak lleva la cuenta de cuántas veces seguidas la cámara detectó
    # * una persona. neg_streak cuenta cuántas veces seguidas no detectó nada.
    # * Cuando pos_streak llega a 2 se confirma la detección y se activa el freno.
    # * Cuando neg_streak llega a 4 y el freno ya expiró, el auto vuelve a circular.
    pos_streak = 0
    neg_streak = 0
    brake_hold = 0       # cuenta regresiva del freno mínimo garantizado (frames)
    threat     = 'none'  # estado actual del sistema: 'none', 'pedestrian' o 'objeto'

    last_press = {}
    last_hits  = 0
    last_score = 0.0
    last_wins  = 0
    last_box   = None

    lidar_alert    = False
    lidar_dist_m   = None
    lidar_dist_str = '---'
    lidar_streak   = 0
    # * static_frames acumula cuántos frames seguidos el láser está bloqueado sin
    # * que la cámara haya confirmado una persona. Cuando supera el límite configurado
    # * asumimos que el obstáculo es fijo como un edificio o un cono, y activamos
    # * un bypass temporal para que el auto pueda avanzar y no quedarse atascado.
    static_frames  = 0
    lidar_pause    = 0   # frames restantes de bypass activo

    driver.setCruisingSpeed(CRUISE_SPEED)
    print("Controlador listo — PID + SVM + LiDAR")

    # =========================================================================
    # LOOP PRINCIPAL — se ejecuta cada 10ms (timestep de Webots)
    # =========================================================================
    while driver.step() != -1:
        t  = time.time()
        dt = max(t - prev_t, 1e-3)
        frame_cnt += 1

        # * Cada frame capturamos la imagen de la cámara frontal del BMW y la
        # * redimensionamos al tamaño del display de diagnóstico para procesarla.
        image = get_image(camera)
        bgr   = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(bgr, (dw, dh))

        # * Leemos el láser cada 3 frames para no sobrecargar el sistema. El
        # * contador lidar_streak sube solo con lecturas consecutivas de alerta
        # * para descartar picos aislados que no representan un obstáculo real.
        if lidar and frame_cnt % LIDAR_EVERY == 0:
            lidar_alert, lidar_dist_m, lidar_dist_str = lidar_read(lidar, _fov, _n_rays)
            lidar_streak = (lidar_streak + 1) if lidar_alert else 0
            print(f"[LIDAR] f={frame_cnt:05d} dist={lidar_dist_str} alert={'SI' if lidar_alert else 'no'} streak={lidar_streak}/{LIDAR_CONFIRM} threat={threat}")

        # * Corremos el detector de personas cada 10 frames porque es la operación
        # * más lenta del sistema. pos_streak sube con cada detección confirmada
        # * y neg_streak sube cada vez que no se ve a nadie en la imagen.
        if svm_model and frame_cnt % DETECT_EVERY == 0:
            detected, last_hits, last_score, last_wins, last_box = svm_detect(bgr, svm_model)
            if detected:
                pos_streak = min(pos_streak + 1, CONFIRM_N + 3)
                neg_streak = 0
            else:
                neg_streak = min(neg_streak + 1, RELEASE_N + 3)
                pos_streak = max(pos_streak - 1, 0)
            print(f"[SVM] f={frame_cnt:05d} wins={last_wins} "
                  f"hits={last_hits}/{MIN_HITS} score={last_score:.3f} "
                  f"thresh={SVM_THRESHOLD} pos={pos_streak}/{CONFIRM_N} "
                  f"neg={neg_streak}/{RELEASE_N} "
                  f"lidar={'ALERTA' if lidar_alert else 'ok'}({lidar_dist_str}) threat={threat}")

        # ── Decisión de frenado ───────────────────────────────────────────────
        # * Primero revisamos si la cámara confirmó una persona. Se necesitan dos
        # * detecciones seguidas para evitar falsos positivos. Una vez confirmado,
        # * el auto frena y mantiene el freno al menos un segundo completo aunque
        # * la persona ya no esté en el encuadre, para darle tiempo de cruzar.
        # * Para soltar el freno se requieren cuatro lecturas negativas seguidas
        # * después de que expire ese segundo mínimo garantizado.
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
                threat = 'none'   # freno expiró sin nueva confirmación → liberar

        # * Si el bypass está activo ignoramos el láser durante los frames
        # * programados para que el auto pueda avanzar más allá del obstáculo fijo.
        if lidar_pause > 0:
            lidar_pause -= 1
        else:
            # * A menos de 5 metros el láser frena al auto sin importar nada más,
            # * ni si la cámara ya actuó ni el estado en que esté el sistema.
            if lidar and lidar_dist_m is not None and lidar_dist_m < LIDAR_OVERRIDE_M:
                if threat != 'objeto':
                    brake_hold = HOLD_FRAMES
                threat = 'objeto'

            # * Entre 5 y 8 metros el láser frena solo si la cámara no ha actuado,
            # * para no sobreescribir una detección de persona que ya está activa.
            if lidar and lidar_dist_m is not None and lidar_dist_m < LIDAR_EMERGENCY_M and threat == 'none':
                threat     = 'objeto'
                brake_hold = HOLD_FRAMES

            # * Si el láser confirmó una alerta y no había amenaza activa, frenamos
            # * con el tiempo mínimo garantizado de un segundo.
            if lidar and lidar_streak >= LIDAR_CONFIRM and threat == 'none':
                threat     = 'objeto'
                brake_hold = HOLD_FRAMES

        # * Si el láser lleva más de 2.5 segundos bloqueado y la cámara no confirmó
        # * ninguna persona en todo ese tiempo, asumimos que es un objeto fijo como
        # * un edificio, un poste o un cono. Activamos el bypass para que el auto
        # * avance unos segundos ignorando esa lectura y pueda seguir su recorrido.
        if threat == 'objeto' and pos_streak < CONFIRM_N:
            static_frames += 1
            if static_frames >= STATIC_TIMEOUT:
                threat        = 'none'
                static_frames = 0
                lidar_pause   = LIDAR_PAUSE_F
                print(f"[LiDAR] Obstáculo estático — bypass {LIDAR_PAUSE_F} frames")
        else:
            static_frames = 0

        # ── Seguimiento de carril con PID ─────────────────────────────────────
        # * Para saber a dónde girar el volante el auto busca primero las líneas
        # * amarillas del carril. Si no encuentra ninguna y no está en una cebra,
        # * busca en la imagen en escala de grises: los bordes del asfalto, las
        # * marcas blancas de los carriles cruzados y las orillas del pavimento
        # * son visibles en gris aunque no haya amarillo. Esto es exactamente lo
        # * que hace un conductor en una intersección sin marcas: usa los bordes
        # * de la calzada para mantenerse centrado aunque no haya líneas pintadas.
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        edges = cv2.Canny(ymask, 50, 150)
        roi   = apply_roi(edges, dh, dw)
        lines = filter_lines(cv2.HoughLinesP(roi, 1, np.pi/180, 20,
                                             minLineLength=20, maxLineGap=15))
        center = compute_center(lines)

        # * Si el amarillo cubre más del 20% de la zona de visión hay una cebra
        # * peatonal, no una línea de carril. El PID no actúa y el auto cruza recto.
        yellow_frac = (ymask[int(dh * 0.60):] > 0).mean()
        if yellow_frac > 0.20:
            center = None   # cebra detectada → cruzar con el ángulo anterior

        # * Si no hay líneas amarillas y tampoco hay cebra, buscamos en escala de
        # * grises los bordes del asfalto. Pero esperamos a estar 30 frames dentro
        # * de la zona sin línea antes de activar esta búsqueda: los primeros frames
        # * después de una cebra o un frenado por peatón el campo visual todavía
        # * tiene ruido de los bordes gruesos del cruce y los cuerpos de las personas.
        # * Tampoco activamos el gris si la SVM acaba de ver peatones, porque sus
        # * siluetas crean bordes diagonales que el algoritmo confundiría con la orilla
        # * de la calzada. Solo usamos el resultado si hay bordes en ambos lados.
        if center is None and yellow_frac <= 0.20 and no_line_frames > 30 and pos_streak == 0:
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges_g = cv2.Canny(gray, 50, 150)
            roi_g   = apply_roi(edges_g, dh, dw)
            lines_g = filter_lines(cv2.HoughLinesP(roi_g, 1, np.pi/180, 40,
                                                   minLineLength=40, maxLineGap=10))
            if lines_g is not None:
                lx_g, rx_g = [], []
                for l in lines_g:
                    x1, y1, x2, y2 = l[0]
                    if x2 == x1: continue
                    slope = (y2 - y1) / (x2 - x1)
                    (lx_g if slope < 0 else rx_g).append((x1 + x2) / 2)
                if lx_g and rx_g:   # ambos bordes visibles → centro confiable
                    center = (np.mean(lx_g) + np.mean(rx_g)) / 2.0

        # ── Pantalla de diagnóstico ───────────────────────────────────────────
        # * La pantalla se actualiza cada 3 frames para no ralentizar la simulación.
        # * Muestra en tiempo real qué está viendo cada sistema: el rectángulo
        # * verde-amarillo es la zona de búsqueda de personas, el triángulo es el
        # * cono del láser, la barra vertical cambia de color según el estado del
        # * láser, y las líneas cian son los bordes del carril que usa el PID.
        if frame_cnt % DISPLAY_EVERY == 0:
            viz = frame.copy()

            roi_y0 = int(dh * 0.59)
            roi_y1 = int(dh * 0.85)
            cv2.rectangle(viz, (0, roi_y0), (dw - 1, roi_y1), (200, 200, 0), 1)

            hx0 = int(dw * 0.30);  hx1 = int(dw * 0.70)
            cv2.line(viz, (hx0, roi_y0), (hx0, roi_y1), (0, 140, 255), 1)
            cv2.line(viz, (hx1, roi_y0), (hx1, roi_y1), (0, 140, 255), 1)

            lidar_bar_col = (0, 0, 255) if lidar_alert else (0, 200, 0)
            cv2.rectangle(viz, (0, 0), (3, dh - 1), lidar_bar_col, -1)

            cam_fov_deg = math.degrees(1.0)     # FOV de la cámara ≈ 57° (1 rad)
            cone_px     = int(_half_deg * (dw / cam_fov_deg))
            cx_lidar    = dw // 2
            cv2.line(viz, (cx_lidar, dh - 1), (cx_lidar - cone_px, 0), lidar_bar_col, 2)
            cv2.line(viz, (cx_lidar, dh - 1), (cx_lidar + cone_px, 0), lidar_bar_col, 2)

            if last_box is not None and last_score >= SVM_THRESHOLD:
                bx1, by1, bx2, by2 = last_box
                ih, iw = bgr.shape[:2]
                fx1 = int(bx1 * dw / iw);  fy1 = int(by1 * dh / ih)
                fx2 = int(bx2 * dw / iw);  fy2 = int(by2 * dh / ih)
                cv2.rectangle(viz, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

            if lines is not None:
                for l in lines:
                    cv2.line(viz, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 255), 1)

            display_color(display, viz)

            if threat == 'objeto':
                display.setColor(0xFF6600)   # naranja = obstáculo genérico (láser)
                display.drawText("OBJETO", 2, 2)
            elif threat == 'pedestrian':
                display.setColor(0xFF0000)   # rojo = persona confirmada por cámara
                display.drawText("PEATON", 2, 2)
            else:
                display.setColor(0x00FF00)   # verde = circulando sin amenaza
                display.drawText("PID OK", 2, 2)

            display.setColor(0xFFFFFF)
            display.drawText(f"V:{CRUISE_SPEED} St:{steering:.2f}", 2, 12)

            score_color = 0xFF4444 if last_score >= SVM_THRESHOLD else (0xFFAA00 if last_score >= 0.0 else 0xCCCCCC)
            display.setColor(score_color)
            display.drawText(f"SVM:{last_score:.3f}(>{SVM_THRESHOLD})", 2, 22)

            pos_color = 0xFF2222 if pos_streak >= CONFIRM_N else 0xFF8800
            display.setColor(pos_color)
            display.drawText(f"pos:{pos_streak}/{CONFIRM_N}", 2, 32)

            lidar_color = 0xFF2222 if lidar_alert else 0x44FF44
            display.setColor(lidar_color)
            display.drawText(f"LiDAR:{lidar_dist_str} ±{_half_deg}°", 2, 42)

            display.setColor(0xAAAAAA)
            display.drawText(f"hold:{brake_hold}", 2, 52)

        # ── Acción de control: frenar o circular ─────────────────────────────
        # * Si hay una amenaza activa el auto frena completamente y enciende los
        # * intermitentes solo cuando el láser detectó un obstáculo, no cuando fue
        # * la cámara quien vio a la persona. Mientras el auto está detenido el
        # * volante vuelve poco a poco al centro: no tiene sentido conservar el
        # * ángulo de curva durante los segundos que dura el freno porque la cebra
        # * y la intersección que siguen son rectas y el auto debe reanudar derecho.
        if threat != 'none':
            driver.setCruisingSpeed(0)
            driver.setBrakeIntensity(1.0)
            driver.setHazardFlashers(threat == 'objeto')
            integral  = 0.0    # resetear integral evita que el PID arranque sesgado
            steering *= 0.90   # centrar volante mientras frena: 0.90^20 ≈ 0.12
            prev_t = t
            continue

        # * Sin amenaza el auto circula y el PID corrige el volante en tiempo real
        # * calculando cuánto se alejó el carril del centro de la imagen.
        driver.setHazardFlashers(False)
        driver.setBrakeIntensity(0.0)
        driver.setCruisingSpeed(CRUISE_SPEED)

        if center is not None:
            # * Calculamos el error como la distancia entre donde está el centro del
            # * carril y donde debería estar. Un error positivo significa que el auto
            # * está muy a la derecha; negativo, muy a la izquierda. El PID convierte
            # * ese error en un ángulo de volante y el suavizado evita tirones bruscos.
            no_line_frames = 0
            error    = (center - setpoint) / setpoint   # normalizado a [-1, 1]
            integral = max(-0.5, min(0.5, integral + error * dt))
            raw_s    = Kp * error + Ki * integral + Kd * (error - prev_err) / dt
            raw_s    = max(-MAX_ANGLE, min(MAX_ANGLE, raw_s))
            steering = max(steering - MAX_STEER_RATE,
                           min(steering + MAX_STEER_RATE, raw_s))
            prev_err = error
        else:
            # * Cuando no hay línea amarilla visible el auto congela el volante en
            # * la posición que traía durante 20 frames (200 ms) y después lo devuelve
            # * gradualmente a cero. El truco es que en la ciudad los cruces vienen
            # * justo después de una curva, así que el tramo sin línea es recto.
            # * Un decay de 0.95 por frame devuelve el ángulo a menos del 1% en
            # * 90 frames más — el auto llega derecho al otro lado del cruce.
            no_line_frames += 1
            integral       *= 0.6    # reducir el historial acumulado del integrador
            prev_err        = 0.0    # evitar sacudida al recuperar la línea
            if no_line_frames > 20:
                steering *= 0.95     # decay: 0.95^90 ≈ 0.01 → prácticamente recto al salir

        driver.setSteeringAngle(steering)
        prev_t = t

        # * Presionar A guarda una captura de pantalla con marca de tiempo,
        # * útil para revisar casos difíciles o recolectar imágenes de entrenamiento.
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and t - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = t
                if key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"[A] {ts}.png guardada")

if __name__ == "__main__":
    main()
