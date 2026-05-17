"""
=============================================================================
ACTIVIDAD: Detección de Carriles con Controlador PID en Webots
=============================================================================
Alumno  : Joel Arturo Becerril Balderas
Matrícula: A01797427
Materia  : Navegación Autónoma — Maestría en Inteligencia Artificial Aplicada
Tecnológico de Monterrey

DESCRIPCIÓN GENERAL:
Este script implementa un controlador de seguimiento de carril para un
vehículo autónomo simulado en Webots. El vehículo sigue la línea amarilla
central de la carretera utilizando visión por computadora y un controlador
PID para ajustar el ángulo de dirección en tiempo real.

PIPELINE DE PROCESAMIENTO (según los materiales del módulo):
    1. Obtención de imagen desde la cámara a bordo
    2. Segmentación de color amarillo en espacio HSV
    3. Detección de bordes con algoritmo de Canny
    4. Definición y aplicación de Región de Interés (ROI) con fillPoly
    5. Detección de líneas con Transformada de Hough (HoughLinesP)
    6. Cálculo del error más pequeño respecto al setpoint (centro de imagen)
    7. Control PID: la salida es el ángulo de dirección del vehículo
    8. Manejo de cruces sin línea: hold del último ángulo conocido

CARACTERÍSTICAS DEL CONTROLADOR PID:
    - Setpoint: mitad del ancho de la imagen (centro horizontal)
    - Error: distancia del punto medio X de la línea más cercana al setpoint
    - Filtro de líneas horizontales: se descartan líneas de cebra en cruces
    - Velocidad constante: 50 km/h (mínimo requerido por la actividad)
    - Ángulo por omisión en cruces: mantiene el último ángulo válido

CONTROLES:
    M   — alterna entre modo PID Autopilot y modo Manual
    ↑↓  — (manual) aumentar/reducir velocidad
    ←→  — (manual) girar izquierda/derecha
    A   — capturar imagen con timestamp
=============================================================================
"""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from collections import deque
from datetime import datetime
import os
import time


# =============================================================================
# PARÁMETROS DE CONFIGURACIÓN DEL VEHÍCULO
# =============================================================================
# Velocidad constante de 50 km/h — mínimo requerido por la actividad.
# La velocidad no varía durante el recorrido autónomo.
SPEED         = 50
MAX_ANGLE     = 0.5    # Ángulo máximo de dirección en radianes (~28.6°)
DEFAULT_ANGLE = 0.0    # Ángulo por omisión: ir recto

# Parámetros del modo manual (control por teclado)
SPEED_INCR    = 5      # Incremento de velocidad por tecla (km/h)
ANGLE_INCR    = 0.05   # Incremento de ángulo por tecla (radianes)
DEBOUNCE_TIME = 0.1    # Tiempo mínimo entre pulsaciones (segundos)
MANUAL_MODE   = False  # False = inicia en PID Autopilot, True = inicia manual


# =============================================================================
# PARÁMETROS DE DETECCIÓN DE COLOR AMARILLO (ESPACIO HSV)
# =============================================================================
# Se utiliza el espacio de color HSV en lugar de escala de grises porque
# permite aislar el color amarillo de la línea central con mayor precisión,
# independientemente de la iluminación.
#
# En OpenCV el canal H va de 0 a 180 (no 0-360).
# El amarillo en HSV corresponde aproximadamente a H: 15-35.
# S (saturación) y V (valor/brillo) filtran píxeles opacos o muy oscuros.
#
# Rango definido:
#   H: 15–35  → tono amarillo
#   S: 80–255 → saturación mínima para no capturar blancos o grises
#   V: 80–255 → brillo mínimo para no capturar sombras
YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)


# =============================================================================
# PARÁMETROS DEL ALGORITMO DE CANNY (DETECCIÓN DE BORDES)
# =============================================================================
# Canny detecta cambios abruptos de intensidad en la imagen.
# Se aplica sobre la máscara amarilla (no sobre la imagen completa),
# lo que reduce el ruido al enfocarse solo en los píxeles de la línea.
#
# CANNY_LOW  — umbral inferior: gradientes por debajo se descartan
# CANNY_HIGH — umbral superior: gradientes por encima siempre son bordes
# Gradientes entre ambos umbrales se aceptan solo si conectan con un borde fuerte.
CANNY_LOW  = 50
CANNY_HIGH = 150


# =============================================================================
# PARÁMETROS DE LA TRANSFORMADA DE HOUGH (HoughLinesP)
# =============================================================================
# HoughLinesP detecta segmentos de línea recta (no líneas infinitas).
# Cada píxel de borde "vota" por todas las líneas que podrían pasar por él.
# Los segmentos que acumulan suficientes votos se consideran líneas detectadas.
#
# HOUGH_RHO        — resolución de distancia en píxeles (precisión espacial)
# HOUGH_THETA      — resolución angular: 1 grado = π/180 radianes
# HOUGH_THRESHOLD  — mínimo de votos para considerar un segmento válido
#                    (valor bajo = detecta más líneas, incluye más ruido)
# HOUGH_MIN_LENGTH — longitud mínima del segmento en píxeles
#                    (filtra segmentos cortos de ruido o puntos aislados)
# HOUGH_MAX_GAP    — brecha máxima entre segmentos para unirlos en uno solo
#                    (útil para líneas discontinuas como la línea central)
HOUGH_RHO        = 1
HOUGH_THETA      = np.pi / 180
HOUGH_THRESHOLD  = 10
HOUGH_MIN_LENGTH = 10
HOUGH_MAX_GAP    = 150

# ── Umbral mínimo de píxeles amarillos ────────────────────────────────────────
# Si la máscara HSV contiene menos de MIN_YELLOW_PIXELS, la detección no es
# confiable: las pocas líneas encontradas pueden ser ruido o fragmentos de otra
# calle visible en la intersección. Se fuerza error=None y se activa el hold.
# En cámara 128x64: 40px ≈ 0.5% del frame → umbral conservador.
MIN_YELLOW_PIXELS = 40


# =============================================================================
# FILTRO DE LÍNEAS HORIZONTALES
# =============================================================================
# Las líneas de cruce peatonal (cebra) son casi completamente horizontales.
# Se filtran calculando la diferencia vertical entre los extremos del segmento:
#   |y2 - y1| < MIN_VERT_DIFF → línea horizontal → se descarta
#
# NOTA: Las rayas de cebra en este mundo de Webots son AMARILLAS (no blancas),
# por lo que SÍ pasan el filtro HSV y aparecen en la máscara. En los cruces,
# las franjas horizontales de cebra generan muchos píxeles amarillos (>480)
# y se detectan como segmentos casi horizontales → clasificados por case_b/case_a.
#
# El único caso donde líneas "horizontales" entran al pipeline en secciones normales
# es cuando el carril amarillo aparece en perspectiva muy oblicua (curvas cerradas),
# donde un segmento de 15px a 20° tiene |y2-y1| = 5px.
#
# MIN_VERT_DIFF = 5 filtra líneas con ángulo < arcsin(5/L). Para L=10px (mínimo
# de HoughLinesP), solo líneas con ángulo > 30° pasan. Para L=20px, ángulo > 14°.
# Los cruces reales se detectan por yellow_pixels > 200 en at_crosswalk, no aquí.
# NOTA: cámara 128x64 → ROI de solo 32px de alto. En curvas suaves la línea
# corre a ~5° en la cámara: segmento de 20px → |y2-y1| = 1.74px → con
# MIN_VERT_DIFF=2 se filtraba → raw_error=None → SIN LINEA en plena curva → deriva.
# MIN_VERT_DIFF=1 solo filtra líneas PERFECTAMENTE horizontales (|y2-y1|=0).
# Las cebras amarillas en cruces (>480px) ya se manejan por at_crosswalk,
# así que no necesitamos este filtro para ellas.
MIN_VERT_DIFF = 1


# =============================================================================
# GANANCIAS DEL CONTROLADOR PID
# =============================================================================
# El controlador PID ajusta el ángulo de dirección del vehículo basándose
# en el error entre la posición de la línea y el centro de la imagen.
#
# La fórmula es:
#   steering = Kp * error + Ki * integral(error) + Kd * d(error)/dt
#
# Kp (Proporcional):
#   Corrige el error actual. Si la línea está desplazada 20px a la derecha,
#   gira a la derecha proporcionalmente.
#   Valor alto → reacciones más bruscas. Valor bajo → reacciones lentas.
#
# Ki (Integral):
#   Corrige errores acumulados en el tiempo. Elimina desviaciones persistentes
#   que el proporcional no llega a corregir completamente.
#   Valor alto → puede causar oscilaciones. Valor bajo → corrección lenta.
#
# Kd (Derivativo):
#   Anticipa la tendencia del error. Si el error está creciendo rápido,
#   aplica corrección adicional antes de que sea mayor.
#   Actúa como amortiguador de oscilaciones.
#
# Valores calibrados experimentalmente a 50 km/h en el mundo de Webots:
Kp = 0.035   # aumentado para corregir curvas más rápido (era 0.025)
Ki = 0.0002
Kd = 0.004   # reducido para menos oscilación y menor kick en recovery (era 0.008)


# =============================================================================
# FUNCIÓN: get_image
# =============================================================================
def get_image(camera):
    """
    Obtiene la imagen cruda de la cámara a bordo del vehículo.

    Webots devuelve los datos de imagen como un buffer de bytes en formato BGRA
    (Azul, Verde, Rojo, Alpha). Se convierte a un arreglo NumPy con forma
    (alto, ancho, 4) para procesarlo con OpenCV.
    """
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


# =============================================================================
# FUNCIÓN: detect_yellow
# =============================================================================
def detect_yellow(image):
    """
    Segmenta el color amarillo de la imagen usando el espacio de color HSV.

    Pasos:
        1. Convierte de BGRA a BGR (OpenCV no soporta BGRA→HSV directo)
        2. Convierte de BGR a HSV
        3. Aplica una máscara con el rango de amarillo definido

    Retorna una imagen binaria donde los píxeles amarillos son 255 (blanco)
    y el resto es 0 (negro). Esto aísla únicamente la línea central amarilla.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return mask


# =============================================================================
# FUNCIÓN: apply_roi
# =============================================================================
def apply_roi(edges, height, width):
    """
    Define y aplica una Región de Interés (ROI) sobre la imagen de bordes.

    La ROI es un rectángulo que cubre el 50% inferior de la imagen.
    Su propósito es eliminar el cielo, el horizonte y objetos distantes,
    enfocando la detección únicamente en la superficie de la carretera
    inmediatamente frente al vehículo.

    Implementación:
        - Se crea una máscara negra del mismo tamaño que la imagen de bordes
        - Se dibuja el rectángulo de la ROI en blanco usando fillPoly
        - Se aplica la máscara con AND bit a bit sobre la imagen de bordes
        - Solo los bordes dentro de la ROI permanecen visibles

    Vértices de la ROI:
        Inferior-izquierdo  (0, height)
        Superior-izquierdo  (0, height * 0.5)
        Superior-derecho    (width, height * 0.5)
        Inferior-derecho    (width, height)
    """
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0,     height),
        (0,     int(height * 0.5)),
        (width, int(height * 0.5)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(edges, mask)


# =============================================================================
# FUNCIÓN: detect_lines
# =============================================================================
def detect_lines(roi_edges):
    """
    Detecta líneas usando un sistema de dos niveles:

    NIVEL 1 — HoughLinesP (Transformada de Hough Probabilística):
        Detecta segmentos de línea con coordenadas [x1, y1, x2, y2].
        Más preciso en condiciones normales. Requiere longitud mínima,
        por lo que puede fallar cuando quedan pocos píxeles amarillos
        (inicio de curva, borde de intersección).

    NIVEL 2 — HoughLines (Transformada de Hough Estándar):
        Detecta líneas infinitas representadas por (rho, theta).
        Mucho más sensible: puede detectar una línea con muy pocos píxeles.
        Se activa como fallback cuando HoughLinesP no encuentra nada.
        Las líneas infinitas se convierten a segmentos usando los extremos
        de la imagen para calcular coordenadas [x1, y1, x2, y2].

    Este enfoque híbrido permite seguir la línea amarilla incluso cuando
    solo hay fragmentos pequeños visibles en cruces o curvas pronunciadas.

    Retorna lista de [x1, y1, x2, y2] o lista vacía si ninguno detecta.
    """
    # Solo HoughLinesP — el fallback a HoughLines fue eliminado porque con pocos
    # píxeles amarillos (50-80px) y threshold=5 generaba 100-170 líneas fantasma
    # que daban errores consistentes de -4 a -7px y desviaban el auto hasta sacarlo
    # de la carretera. El comportamiento SIN LINEA (hold last_steering) es más seguro.
    lines_p = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )
    if lines_p is not None:
        return lines_p.reshape(-1, 4)
    return []


# =============================================================================
# FUNCIÓN: compute_error
# =============================================================================
def compute_error(lines, setpoint, expected_x, vert_diff=None):
    """
    Selecciona la línea más cercana a expected_x y retorna su error respecto al setpoint.

    vert_diff: umbral de filtro de líneas horizontales. Si no se pasa, usa MIN_VERT_DIFF.
    Se pasa dinámicamente desde el bucle principal para filtrar más agresivamente
    cuando hay muchos píxeles amarillos (zona de cruce/intersección).

    Retorna (error, best_line) o (None, None) si no hay líneas válidas.
    """
    if vert_diff is None:
        vert_diff = MIN_VERT_DIFF
    best_error    = None
    best_line     = None
    best_dist_exp = float('inf')
    for x1, y1, x2, y2 in lines:
        if abs(y2 - y1) < vert_diff:
            continue
        mid_x    = (x1 + x2) / 2
        dist_exp = abs(mid_x - expected_x)
        if dist_exp < best_dist_exp:
            best_dist_exp = dist_exp
            best_error    = mid_x - setpoint
            best_line     = (x1, y1, x2, y2)
    return best_error, best_line


# =============================================================================
# FUNCIÓN: display_image
# =============================================================================
def display_image(display, image):
    """
    Muestra la imagen procesada en el display a bordo del vehículo en Webots.

    Si la imagen es en escala de grises (2D), se convierte a RGB apilando
    el mismo canal tres veces antes de enviarlo al display.
    """
    if len(image.shape) == 2:
        image = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image.tobytes(), Display.RGB,
        width=image.shape[1], height=image.shape[0]
    )
    display.imagePaste(image_ref, 0, 0, False)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """
    Bucle principal del controlador. Inicializa los dispositivos del vehículo
    y ejecuta el pipeline de visión + PID en cada paso de simulación.
    """

    # ── Inicialización del vehículo y dispositivos ─────────────────────────────
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    # Cámara a bordo — proporciona imagen BGRA en cada timestep
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Display a bordo — muestra la máscara amarilla para verificar detección
    display_img = Display("display_image")

    # Teclado — permite cambiar entre modo manual y autopilot
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # ── Dimensiones de la imagen y setpoint ───────────────────────────────────
    # El setpoint es el centro horizontal de la imagen.
    # Para una cámara de 256px de ancho, setpoint = 128.
    # El controlador buscará mantener la línea amarilla en esta posición.
    width    = camera.getWidth()
    height   = camera.getHeight()
    setpoint = width / 2

    # ── Variables de estado del controlador PID ────────────────────────────────
    integral      = 0.0   # acumulado del error en el tiempo (término I)
    prev_error    = 0.0   # error del frame anterior (para calcular derivada)
    smooth_error  = 0.0   # error suavizado con EMA (filtro exponencial)
    # Factor de suavizado EMA (Exponential Moving Average):
    # alpha cercano a 1 → más peso al frame actual (más reactivo, más ruidoso)
    # alpha cercano a 0 → más peso al historial (más suave, respuesta más lenta)
    alpha         = 0.35

    # ── Variables para manejo de cruces sin línea ──────────────────────────────
    # Dos situaciones distintas cuando error = None:
    #   A) Cebra/cruce: lines > 0 pero todas horizontales → ir recto de inmediato (×0.80/frame)
    #   B) Sin línea real: lines == 0 → mantener 5 frames y luego decaer (×0.88/frame)
    #
    # smooth_error NO se resetea en cebra breve (1-3 frames) para mantener continuidad
    # del EMA y evitar que el primer raw_error post-cruce sea amplificado al x0.35.
    # Solo se resetea después de 4+ frames sostenidos de cebra (cruce largo).
    #
    # was_cebra: en el primer frame OK tras cebra, se iguala prev_error = smooth_error
    # para que la derivada sea ~0 ese frame y no cause un "kick" brusco de dirección.
    last_steering     = 0.0
    no_line_count     = 0
    consecutive_cebra = 0
    was_cebra         = False
    NO_LINE_HOLD      = 5     # frames de hold solo cuando no hay líneas (~160ms)

    # Buffer circular de los últimos ángulos de dirección en modo PID normal.
    # Al entrar a un cruce, se usa el promedio de este buffer como ángulo de hold.
    # Esto es más estable que last_steering (un solo frame ruidoso) y representa
    # la trayectoria real del auto en los ~200ms previos al cruce.
    # Si el promedio es alto (curva pronunciada), se escala al 40% para evitar
    # amplificar la desviación de la curva durante el cruce.
    steering_buffer = deque(maxlen=6)
    crosswalk_hold  = 0.0   # ángulo capturado al entrar al cruce

    prev_time = time.time()

    # ── Variables del modo manual ──────────────────────────────────────────────
    speed      = 50
    angle      = 0.0
    last_press = {}
    manual     = MANUAL_MODE

    driver.setCruisingSpeed(speed)
    print(f"Modo: {'MANUAL' if manual else 'PID AUTOPILOT'} | M = cambiar modo | A = capturar imagen")
    print(f"Setpoint: {setpoint}px | Cámara: {width}x{height}px")

    # ==========================================================================
    # BUCLE PRINCIPAL DE SIMULACIÓN
    # ==========================================================================
    while robot.step() != -1:
        current_time = time.time()
        dt = max(current_time - prev_time, 1e-6)  # delta tiempo entre frames
        prev_time = current_time

        # ── PASO 1: Obtención de imagen desde la cámara a bordo ───────────────
        image = get_image(camera)

        # ── PASO 2: Segmentación de color amarillo en espacio HSV ─────────────
        # Genera imagen binaria donde solo los píxeles amarillos son visibles.
        # Esto aísla la línea central y descarta carretera, cielo y edificios.
        yellow_mask = detect_yellow(image)

        # ── PASO 3: Detección de bordes con algoritmo de Canny ────────────────
        # Se aplica Canny sobre la máscara amarilla para extraer los contornos
        # de la línea central. Al aplicarlo sobre la máscara (no la imagen
        # original), se reduce el ruido significativamente.
        edges = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)

        # ── PASO 4: Definición y aplicación de la Región de Interés (ROI) ─────
        # fillPoly dibella el rectángulo ROI en la máscara y elimina todo
        # lo que está fuera del 50% inferior de la imagen.
        roi = apply_roi(edges, height, width)

        # ── PASO 5: Detección de líneas con Transformada de Hough ─────────────
        # HoughLinesP devuelve una lista de segmentos [x1, y1, x2, y2].
        # En una recta típica se detectan entre 4 y 12 segmentos.
        lines = detect_lines(roi)

        # Conteo de píxeles amarillos — usado para clasificar el frame y en debug
        yellow_pixels = cv2.countNonZero(yellow_mask)

        # ── PASO 6: Selección de línea más cercana al centro ─────────────────
        # expected_x = setpoint (centro de la imagen) siempre.
        # Esto significa que compute_error SIEMPRE elige la línea más cercana a x=64.
        #
        # Por qué NO usamos expected_x = setpoint + smooth_error:
        # Cuando smooth_error deriva en una intersección (línea equivocada a x=55),
        # expected_x se desplaza a 56, y la línea equivocada (dist=1) gana sobre
        # la correcta (dist=8) → cascada. Con expected_x=setpoint=64, la línea
        # correcta siempre tiene dist=0 y gana sobre cualquier línea equivocada
        # que esté más lejos del centro → cascada eliminada.
        expected_x = setpoint

        # Filtro horizontal adaptativo:
        # En zonas de cruce/intersección (yellow_pixels > 300), la cebra amarilla
        # genera franjas casi horizontales que compiten con la línea central.
        # Se usa un umbral más estricto (5px) para filtrarlas sin afectar curvas normales:
        #   - Línea central a 20° en segmento 20px → |y2-y1|=6.8px → pasa vert_diff=5 ✓
        #   - Franja de cebra a 0-5° → |y2-y1|≤1.7px → filtrada ✓
        # En secciones normales (< 300px) se usa MIN_VERT_DIFF=1 para no perder
        # la línea en curvas suaves.
        # vert_diff=5 en >300px estaba filtrando la línea central en curvas de
        # 430-450px (línea a ~14° → |y2-y1|=4.8px < 5 → filtrada → SIN LINEA → oscilación).
        # vert_diff=3 filtra solo líneas a <8.6° para segmentos de 20px → elimina
        # franjas de cebra (0°) sin eliminar la línea central en curvas pronunciadas.
        adaptive_vert = 3 if yellow_pixels > 300 else MIN_VERT_DIFF

        if yellow_pixels < MIN_YELLOW_PIXELS:
            raw_error = None
            best_line = None
        else:
            raw_error, best_line = compute_error(lines, setpoint, expected_x, adaptive_vert)
            # Sin jump filter: compute_error ya selecciona la línea más cercana a
            # expected_x. El jump filter adicional causaba crashes: rechazaba líneas
            # VÁLIDAS cuando el auto se había desplazado físicamente (post-intersección)
            # y la línea correcta aparecía >20px de smooth_error → 9+ frames SIN LINEA
            # → auto fuera de carretera. El EMA (alpha=0.35) suaviza errores grandes.

        # ── Clasificación del frame ────────────────────────────────────────────
        # at_crosswalk = True únicamente por conteo de píxeles amarillos (>480):
        #
        # Las rayas de cebra en este mundo Webots son AMARILLAS. Cuando el auto
        # entra a un cruce, las franjas horizontales llenan el ROI y generan >480px
        # amarillos. Las curvas más pronunciadas llegan hasta ~440px.
        #
        # Se eliminó el antiguo "case_a" (raw_error=None AND lines>0 AND px>200)
        # porque en curvas suaves la línea amarilla corre casi horizontal en la cámara
        # (ángulo < ~12°): segmentos de 10px tienen |y2-y1| < 2 → filtrados →
        # raw_error=None → case_a se disparaba en plena curva → crosswalk_hold
        # con el escalado ×0.4 hacía que el auto casi fuera recto. Regresión.
        #
        # Con solo case_b (480+), los cruces reales siempre se detectan y las curvas
        # con 200-440px nunca se clasifican como cebra.
        at_crosswalk = (yellow_pixels > 480)

        # Suavizado EMA: reduce el ruido frame a frame sin perder reactividad.
        # No se aplica en at_crosswalk (incluyendo intersecciones grandes) para
        # evitar que errores de líneas equivocadas contaminen el historial.
        # Formula: smooth = alpha * nuevo + (1 - alpha) * histórico
        if raw_error is not None and not at_crosswalk:
            smooth_error = alpha * raw_error + (1 - alpha) * smooth_error
            smooth_error = float(np.clip(smooth_error, -8.0, 8.0))
            error = smooth_error
        else:
            error = None

        # Debug en terminal cada ~20 frames para monitorear detección
        if int(current_time * 10) % 20 == 0:
            estado = "CEBRA" if at_crosswalk else ("SIN LINEA" if error is None else "OK")
            print(f"Amarillo px: {yellow_pixels} | Líneas: {len(lines)} | Error: {round(error, 2) if error is not None else None} | {estado}")

        # ── Display diagnóstico ────────────────────────────────────────────────
        # Imagen RGB con capas visuales diferenciadas:
        #   · Fondo gris: máscara amarilla (qué ve el filtro HSV)
        #   · Gris claro (120,120,120): líneas filtradas (|y2-y1|<MIN_VERT_DIFF)
        #     → en el cruce verás muchas franjas horizontales gris: son la cebra
        #   · Verde: líneas VÁLIDAS (pasan MIN_VERT_DIFF) — candidatas para error
        #   · Rojo (2px): segmento seleccionado como referencia del PID
        #   · Azul vertical: setpoint (x=width/2, centro objetivo)
        disp_bgr = np.stack([yellow_mask // 2, yellow_mask // 2, yellow_mask // 2], axis=-1)
        for x1d, y1d, x2d, y2d in lines:
            if abs(y2d - y1d) < adaptive_vert:
                cv2.line(disp_bgr, (x1d, y1d), (x2d, y2d), (100, 100, 100), 1)  # gris = filtrada
            else:
                cv2.line(disp_bgr, (x1d, y1d), (x2d, y2d), (0, 220, 0), 1)     # verde = válida
        if best_line is not None:
            bx1, by1, bx2, by2 = best_line
            cv2.line(disp_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)          # rojo = seleccionada
        cv2.line(disp_bgr, (int(setpoint), 0), (int(setpoint), height - 1), (255, 0, 0), 1)  # azul = setpoint
        display_image(display_img, disp_bgr[:, :, ::-1])  # BGR→RGB para Webots

        # HUD velocímetro — texto blanco sobre el display diagnóstico
        estado_str = "CEBRA" if at_crosswalk else ("SIN LINEA" if error is None else "OK")
        error_str  = f"E:{round(error, 1)}" if error is not None else f"E:{estado_str}"
        display_img.setColor(0xFFFFFF)
        display_img.drawText(f"V:{SPEED}km/h",    2, 2)
        display_img.drawText(f"St:{steering:.3f}r", 2, 12)
        display_img.drawText(error_str,             2, 22)

        # ── Lectura de teclado ─────────────────────────────────────────────────
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and current_time - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = current_time

                if key == ord('M'):
                    # Alterna entre modo manual y autopilot PID
                    manual = not manual
                    integral = 0.0  # resetea integral al cambiar de modo
                    print(f"Modo: {'MANUAL' if manual else 'PID AUTOPILOT'}")
                elif key == ord('A'):
                    # Captura imagen con timestamp para documentación
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"Imagen guardada: {ts}.png")
                elif manual:
                    if key == keyboard.UP:
                        speed = min(speed + SPEED_INCR, 250)
                        print(f"Velocidad: {speed} km/h")
                    elif key == keyboard.DOWN:
                        speed = max(speed - SPEED_INCR, 0)
                        print(f"Velocidad: {speed} km/h")
                    elif key == keyboard.RIGHT:
                        angle = min(angle + ANGLE_INCR, MAX_ANGLE)
                    elif key == keyboard.LEFT:
                        angle = max(angle - ANGLE_INCR, -MAX_ANGLE)

        # ── PASO 7: Control PID — salida es el ángulo de dirección ────────────
        if manual:
            driver.setCruisingSpeed(speed)
            driver.setSteeringAngle(angle)
        else:
            if error is None:
                # ── PASO 8: Manejo de cruces sin línea ────────────────────────
                # A) Cebra (at_crosswalk=True):
                #    smooth_error NO se resetea en cebra breve (≤4 frames) para
                #    mantener continuidad del EMA. El EMA conserva la historia
                #    y el primer raw_error post-cruce no es amplificado al ×0.35.
                #    Solo se resetea después de 4+ frames sostenidos (cruce largo).
                #    El ángulo decae a recto (×0.80/frame) para no acumular deriva.
                #
                # B) Sin línea real (at_crosswalk=False):
                #    Se mantiene el ángulo NO_LINE_HOLD frames y luego decae (×0.88).
                #
                # En ambos casos: was_cebra=True señala al bloque PID siguiente que
                # en el primer frame OK debe igualar prev_error=smooth_error para
                # que la derivada sea ~0 y no cause un kick brusco de dirección.
                integral  = 0.0
                was_cebra = True

                if at_crosswalk:
                    consecutive_cebra += 1
                    if consecutive_cebra > 4:
                        smooth_error = 0.0   # reset EMA solo en cruce largo

                    if consecutive_cebra == 1:
                        # Primer frame de cruce: captura el promedio de trayectoria
                        # de los últimos 6 steerings (más estable que last_steering).
                        recent = list(steering_buffer)
                        crosswalk_hold = float(np.mean(recent)) if recent else last_steering
                        # Se mantiene el ángulo completo (no se escala).
                        # Si el cruce viene después de una curva, el auto DEBE
                        # continuar girando — reducir a 40% causaba que se saliera.

                    # Hold fijo los primeros 4 frames; luego decae hacia recto (×0.88)
                    if consecutive_cebra <= 4:
                        steering = crosswalk_hold
                    else:
                        steering = last_steering * 0.88

                    prev_error = smooth_error  # mantiene continuidad del término D
                else:
                    consecutive_cebra = 0
                    no_line_count    += 1
                    if no_line_count <= NO_LINE_HOLD:
                        steering = last_steering
                    else:
                        steering = last_steering * 0.88
                last_steering = steering
            else:
                no_line_count     = 0
                consecutive_cebra = 0

                # En el primer frame OK tras cebra: iguala prev_error a smooth_error
                # para que derivative = (smooth - smooth) / dt ≈ 0 → sin kick D.
                # A partir del siguiente frame el derivativo retoma normalmente.
                if was_cebra:
                    prev_error = smooth_error
                    was_cebra  = False

                # Término proporcional: reacción inmediata al error actual
                p_term = Kp * error

                # Término integral: corrige desviaciones acumuladas en el tiempo
                integral += error * dt
                i_term = Ki * integral

                # Término derivativo: amortigua oscilaciones anticipando cambios
                derivative = (error - prev_error) / dt
                d_term = Kd * derivative

                steering   = p_term + i_term + d_term
                prev_error = error

                # Registra el ángulo aplicado para el hold en próximos cruces
                steering_buffer.append(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))

            # Limita el ángulo al rango físico del vehículo
            steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
            last_steering = steering
            driver.setCruisingSpeed(SPEED)   # velocidad constante 50 km/h
            driver.setSteeringAngle(steering)


if __name__ == "__main__":
    main()
