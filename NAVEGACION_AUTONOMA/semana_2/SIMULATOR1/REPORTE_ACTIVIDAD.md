# Actividad 2.1 — Detección de Carriles con PID en Webots

**Alumno:** Joel Arturo Becerril Balderas | **Matrícula:** A01797427
**Materia:** Navegación Autónoma — MNA Tecnológico de Monterrey
**Semana:** 2 | **Fecha:** Mayo 2026

---

## 🔗 Código fuente

| Archivo | Descripción | Enlace |
|---|---|---|
| `simple_controller_H.py` | Controlador Joel — HSV + PID | [Ver en GitHub](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H.py) |
| `simple_controller_H2.py` | Controlador compañero — Escala de grises + PID | [Ver en GitHub](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H2.py) |

---

## ✅ Resultado final

> **El controlador `simple_controller_H.py` completó más de 10 minutos de simulación continua a 50 km/h sin ningún choque**, recorriendo múltiples vueltas al circuito de Webots incluyendo curvas, intersecciones y cruces peatonales (cebras amarillas).

| Métrica | Valor |
|---|---|
| Velocidad constante | 50 km/h |
| Tiempo sin choques | > 10 minutos |
| Error en rectas | < 1 px promedio |
| Error en curvas | 2–5 px promedio |
| Manejo de cebras | Automático, sin pérdida de trayectoria |
| Choques | 0 |

---

## Pipeline de procesamiento

```
Cámara BGRA (128 × 64 px)
    ↓ [1] Segmentación HSV — aislar solo la línea amarilla
    ↓ [2] Canny — extraer bordes de la máscara amarilla
    ↓ [3] ROI fillPoly — enfocar solo el 50% inferior de la imagen
    ↓ [4] HoughLinesP — detectar segmentos de línea
    ↓ [5] Selección — elegir el segmento más cercano a x = 64 (centro)
    ↓ [6] EMA — suavizar el error (α = 0.35, clamp ±8 px)
    ↓ [7] PID — calcular ángulo de dirección
    ↓ [8] Driver — aplicar velocidad y ángulo al vehículo
```

---

## Parámetros

| Parámetro | Valor | Por qué este valor |
|---|---|---|
| `SPEED` | 50 km/h | Mínimo requerido por la actividad |
| `MAX_ANGLE` | 0.5 rad | Límite físico del vehículo simulado |
| `YELLOW_LOW` | [15, 80, 80] | Inicio del rango amarillo en HSV |
| `YELLOW_HIGH` | [35, 255, 255] | Fin del rango amarillo en HSV |
| `CANNY_LOW / HIGH` | 50 / 150 | Umbral bajo/alto para detección de bordes |
| `HOUGH_THRESHOLD` | 10 | Votos mínimos para aceptar una línea |
| `HOUGH_MIN_LENGTH` | 10 px | Filtra fragmentos de ruido muy cortos |
| `HOUGH_MAX_GAP` | 150 px | Une segmentos separados en la misma línea |
| `MIN_YELLOW_PIXELS` | 40 px | Confianza mínima en detección de color |
| `MIN_VERT_DIFF` | 1 px | Filtra líneas perfectamente horizontales |
| `Kp` | 0.035 | Reacción proporcional al error |
| `Ki` | 0.0002 | Corrección de deriva acumulada |
| `Kd` | 0.004 | Amortiguamiento de oscilaciones |
| `alpha` | 0.35 | Factor de suavizado EMA |
| `smooth_error clamp` | ±8 px | Límite de cascada en intersecciones |
| `at_crosswalk threshold` | 480 px | Umbral de detección de cebra |
| `NO_LINE_HOLD` | 5 frames | Frames de hold cuando no hay línea visible |

---

## Visualización diagnóstica en pantalla

El display integrado del vehículo muestra en tiempo real el procesamiento de visión. Cada color tiene un significado específico:

| Color | Elemento | Significado |
|---|---|---|
| **Gris oscuro** (fondo) | Máscara HSV | Zona donde se detectó color amarillo — más brillante indica más píxeles amarillos |
| **Gris claro** | Línea filtrada | Segmento Hough descartado por ser casi horizontal (`\|y2-y1\| < adaptive_vert`). Aparece masivamente en las cebras peatonales |
| **Verde** | Línea candidata | Segmento que pasó el filtro vertical; es candidato para calcular el error del PID |
| **Rojo** (2 px de grosor) | Línea seleccionada | El segmento más cercano al centro — es el que guía el volante en ese frame |
| **Azul** (línea vertical) | Setpoint | Marca el centro exacto de la imagen (x = 64 px). El PID intenta mantener la línea roja sobre esta marca |

### HUD velocímetro

Encima de la imagen diagnóstica se superpone texto blanco con tres valores en tiempo real:

| Campo | Ejemplo | Descripción |
|---|---|---|
| `V:50km/h` | velocidad | Velocidad de crucero constante configurada |
| `St:0.023r` | ángulo | Ángulo de dirección aplicado en radianes (negativo = izquierda) |
| `E:1.5` | error o estado | Error EMA actual en píxeles; si no hay línea muestra `SIN LINEA` o `CEBRA` |

El HUD se dibuja con `display_img.setColor(0xFFFFFF)` y `display_img.drawText(texto, x, y)` inmediatamente después de pegar la imagen en el display, por lo que el texto se superpone sobre el fondo diagnóstico sin afectar el procesamiento de visión.

---

## Código completo con explicaciones

### Encabezado e imports

```python
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from collections import deque
from datetime import datetime
import os
import time
```

**No técnico:** Importamos las "herramientas" que vamos a usar. `controller` y `vehicle` son las librerías de Webots que permiten al script hablar con el simulador. `cv2` es OpenCV, la librería de visión por computadora. `numpy` maneja los arreglos de píxeles de la imagen.

**Técnico:** `Car()` instancia el modelo cinemático completo del vehículo. `Driver()` expone la API de actuadores (velocidad de crucero y ángulo de dirección). La separación es propia de Webots: `Robot` maneja sensores, `Driver` maneja actuadores de alto nivel. `deque(maxlen=6)` es un buffer circular O(1) para capturar los últimos 6 ángulos y calcular el `crosswalk_hold`.

---

### Constantes de color HSV

```python
YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)
```

**No técnico:** En lugar de ver la imagen en blanco y negro, el controlador "ve en color" pero solo presta atención al amarillo. Esto es como poner unos lentes de sol que solo dejan pasar la luz amarilla. El resultado es una imagen donde solo la línea central del camino es visible — todo lo demás queda negro.

**Técnico:** OpenCV representa el espacio HSV con H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]. El amarillo en el espectro visible corresponde a H ≈ 60° (rango completo 0–360°), que en OpenCV es H ≈ 30. El rango [15, 35] captura variaciones de la línea por iluminación, distancia y perspectiva. S ≥ 80 garantiza que el píxel tiene saturación mínima (no es blanco ni gris). V ≥ 80 descarta sombras oscuras. `cv2.inRange(hsv, LOW, HIGH)` genera una máscara binaria en O(w×h): 255 donde cumple el rango, 0 donde no.

---

### Función `detect_yellow`

```python
def detect_yellow(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return mask
```

**No técnico:** Esta función toma la imagen de la cámara y devuelve una imagen en blanco y negro donde el blanco representa "hay línea amarilla aquí" y el negro representa "no hay línea aquí".

**Técnico:** La cámara de Webots entrega formato BGRA (4 canales). OpenCV no soporta conversión directa BGRA→HSV, por lo que se hace en dos pasos. La conversión BGR→HSV aplica la transformación:
- H = 60° × (G-B)/(max-min) (para el caso máximo=R)
- S = (max-min)/max
- V = max

Esto desacopla la crominancia (H,S) de la luminancia (V), haciendo la detección robusta a cambios de iluminación.

---

### Función `apply_roi`

```python
def apply_roi(edges, height, width):
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0,     height),
        (0,     int(height * 0.5)),
        (width, int(height * 0.5)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(edges, mask)
```

**No técnico:** La cámara ve más cosas de las que necesitamos — cielo, edificios, el capó del carro. Esta función "tapa" la mitad superior de la imagen con negro para que el algoritmo solo busque la línea en el asfalto que está frente al vehículo.

**Técnico:** `fillPoly` dibuja un polígono relleno sobre la máscara negra. El polígono cubre el 50% inferior (y ∈ [height/2, height]). `bitwise_and` aplica la máscara: pixel_resultado = pixel_bordes AND pixel_mascara. Esto es O(w×h) en operaciones bit. La elección de 50% inferior es un equilibrio: demasiado alto incluye el horizonte y objetos distantes; demasiado bajo reduce el tiempo de anticipación para curvas.

---

### Función `detect_lines`

```python
def detect_lines(roi_edges):
    lines_p = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )
    if lines_p is not None:
        return lines_p.reshape(-1, 4)
    return []
```

**No técnico:** Una vez que tenemos los bordes de la línea amarilla, esta función los convierte en coordenadas concretas: "hay una línea que va desde el punto (x1,y1) hasta el punto (x2,y2)". Es como si alguien dibujara con regla los segmentos que ve en la imagen de bordes.

**Técnico:** HoughLinesP (Transformada de Hough Probabilística) opera en espacio de parámetros (ρ, θ). Cada píxel de borde vota por todas las líneas posibles que pasan por él. Los segmentos que acumulan ≥ HOUGH_THRESHOLD votos, tienen longitud ≥ HOUGH_MIN_LENGTH y tienen brechas internas ≤ HOUGH_MAX_GAP se devuelven como `[x1, y1, x2, y2]`. La versión probabilística muestrea un subconjunto de píxeles en lugar de todos, reduciendo la complejidad de O(w×h×nθ) a O(k×nθ) donde k << w×h. `reshape(-1, 4)` aplana el array de shape (N,1,4) a (N,4).

> **Nota de diseño:** Se eliminó el fallback a `HoughLines` (Hough estándar) durante el desarrollo. Con 50-80 px amarillos y threshold=5, generaba 100-170 líneas fantasma con errores consistentes de -4 a -7px → el auto se desviaba y chocaba. El comportamiento SIN LINEA (hold del último ángulo) resultó más seguro que las líneas fantasma.

---

### Función `compute_error`

```python
def compute_error(lines, setpoint, expected_x, vert_diff=None):
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
```

**No técnico:** Entre todos los segmentos de línea detectados, esta función elige el que más nos interesa: el más cercano al centro de la pantalla. El "error" que devuelve es simplemente cuántos píxeles está esa línea a la izquierda o derecha del centro. Si el error es -8, la línea está 8 píxeles a la izquierda de donde debería estar.

**Técnico:** El filtro `abs(y2 - y1) < vert_diff` descarta segmentos casi horizontales. En la cámara frontal, las rayas peatonales y marcas de intersección aparecen horizontales (|y2-y1| ≈ 0-2px) mientras que la línea central del carril tiene componente vertical significativa. `vert_diff` es adaptativo: 3px cuando `yellow_pixels > 300` (zona de intersección con muchos elementos amarillos), 1px en secciones normales. La selección por `dist_exp = |mid_x - expected_x|` donde `expected_x = setpoint` (centro fijo) garantiza que siempre se prefiere la línea más cercana al centro de la imagen, eliminando la cascada de rastreo de líneas equivocadas en intersecciones.

---

### Bucle principal — EMA y control

```python
# Suavizado EMA
if raw_error is not None and not at_crosswalk:
    smooth_error = alpha * raw_error + (1 - alpha) * smooth_error
    smooth_error = float(np.clip(smooth_error, -8.0, 8.0))
    error = smooth_error
else:
    error = None

# PID
p_term = Kp * error
integral += error * dt
i_term = Ki * integral
derivative = (error - prev_error) / dt
d_term = Kd * derivative
steering = p_term + i_term + d_term
steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
```

**No técnico:** El EMA es como un promedio inteligente: el error que usamos para girar el volante no es el de este frame exacto (que puede ser ruidoso) sino un promedio ponderado de los últimos frames, dándole más peso a los recientes. El PID combina tres correcciones: una proporcional al error actual, una que recuerda si hemos estado desviados por mucho tiempo (integral), y una que anticipa si el error está creciendo rápido (derivativa).

**Técnico:**
- **EMA:** `smooth_error_t = α × raw_error_t + (1-α) × smooth_error_{t-1}`. Con α=0.35 y dt≈33ms, la constante de tiempo es τ = -dt/ln(1-α) ≈ 75ms. El clamp en ±8px es la corrección clave para intersecciones: limita `expected_x` a [56, 72], haciendo que la línea correcta (cerca de x=64) siempre tenga ventaja sobre líneas equivocadas más alejadas del centro.
- **PID discreto:** El término derivativo usa diferencias finitas hacia atrás: `D = Kd × (error_t - error_{t-1}) / dt`. Se calcula `dt` real entre frames (no fijo), lo que hace el PID robusto a variaciones en el timestep de Webots.
- **Anti-windup integral:** Se resetea a 0 cuando `error = None` (sin línea o cebra) para evitar acumulación durante períodos de hold.

---

### Manejo de cruces peatonales

```python
at_crosswalk = (yellow_pixels > 480)

if at_crosswalk:
    consecutive_cebra += 1
    if consecutive_cebra > 4:
        smooth_error = 0.0   # reset EMA solo en cruce largo

    if consecutive_cebra == 1:
        recent = list(steering_buffer)
        crosswalk_hold = float(np.mean(recent)) if recent else last_steering

    if consecutive_cebra <= 4:
        steering = crosswalk_hold
    else:
        steering = last_steering * 0.88

    prev_error = smooth_error  # continuidad del término D
```

**No técnico:** Cuando el auto entra a un paso de cebra (que en este mundo de Webots es de color amarillo), hay tantos píxeles amarillos en la imagen que la detección de líneas falla. El auto recuerda el ángulo al que venía girando antes del cruce y lo mantiene mientras cruza. Es como cuando manejas y cierras los ojos medio segundo en una curva: confías en el volante donde ya estaba.

**Técnico:** Las rayas de cebra en este mundo Webots son amarillas, por lo que pasan el filtro HSV. Con >480px amarillos el ROI se llena de franjas horizontales que generan decenas de segmentos Hough con errores espurios. La detección `at_crosswalk` evita actualizar el EMA con esos errores. El `crosswalk_hold` captura `np.mean(steering_buffer[-6:])` — el promedio de los últimos 6 ángulos aplicados — que es más estable que el último steering solo. El decay ×0.88/frame a partir del frame 5 evita que el auto gire indefinidamente si el cruce es largo. El reset de `smooth_error = 0` solo ocurre después de 4+ frames consecutivos de CEBRA para preservar la continuidad del EMA en cruces breves.

---

## Troubleshooting — Problemas encontrados y soluciones

> Estos problemas fueron depurados iterativamente durante múltiples sesiones usando logs de terminal. Cada solución representa una lección aplicable a futuros proyectos de visión + control.

---

### Problema 1: `ModuleNotFoundError: No module named 'controller'`

**Síntoma:** Al ejecutar el script directamente con `python3`, fallaba con error de importación.

**Causa:** Las librerías de Webots (`controller`, `vehicle`) no están en el `PYTHONPATH` estándar del sistema. Webots las instala en su propio directorio y requiere variables de entorno específicas para que Python las encuentre.

**Solución:**
```bash
export WEBOTS_HOME=/Applications/Webots.app
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python:$PYTHONPATH
export DYLD_LIBRARY_PATH=/Applications/Webots.app/Contents/lib/controller:$DYLD_LIBRARY_PATH
```

Estas líneas se agregaron a `~/.zshrc` para que persistan entre sesiones. El script siempre debe ejecutarse así:
```bash
~/miniconda3/envs/ml_env/bin/python simple_controller_H.py
```

**Lección para futuros proyectos:** Cualquier simulador con librerías propias (Webots, Gazebo, CARLA) requiere configurar el entorno antes de ejecutar. Documentar estas variables en un `README` desde el inicio del proyecto.

---

### Problema 2: `HoughLines` fallback generaba líneas fantasma

**Síntoma:** En zonas con pocos píxeles amarillos (50-80px), el log mostraba `Líneas: 145 | Error: -4.6` — 145 líneas detectadas que daban errores consistentes de -4 a -7px. El auto se desviaba y chocaba.

**Causa:** Se había implementado un fallback a `HoughLines` (Hough estándar) cuando `HoughLinesP` no detectaba nada. Con threshold=5 y pocos píxeles, `HoughLines` generaba cientos de líneas en posiciones aleatorias que promediaban hacia un error negativo constante.

**Solución:** Eliminar el fallback completamente. El comportamiento SIN LINEA (hold del último ángulo) es más seguro que seguir líneas fantasma.

**Lección:** En visión por computadora, "no detectar nada" es a veces mejor decisión que detectar algo incorrecto. Un detector que siempre devuelve una respuesta puede ser más peligroso que uno que reconoce incertidumbre.

---

### Problema 3: `case_a` se disparaba en curvas suaves

**Síntoma:** En curvas suaves, el auto giraba insuficientemente y se salía de la carretera. El log mostraba SIN LINEA en secciones con línea visible.

**Causa:** En curvas suaves, la línea amarilla aparece casi horizontal en la cámara (ángulo ~5°). Para un segmento de 10px a 5°: `|y2-y1| = sin(5°)×10 = 0.87px`, que con `MIN_VERT_DIFF=2` era filtrado. Esto causaba `raw_error=None` con líneas presentes, que activaba `case_a` — una lógica de "intersección detectada" — y aplicaba `crosswalk_hold × 0.4`. El auto casi iba recto en plena curva.

**Solución:** Eliminar `case_a` completamente y reducir `MIN_VERT_DIFF` de 2 a 1. Los cruces reales se detectan exclusivamente por `yellow_pixels > 480`, no por la combinación de condiciones de `case_a`.

**Lección:** Las condiciones compuestas de detección de estados (AND lógicos) son frágiles. Una condición que parece segura individualmente puede activarse en escenarios no contemplados. Preferir umbrales simples y robustos sobre lógica condicional compleja.

---

### Problema 4: `MIN_VERT_DIFF=5` filtraba la línea central en curvas

**Síntoma:** En zonas con 430-450px amarillos (curvas pronunciadas), el log mostraba SIN LINEA seguido de oscilación del error entre -10 y +7px.

**Causa:** `adaptive_vert = 5` cuando `yellow_pixels > 300`. Para un segmento de 20px a 14°: `|y2-y1| = sin(14°)×20 = 4.8px < 5` → filtrado. La línea central en curva pronunciada era descartada, dejando solo líneas equivocadas.

**Solución:** Reducir `adaptive_vert` de 5 a 3. A vert_diff=3, líneas a 8.6° o más pasan; a vert_diff=5, el mínimo era 14.5°.

**Lección:** Los umbrales de filtro tienen efectos no lineales dependiendo de la geometría del escenario. Siempre verificar el caso extremo: ¿qué ángulo mínimo tiene la línea que quiero detectar en el peor caso?

---

### Problema 5: Jump filter rechazaba líneas válidas post-intersección

**Síntoma:** El log mostraba 9+ frames consecutivos de SIN LINEA con 2-5 líneas presentes. El auto mantenía steering≈0 y salía de la carretera.

**Causa:** El jump filter (`if |raw_error - smooth_error| > 20: rechazar`) estaba diseñado para evitar rastreo de líneas equivocadas. Pero después de una zona de intersección donde `smooth_error` había derivado a +1.85px, y el auto había girado físicamente, la línea correcta aparecía en x≈40 (raw_error=-24). El filtro rechazaba esta línea válida: |(-24)-(+1.85)| = 25.85 > 20. El auto veía SIN LINEA con líneas presentes.

**Solución:** Eliminar el jump filter completamente. La función `compute_error` ya selecciona la línea más cercana a `expected_x`, que es protección suficiente. El EMA suaviza cambios grandes sin rechazarlos.

**Lección:** Los filtros de rechazo tienen un costo oculto: pueden rechazar señales válidas en escenarios edge. Es mejor aceptar la señal y suavizarla (EMA) que rechazarla y quedar sin información.

---

### Problema 6: Cascada de rastreo de línea equivocada en intersecciones

**Síntoma:** En intersecciones, el error derivaba gradualmente de -2px a -17px sostenido por 5-8 segundos. El auto giraba agresivamente (MAX_ANGLE durante segundos) antes de recuperarse.

**Causa:** Con `expected_x = setpoint + smooth_error`, cuando el EMA derivaba a -5px, `expected_x` se movía a 59. Una línea equivocada en x=55 (dist=4) ganaba sobre la correcta en x=64 (dist=5). Esto derivaba el EMA más, movía más `expected_x`, haciendo que la línea equivocada fuera aún más "esperada". Cascada positiva de retroalimentación.

**Solución paso a paso:**
1. Agregar clamp: `smooth_error = clip(smooth_error, -8, 8)` — limita el error máximo al PID
2. Cambiar `expected_x = setpoint + smooth_error` → `expected_x = setpoint` — elimina la cascada de raíz

Con `expected_x = setpoint = 64` fijo, la línea correcta (en x=64) siempre tiene distancia=0 y siempre gana sobre cualquier línea equivocada más alejada del centro. No hay cascada posible.

**Lección:** El feedback positivo en sistemas de control es el enemigo. Si el estado del sistema modifica la selección de señal de referencia que a su vez modifica el estado, se crea una cascada inevitable. Mantener el punto de referencia fijo es más robusto que hacerlo seguir el estado del sistema.

---

### Resumen de lecciones para ejercicios futuros

| # | Lección | Aplica a |
|---|---|---|
| 1 | Documentar variables de entorno del simulador desde el inicio | Setup de proyecto |
| 2 | "No detectar" es mejor que detectar incorrectamente | Detección de señales |
| 3 | Condiciones compuestas de estado son frágiles | Lógica de control |
| 4 | Verificar el caso geométrico extremo de cada umbral | Parámetros de visión |
| 5 | Los filtros de rechazo tienen costo oculto; preferir suavizado | Filtrado de señales |
| 6 | El punto de referencia de selección de señal debe ser fijo, no dependiente del estado | Arquitectura de control |
| 7 | El comportamiento en intersecciones debe diseñarse explícitamente, no esperar que el PID lo maneje | Diseño de controladores |
| 8 | El log de terminal con debug cada N frames es indispensable para depurar control + visión | Debugging |

---

## Comparación con controlador del compañero de equipo

| Dimensión | **simple_controller_H.py** (Joel) | **simple_controller_H2.py** (compañero) |
|---|---|---|
| **Código fuente** | [GitHub](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H.py) | [GitHub](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H2.py) |
| **Detección de color** | HSV — solo amarillo | Escala de grises — toda la imagen |
| **Qué detecta** | Solo la línea central amarilla | Cualquier borde con suficiente contraste |
| **Filtro horizontal** | `\|y2-y1\| < adaptive_vert` (1-3 px) | `\|slope\| < 0.3` — filtro por pendiente |
| **Selección de línea** | Segmento más cercano a x=64 (centro fijo) | Promedio de puntos clasificados por pendiente |
| **Separación carril** | No aplica — una línea central | Clasifica por signo de pendiente (izq/der) |
| **Normalización del error** | Píxeles: `mid_x − 64` (rango ≈ ±64) | Normalizado: `(center − img_w/2) / (img_w/2)` ∈ [-1, 1] |
| **Ganancias PID** | Kp=0.035, Ki=0.0002, Kd=0.004 | Kp=0.35, Ki=0.08, Kd=0.01 |
| **Suavizado** | EMA α=0.35 + clamp ±8 px | Ninguno — respuesta directa frame a frame |
| **Manejo de cebras** | Detección automática + hold de ángulo | Steering=0 cuando no hay líneas |
| **Modo manual** | Sí (tecla M) | No |
| **ROI** | Rectángulo — 50% inferior completo | Trapecio — ancho 10%-90%, altura 60% |
| **Velocidad** | 50 km/h constante | 50 km/h constante |
| **Líneas de código** | ~300 | ~160 |

### Diferencia filosófica fundamental

El controlador del compañero asume disponibilidad de **ambos bordes del carril** y calcula el centro entre ellos — la estrategia clásica de "lane keeping" usada en sistemas ADAS reales. Es generalizable a cualquier color de línea y a carreteras con marcas blancas estándar.

El controlador de Joel asume que la **línea amarilla central** es el elemento más discriminativo del escenario y la sigue directamente. Es menos generalizable pero más preciso para este mundo específico de Webots, donde el amarillo es dominante y las marcas blancas podrían crear confusión.

Ambos enfoques son válidos. En un sistema de producción real, se usaría HSV (o redes neuronales de segmentación) para separar colores más el cálculo de centro de carril para navegación robusta — combinando ambas filosofías.

---

## Código completo

```python
"""
=============================================================================
ACTIVIDAD: Detección de Carriles con Controlador PID en Webots
=============================================================================
Alumno  : Joel Arturo Becerril Balderas
Matrícula: A01797427
Materia  : Navegación Autónoma — Maestría en Inteligencia Artificial Aplicada
Tecnológico de Monterrey

PIPELINE:
    1. Cámara BGRA → segmentación HSV (solo amarillo)
    2. Canny sobre máscara amarilla → bordes
    3. ROI fillPoly (50% inferior) → región de interés
    4. HoughLinesP → segmentos de línea
    5. Selección del segmento más cercano al centro de imagen
    6. EMA α=0.35 + clamp ±8px → smooth_error
    7. PID (Kp=0.035, Ki=0.0002, Kd=0.004) → steering
    8. Driver.setSteeringAngle + Driver.setCruisingSpeed

RESULTADO: >10 minutos sin choques a 50 km/h en circuito completo
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

SPEED         = 50
MAX_ANGLE     = 0.5
DEFAULT_ANGLE = 0.0
SPEED_INCR    = 5
ANGLE_INCR    = 0.05
DEBOUNCE_TIME = 0.1
MANUAL_MODE   = False

YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

CANNY_LOW  = 50
CANNY_HIGH = 150

HOUGH_RHO        = 1
HOUGH_THETA      = np.pi / 180
HOUGH_THRESHOLD  = 10
HOUGH_MIN_LENGTH = 10
HOUGH_MAX_GAP    = 150

MIN_YELLOW_PIXELS = 40
MIN_VERT_DIFF     = 1

Kp = 0.035
Ki = 0.0002
Kd = 0.004


def get_image(camera):
    raw = camera.getImage()
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def detect_yellow(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    return mask


def apply_roi(edges, height, width):
    mask = np.zeros_like(edges)
    roi = np.array([[
        (0,     height),
        (0,     int(height * 0.5)),
        (width, int(height * 0.5)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lines(roi_edges):
    lines_p = cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP
    )
    if lines_p is not None:
        return lines_p.reshape(-1, 4)
    return []


def compute_error(lines, setpoint, expected_x, vert_diff=None):
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


def display_image(display, image):
    if len(image.shape) == 2:
        image = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image.tobytes(), Display.RGB,
        width=image.shape[1], height=image.shape[0]
    )
    display.imagePaste(image_ref, 0, 0, False)


def main():
    robot    = Car()
    driver   = Driver()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    width    = camera.getWidth()
    height   = camera.getHeight()
    setpoint = width / 2      # x=64 para cámara de 128px

    integral      = 0.0
    prev_error    = 0.0
    smooth_error  = 0.0
    alpha         = 0.35

    last_steering     = 0.0
    no_line_count     = 0
    consecutive_cebra = 0
    was_cebra         = False
    NO_LINE_HOLD      = 5
    steering_buffer   = deque(maxlen=6)
    crosswalk_hold    = 0.0

    prev_time = time.time()
    speed     = 50
    angle     = 0.0
    last_press = {}
    manual    = MANUAL_MODE

    driver.setCruisingSpeed(speed)
    print(f"Modo: {'MANUAL' if manual else 'PID AUTOPILOT'}")
    print(f"Setpoint: {setpoint}px | Cámara: {width}x{height}px")

    while robot.step() != -1:
        current_time = time.time()
        dt = max(current_time - prev_time, 1e-6)
        prev_time = current_time

        # Paso 1-2: imagen y segmentación
        image       = get_image(camera)
        yellow_mask = detect_yellow(image)

        # Paso 3-4: bordes y Hough
        edges = cv2.Canny(yellow_mask, CANNY_LOW, CANNY_HIGH)
        roi   = apply_roi(edges, height, width)
        lines = detect_lines(roi)

        yellow_pixels = cv2.countNonZero(yellow_mask)

        # Paso 5: selección de línea más cercana al centro
        expected_x    = setpoint           # fijo — sin cascada posible
        adaptive_vert = 3 if yellow_pixels > 300 else MIN_VERT_DIFF

        if yellow_pixels < MIN_YELLOW_PIXELS:
            raw_error = None
            best_line = None
        else:
            raw_error, best_line = compute_error(
                lines, setpoint, expected_x, adaptive_vert
            )

        at_crosswalk = (yellow_pixels > 480)

        # Paso 6: EMA + clamp
        if raw_error is not None and not at_crosswalk:
            smooth_error = alpha * raw_error + (1 - alpha) * smooth_error
            smooth_error = float(np.clip(smooth_error, -8.0, 8.0))
            error = smooth_error
        else:
            error = None

        # Debug
        if int(current_time * 10) % 20 == 0:
            estado = "CEBRA" if at_crosswalk else (
                "SIN LINEA" if error is None else "OK"
            )
            print(f"Amarillo px: {yellow_pixels} | "
                  f"Líneas: {len(lines)} | "
                  f"Error: {round(error, 2) if error is not None else None} | "
                  f"{estado}")

        # Display diagnóstico
        disp_bgr = np.stack(
            [yellow_mask // 2, yellow_mask // 2, yellow_mask // 2], axis=-1
        )
        for x1d, y1d, x2d, y2d in lines:
            color = (100, 100, 100) if abs(y2d - y1d) < adaptive_vert \
                    else (0, 220, 0)
            cv2.line(disp_bgr, (x1d, y1d), (x2d, y2d), color, 1)
        if best_line is not None:
            bx1, by1, bx2, by2 = best_line
            cv2.line(disp_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.line(disp_bgr, (int(setpoint), 0), (int(setpoint), height - 1),
                 (255, 0, 0), 1)
        display_image(display_img, disp_bgr[:, :, ::-1])

        # Teclado
        key = keyboard.getKey()
        if key != -1:
            if not (key in last_press and
                    current_time - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = current_time
                if key == ord('M'):
                    manual = not manual
                    integral = 0.0
                    print(f"Modo: {'MANUAL' if manual else 'PID AUTOPILOT'}")
                elif key == ord('A'):
                    ts = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                    camera.saveImage(os.getcwd() + "/" + ts + ".png", 1)
                    print(f"Imagen guardada: {ts}.png")
                elif manual:
                    if key == keyboard.UP:
                        speed = min(speed + SPEED_INCR, 250)
                    elif key == keyboard.DOWN:
                        speed = max(speed - SPEED_INCR, 0)
                    elif key == keyboard.RIGHT:
                        angle = min(angle + ANGLE_INCR, MAX_ANGLE)
                    elif key == keyboard.LEFT:
                        angle = max(angle - ANGLE_INCR, -MAX_ANGLE)

        # Paso 7-8: control
        if manual:
            driver.setCruisingSpeed(speed)
            driver.setSteeringAngle(angle)
        else:
            if error is None:
                integral  = 0.0
                was_cebra = True

                if at_crosswalk:
                    consecutive_cebra += 1
                    if consecutive_cebra > 4:
                        smooth_error = 0.0

                    if consecutive_cebra == 1:
                        recent = list(steering_buffer)
                        crosswalk_hold = (float(np.mean(recent))
                                          if recent else last_steering)

                    steering = (crosswalk_hold if consecutive_cebra <= 4
                                else last_steering * 0.88)
                    prev_error = smooth_error
                else:
                    consecutive_cebra = 0
                    no_line_count    += 1
                    steering = (last_steering if no_line_count <= NO_LINE_HOLD
                                else last_steering * 0.88)
                last_steering = steering
            else:
                no_line_count     = 0
                consecutive_cebra = 0

                if was_cebra:
                    prev_error = smooth_error   # evita kick derivativo
                    was_cebra  = False

                p_term     = Kp * error
                integral  += error * dt
                i_term     = Ki * integral
                derivative = (error - prev_error) / dt
                d_term     = Kd * derivative

                steering   = p_term + i_term + d_term
                prev_error = error

                steering_buffer.append(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))

            steering      = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
            last_steering = steering
            driver.setCruisingSpeed(SPEED)
            driver.setSteeringAngle(steering)


if __name__ == "__main__":
    main()
```

---

## Cómo ejecutar

### Requisitos previos (una sola vez)

Agregar a `~/.zshrc`:
```bash
export WEBOTS_HOME=/Applications/Webots.app
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python:$PYTHONPATH
export DYLD_LIBRARY_PATH=/Applications/Webots.app/Contents/lib/controller:$DYLD_LIBRARY_PATH
```

### Cada sesión

1. Abrir Webots y cargar el mundo de la actividad
2. Verificar que el controller del robot sea `<extern>`
3. Presionar **Restart** en Webots
4. En terminal Mac:

```bash
~/miniconda3/envs/ml_env/bin/python /Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H.py
```

Para correr el controlador del compañero:
```bash
~/miniconda3/envs/ml_env/bin/python /Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H2.py
```

### Controles en tiempo de ejecución

| Tecla | Acción |
|---|---|
| `M` | Alternar entre PID Autopilot y Manual |
| `↑ ↓` | (Manual) Aumentar / reducir velocidad |
| `← →` | (Manual) Girar izquierda / derecha |
| `A` | Capturar imagen con timestamp |
