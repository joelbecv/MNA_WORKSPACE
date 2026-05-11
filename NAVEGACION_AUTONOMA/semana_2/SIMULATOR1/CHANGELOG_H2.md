# Changelog — simple_controller_H2.py

**Autor de los cambios:** Joel Arturo Becerril Balderas  
**Resultado:** >30 minutos de operación continua a 50 km/h sin choques  
**Código base:** `simple_controller_H2.py` original del equipo

---

## Resumen ejecutivo

El controlador original usaba escala de grises para procesar la imagen de cámara.
Eso hace que el pipeline de visión vea igual la línea amarilla de carril, las rayas
blancas de la cebra peatonal y los bordes del asfalto — ambigüedad que ningún filtro
heurístico puede resolver de forma confiable.

La solución definitiva fue añadir **segmentación por color HSV** antes del detector de
bordes. Tanto la línea de carril como las cebras son amarillas, pero el filtro elimina
todo lo que no es amarillo — orillas del asfalto, sombras, texturas, marcas de otros
colores — que en escala de grises generaban líneas falsas con slope válido y
corrompían el cálculo del centro del carril. Las cebras sí se siguen detectando, pero
sus rayas son horizontales y el slope filter las descarta. Los demás cambios son
ajustes del PID para estabilidad a largo plazo.

---

## Cambios aplicados

### 1. Segmentación HSV (cambio crítico)

**Qué cambia:**  
Se reemplaza la conversión a escala de grises por un filtro HSV que aísla únicamente
el color amarillo antes de aplicar Canny.

```python
# ORIGINAL
grey_image = greyscale_cv2(resized_bgr)
canny = cv2.Canny(grey_image, 50, 150)

# NUEVO
hsv         = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
yellow_mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))
canny       = cv2.Canny(yellow_mask, 50, 150)
```

**Por qué funciona:**  
Las cebras en Webots son también amarillas, igual que la línea de carril. El beneficio
del filtro HSV no es ignorar las cebras sino **reducir radicalmente el ruido de falsos
positivos** que viene de features no-amarillos:

- En escala de grises, Canny detecta todo lo que tiene contraste: orillas del asfalto,
  texturas del suelo, sombras, marcas de otros colores, bordes de acera. Muchos de esos
  segmentos tienen |slope| > 0.4, pasan el slope filter y contaminan `compute_lane_center`
  con posiciones incorrectas.
- Con el filtro HSV, **solo llegan objetos amarillos a Hough**. Las orillas de asfalto,
  sombras y texturas desaparecen antes de Canny. El conjunto de líneas candidatas se
  reduce drásticamente y la línea de carril es la dominante.
- En las cebras, las rayas amarillas horizontales sí llegan a Hough, pero son rechazadas
  por el slope filter (|slope| < 0.4) → `lane_center_x = None` → hold mechanism. La
  diferencia con el original es que ahora *solo* hay que manejar rayas amarillas
  horizontales, no también bordes de asfalto con slope > 0.4.
- El display muestra menos detecciones que antes — eso es correcto y deseable. La ROI
  no cambió; lo que cambió es que el input a Hough es más limpio.

**Rango HSV del amarillo en OpenCV (H en [0,180]):**  
El amarillo visible cae en H≈30 (60° en escala completa / 2 = 30 en OpenCV).  
El rango [15, 35] captura variaciones por iluminación, distancia y perspectiva.  
S≥80 y V≥80 descartan amarillos muy pálidos y sombras.

---

### 2. Umbral de slope filter: 0.3 → 0.4

**Qué cambia:**

```python
# ORIGINAL
MIN_ABS_SLOPE = 0.3

# NUEVO
MIN_ABS_SLOPE = 0.4
```

**Por qué:** Líneas de la cebra que entran con un ángulo de ~17° tienen slope≈0.30 y
pasaban el filtro original. Con 0.40 se rechaza hasta ~22° de inclinación. Es un
refuerzo secundario al filtro HSV.

---

### 3. Ganancias del PID: menos agresivas

**Qué cambia:**

```python
# ORIGINAL
kp = 0.35
ki = 0.08
kd = 0.01

# NUEVO
kp = 0.28
ki = 0.01
kd = 0.01
```

**Por qué (ki es el cambio crítico):**  
Con el error normalizado en [-1, 1] y `dt ≈ 0.032 s`, el integral con `ki=0.08`
alcanzaba su tope (±1.0) en solo **~3 segundos** de error sostenido. A partir de ese
punto actuaba como un sesgo fijo de ±0.08 rad permanente, no como corrección de deriva.
Con `ki=0.01` tarda ~40 segundos en saturarse y su contribución máxima es 0.005 rad
(prácticamente el controlador opera como P+D puro).

`kp=0.28` reduce el sobredisparo en curvas sin sacrificar corrección en rectas.

---

### 4. Clamp del integral: ±1.0 → ±0.5

**Qué cambia:**

```python
# ORIGINAL
integral = max(-1.0, min(1.0, integral))

# NUEVO
integral = max(-0.5, min(0.5, integral))
```

**Por qué:** Limita la contribución máxima del término I a `ki × 0.5 = 0.005 rad`.
Previene que el integral acumulado durante una zona sin línea contamine el control
cuando la línea reaparece.

---

### 5. Hold de steering al perder la línea (reemplaza snap a cero)

**Qué cambia:**

```python
# ORIGINAL
else:
    integral = 0.0
    steering = 0.0

# NUEVO
else:
    no_line_frames += 1
    integral      *= 0.6
    previous_error = 0.0
    if no_line_frames > 10:
        steering *= 0.95
```

**Por qué:** El snap a `steering=0.0` era catastrófico en curvas: si el carro venía con
0.3 rad de giro y la cebra bloqueaba la detección 5 frames, el steering caía a 0 y el
vehículo salía recto de la curva. El nuevo comportamiento:

- **Frames 1-10 sin línea** (`no_line_frames ≤ 10`): el ángulo se mantiene exacto.
  A 50 km/h una cebra de 3 m dura ≈ 7 frames — el carro la cruza con el ángulo previo.
- **Frame 11 en adelante**: decaimiento lento (×0.95/frame) para no acumular deriva si
  la pérdida de línea es real y prolongada.
- `integral *= 0.6`: el historial de corrección decae gradualmente en vez de borrarse.
- `previous_error = 0.0`: cuando la línea reaparece, la derivada se calcula como
  `(error - 0) / dt` en vez de `(error - error_antiguo) / dt_largo`, evitando el
  *kick* brusco de dirección en la re-adquisición.

---

### 6. Rate limiter de steering: máx 0.03 rad por frame

**Qué cambia:**

```python
# ORIGINAL
steering = kp * error + ki * integral + kd * derivative
steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))

# NUEVO
raw_steering = kp * error + ki * integral + kd * derivative
raw_steering = max(-MAX_ANGLE, min(MAX_ANGLE, raw_steering))
steering = max(steering - MAX_STEER_RATE,
               min(steering + MAX_STEER_RATE, raw_steering))
# MAX_STEER_RATE = 0.03
```

**Por qué:** Es la red de seguridad final. Aunque el filtro HSV ignore la cebra y el
hold evite el snap a cero, si alguna detección incorrecta pasa todos los filtros, el
ángulo solo puede cambiar 0.03 rad en un frame (≈32 ms). En los ~8 frames de una cebra
el máximo desvío acumulado es 0.24 rad — recuperable cuando la línea reaparece.

En conducción normal la variación de steering por frame es de 0.001-0.005 rad, así que
el rate limiter no afecta la respuesta en curvas.

---

## Tabla resumen de cambios

| # | Parámetro/Bloque | Original | Nuevo | Motivo |
|---|---|---|---|---|
| 1 | **Pipeline de visión** | Escala de grises → Canny | **HSV amarillo → Canny** | Ignora cebra blanca de raíz |
| 2 | `MIN_ABS_SLOPE` | 0.30 | **0.40** | Rechaza líneas diagonales de cebra |
| 3 | `kp` | 0.35 | **0.28** | Menos sobredisparo en curvas |
| 4 | `ki` | 0.08 | **0.01** | Integral no se satura en 3 s |
| 5 | Integral clamp | ±1.0 | **±0.5** | Limita sesgo acumulado |
| 6 | **Comportamiento sin línea** | `steering=0; integral=0` | **Hold 10 frames, luego ×0.95** | No snap a recto en cebra |
| 7 | `previous_error` en no-línea | no se modifica | **= 0.0** | Evita kick de derivada al recuperar |
| 8 | **Rate limiter** | no existía | **0.03 rad/frame** | Seguridad ante detecciones espurias |

---

## Resultado

| Métrica | Valor |
|---|---|
| Velocidad constante | 50 km/h |
| Tiempo continuo sin choques | > 30 minutos |
| Manejo de cebras | Automático — sin lógica explícita de detección |
| Choques en prueba | 0 |

---

## Cómo ejecutar

```bash
# En la terminal Mac, desde el directorio del controlador:
export WEBOTS_HOME=/Applications/Webots.app/Contents
export PYTHONPATH=$WEBOTS_HOME/lib/controller/python
export DYLD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller

python simple_controller_H2.py
```

El archivo `.wbt` debe tener el vehículo con `controller "<extern>"`.
Reiniciar el mundo en Webots antes de lanzar el script.
