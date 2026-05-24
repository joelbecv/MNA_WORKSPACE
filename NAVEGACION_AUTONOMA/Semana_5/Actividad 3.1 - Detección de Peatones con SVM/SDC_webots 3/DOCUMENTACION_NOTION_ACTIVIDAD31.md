# Actividad 3.1 — Sistema de Detección de Peatones para Vehículo Autónomo

> **Maestría en Inteligencia Artificial — Navegación Autónoma · Semana 5**
> Simulador: Webots R2025a · Vehículo: BMW X5 · Mundo: ciudad urbana

---

## Resumen Ejecutivo

Este proyecto implementa un sistema de seguridad activa para un vehículo autónomo que circula en entorno urbano. El auto debe resolver el reto central de la conducción autónoma: **moverse eficientemente sin poner en riesgo a los peatones**.

El sistema emplea **tres capas de inteligencia trabajando en paralelo**:

| Capa | Tecnología | Función | Velocidad |
|------|-----------|---------|-----------|
| 1 — Navegación | PID + Visión | Sigue la línea amarilla del carril | Cada frame (10 ms) |
| 2 — Detección de personas | SVM + HOG | Identifica siluetas humanas | Cada 100 ms |
| 3 — Detección de obstáculos | LiDAR Sick LMS 291 | Detecta cualquier objeto físico | Cada 30 ms |

**Resultado de negocio:** El sistema reduce simultáneamente los **falsos negativos** (no detectar un peatón real → colisión) y los **falsos positivos** (frenadas innecesarias → pérdida de eficiencia y confort de viaje).

---

## 1. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     BMW X5 — Webots R2025a                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────────┐   ┌────────────┐  │
│  │   CÁMARA     │   │   LIDAR Sick LMS 291  │   │  DISPLAY   │  │
│  │  frontal     │   │   180 rayos · 180°    │   │  200×150   │  │
│  └──────┬───────┘   └──────────┬────────────┘   └─────┬──────┘  │
│         │                      │                       │         │
│  ┌──────▼───────────────────────▼────────────────────┐ │         │
│  │              simple_controller_stv3.py             │ │         │
│  │                                                    │ │         │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  │ │         │
│  │  │  PID carril  │  │  SVM + HOG   │  │  LiDAR   │  │ │         │
│  │  │  (CAPA 1)    │  │  (CAPA 2)    │  │ (CAPA 3) │  │ │         │
│  │  └──────┬──────┘  └──────┬───────┘  └────┬─────┘  │ │         │
│  │         │                │                │         │ │         │
│  │  ┌──────▼────────────────▼────────────────▼──────┐  │ │         │
│  │  │           LÓGICA DE AMENAZA (4 niveles)        │  │─┘         │
│  │  └──────────────────────────┬─────────────────────┘  │         │
│  │                             │                          │         │
│  │  ┌──────────────────────────▼─────────────────────┐  │         │
│  │  │      ACCIÓN: setCruisingSpeed + setBrake        │  │         │
│  │  └────────────────────────────────────────────────┘  │         │
│  └────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Capa 1 — Navegación: Seguimiento de Carril con PID

### 2.1 Filosofía

El auto solo "ve" el **color amarillo** de las líneas del carril. Ignorar el resto de la imagen (edificios, peatones, sombras) garantiza que nada desvíe el volante por error.

### 2.2 Pipeline de visión

```
Imagen BGRA (cámara)
    │
    ▼
Convertir a HSV
    │
    ▼
Máscara amarilla: H∈[15°,35°]  S∈[80,255]  V∈[80,255]
    │
    ▼
Canny (umbral 50/150) → bordes de las líneas
    │
    ▼
ROI trapezoidal → solo la zona del carril
    │
    ▼
HoughLinesP → segmentos de línea
    │
    ▼
Filtrar pendiente ≥ 0.6 → descartar franjas peatones horizontales
    │
    ▼
Separar líneas izq (pendiente < 0) / der (pendiente > 0)
    │
    ▼
Centro del carril = promedio(midpoints_izq + midpoints_der)
```

### 2.3 Controlador PID — la fórmula

El **error** es la desviación del centro del carril respecto al centro del display, normalizado a `[-1, 1]`:

```
error = (centro_detectado - centro_display) / centro_display
```

El **ángulo de dirección** se calcula con las tres componentes clásicas del PID:

```
u(t) = Kp·e(t)  +  Ki·∫e(t)dt  +  Kd·de(t)/dt

Kp = 0.28   → Proporcional: corrige el error actual
Ki = 0.01   → Integral:     elimina error estacionario acumulado
Kd = 0.01   → Derivativo:   amortigua oscilaciones del volante
```

**Suavizado de dirección:** Para evitar movimientos bruscos del volante (mala experiencia de conducción), el ángulo no puede cambiar más de `±0.03 rad` por frame:

```python
steering = clamp(prev_steering ± MAX_STEER_RATE, raw_pid_output)
```

**Justificación de negocio de los parámetros:**

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `CRUISE_SPEED` | 30 km/h | Velocidad urbana estándar. A 30 km/h el sistema tiene ~1.3 s para reaccionar en 10 m |
| `Kp = 0.28` | Empírico | Suficientemente agresivo para curvas cerradas, sin sobre-corregir en rectas |
| `MIN_ABS_SLOPE = 0.6` | Geométrico | Las franjas de cruce peatonal son casi horizontales (pendiente ~0.1) vs líneas de carril diagonales (0.7–1.5) |

---

## 3. Capa 2 — Detección de Personas: SVM + HOG

### 3.1 ¿Qué es HOG y por qué funciona para detectar personas?

**HOG (Histogram of Oriented Gradients)** fue introducido por Dalal & Triggs en CVPR 2005 y sigue siendo el estándar para detectar peatones en tiempo real.

La idea clave: **la forma de una persona se puede describir mejor por la distribución de los bordes (gradientes) que por el color o la textura**.

```
Ventana 64×128 px
    │
    ▼
Dividir en celdas de 16×16 px → 4×8 = 32 celdas
    │
    ▼
Calcular gradiente (dx, dy) en cada píxel
    │
    ▼
Histograma de orientaciones (11 bins: 0°–180°) por celda
    │
    ▼
Agrupar celdas en bloques 2×2 → normalizar contra variaciones de iluminación
    │
    ▼
Bloques posibles: (4-2+1) × (8-2+1) = 3 × 7 = 21 bloques
    │
    ▼
21 bloques × 2×2 celdas × 11 orientaciones = 924 valores ← descriptor final
```

**924 números** representan completamente la distribución de bordes en una ventana. Si el patrón se parece al de una persona real, la SVM lo detecta.

### 3.2 ¿Por qué SVM y no una red neuronal?

| Criterio | SVM + HOG | CNN |
|---------|----------|-----|
| Velocidad en CPU | ✅ ~8 ms por ventana | ❌ ~50-200 ms |
| Datos de entrenamiento | ✅ 2,752 imágenes bastan | ❌ Necesita miles o millones |
| Interpretabilidad | ✅ Hiperplano claro | ❌ Caja negra |
| Robustez a variaciones | ✅ Invariante a iluminación (L2-norm) | Variable |

Para un simulador en tiempo real con CPU, SVM+HOG es la opción correcta.

### 3.3 El modelo entrenado

- **Dataset:** INRIA Person Dataset — 2,752 imágenes (1,239 positivas + 1,513 negativas)
- **Pipeline sklearn:** `StandardScaler → SVC(kernel='rbf', C=1.0, gamma='scale')`
- **Archivo:** `pedestrian_svm.joblib`

### 3.4 Sliding Window — cómo barre la imagen

```
Imagen completa (cámara BMW)
    │
    ▼
ROI vertical: 59%–85% del alto de la imagen
(filtra cielo, edificios y el capó del auto)
    │
    ▼
Escala 4× → ROI crece de ~34px a ~136px de alto
(HOG necesita mínimo 128px de alto para la ventana estándar)
    │
    ▼
Barrido horizontal: 30%–70% del ancho
(los peatones peligrosos están en el carril, no en las banquetas)
    │
    ▼
Para cada posición (paso=32px):
    ├── Pre-filtro: ¿>15% amarillo? → saltar (no es persona, es carril)
    ├── Calcular HOG (924 valores)
    └── SVM.decision_function() → score
         ├── score ≥ 0.25 → hit positivo
         └── score < 0.25 → fondo
    │
    ▼
≥ 1 hit en esta pasada → detección = True
```

### 3.5 El umbral calibrado: 0.25

**Domain gap (diferencia de dominio):** El modelo fue entrenado con personas reales (fotos INRIA) pero se ejecuta en un mundo virtual (Webots). Las personas de Webots son modelos 3D con texturas diferentes a fotografías reales.

Efecto práctico en los scores:

| Contexto | Score típico | Interpretación |
|---------|-------------|----------------|
| INRIA (entrenamiento) | 0.8 – 1.5 | El modelo "conoce" estas imágenes |
| Fondo en Webots | 0.06 – 0.19 | Ruido esperado |
| Peatón en Webots | 0.25 – 0.70 | Score reducido por domain gap |

**Umbral en 0.25:** suficientemente bajo para capturar peatones virtuales sin dispararse con el fondo.

### 3.6 Sistema de confirmación — CONFIRM_N = 2

Un solo frame positivo podría ser un falso positivo (sombra, cruce). El sistema requiere **2 scans positivos consecutivos** antes de frenar:

```
scan 1: pos_streak = 1  → no frena todavía
scan 2: pos_streak = 2  → FRENO + brake_hold = 100 frames
```

Para liberar el freno: **4 scans negativos consecutivos** DESPUÉS de que expire el hold de 100 frames (~1 segundo). Esto evita que una oscilación temporal del score libere el freno prematuramente.

**Costo de un falso negativo (no detectar):** Colisión → inaceptable.
**Costo de un falso positivo (frenada innecesaria):** Incomodidad → tolerable.
Por eso, el sistema está calibrado para ser conservador.

---

## 4. Capa 3 — Detección de Obstáculos: LiDAR

### 4.1 El sensor Sick LMS 291

El Sick LMS 291 es un escáner láser industrial estándar en vehículos autónomos reales. En Webots:

| Característica | Valor |
|----------------|-------|
| FOV | 180° (π rad) |
| Rayos horizontales | 180 |
| Resolución angular | 1° por rayo |
| Rango | 0 – 80 m |
| Plano de escaneo | Horizontal (a la altura del sensor) |

### 4.2 Geometría del cono de detección

Con `LIDAR_CONE_DEG = 61°` usamos solo los **61 rayos centrales** del sensor:

```
  Sensor (centro inferior)
         │
         │ eje del auto
        / \
       /   \
      /     \
     /  61°  \
    /─────────\
   ◄── 30.5° ──►  ← mitad del cono

A 8m de distancia:
  Alcance lateral = 8 × tan(30.5°) = 8 × 0.59 = 4.7 m

Una carretera urbana tiene ~3.5 m de ancho por carril.
El cono cubre ±4.7 m → detecta objetos en el carril propio y parte del adyacente.
```

**¿Por qué no usar los 180°?**
Los 180° incluyen postes de luz, banquetas, edificios y señales a los lados. El cono de 61° solo apunta hacia el frente del vehículo donde están los obstáculos relevantes.

### 4.3 Lógica de 4 niveles

```
Cada 3 frames (~30 ms):

NIVEL 4 — Alerta normal con confirmación:
  lidar_streak ≥ 1  AND  threat == 'none'
  → threat = 'objeto'  brake_hold = 100 frames

NIVEL 3 — Emergencia < 8 m:
  dist < 8 m  AND  threat == 'none'
  → threat = 'objeto'  brake_hold = 100 frames
  (sin necesidad de confirmación previa)

NIVEL 2 — Override total < 5 m:
  dist < 5 m  (SIEMPRE, sin importar estado anterior)
  → threat = 'objeto'  (reinicia brake_hold solo al transicionar)

NIVEL 1 — SVM confirma peatón:
  pos_streak ≥ 2
  → threat = 'pedestrian'  brake_hold = 100 frames
```

**Justificación de negocio:**

| Distancia | Tiempo disponible a 30 km/h | Acción |
|-----------|---------------------------|--------|
| 8 m | 0.96 s | Alerta — frena con confirmación |
| 5 m | 0.60 s | Override — frena inmediatamente |
| 3 m (típico freno) | 0.36 s | Ya está frenando |

A 30 km/h con frenado completo, la distancia de parada es ~8 m. El sistema activa el freno a exactamente esa distancia.

### 4.4 Limitación de simulación conocida

Los PROTO de peatones en Webots **no tienen `boundingObject`** visible en el plano horizontal, por lo que el LiDAR no los detecta directamente. El LiDAR detecta objetos físicos como conos de tráfico y obstáculos estáticos. **La detección de peatones recae en el SVM.**

Esta es una limitación del simulador, no del sistema real donde el LiDAR detectaría correctamente las piernas de los peatones.

---

## 5. Visualización en el Display

El display de 200×150 px muestra en tiempo real el estado del sistema:

```
┌─────────────────────────────┐
│B PID OK                     │  ← B=barra LiDAR (verde/rojo)
│  V:30 St:0.02               │  ← velocidad y ángulo de dirección
│  SVM:0.180(>0.25)           │  ← score SVM vs umbral
│  pos:1/2                    │  ← racha positiva hacia confirmación
│  LiDAR:---  ±30.5°          │  ← distancia y cobertura del cono
│  hold:0                     │  ← frames restantes de freno
│                             │
│    ◄ triángulo verde/rojo ► │  ← cono LiDAR proyectado
│                             │
│  ┌──── ROI SVM ────────┐    │  ← rectángulo cian (59%–85%)
│  │  │ barrido  │       │    │  ← líneas naranjas (30%–70%)
│  │  └──────────┘       │    │
│  └─────────────────────┘    │
│                             │
│  ─── líneas amarillas PID ──│  ← segmentos Hough del carril
└─────────────────────────────┘
```

| Elemento visual | Color | Significado |
|----------------|-------|-------------|
| Texto superior | Verde = libre · Naranja = objeto · Rojo = peatón | Estado del sistema |
| Barra lateral | Verde / Rojo | LiDAR libre / alerta |
| Triángulo | Verde / Rojo | Cono activo del LiDAR |
| Rectángulo cian | Cian | ROI del SVM (zona de análisis) |
| Líneas naranjas | Naranja | Límites horizontales del barrido |
| Rectángulo rojo | Rojo | Ventana SVM con mayor score |
| Líneas amarillas | Amarillo | Segmentos Hough detectados (PID) |

---

## 6. Flujo de Decisión — Diagrama Completo

```
┌──── CADA FRAME (10 ms) ────────────────────────────────────────────┐
│                                                                      │
│  Captura imagen ──► PID: calcula ángulo de dirección               │
│                                                                      │
│  Cada 3 frames:    LiDAR lee cono central ──► lidar_streak++?       │
│                                                                      │
│  Cada 10 frames:   SVM sliding window ──► pos_streak++ o neg--     │
│                                                                      │
│  ┌─── Lógica de amenaza ──────────────────────────────────────┐    │
│  │                                                              │    │
│  │  pos_streak ≥ 2  ──► threat = 'pedestrian'  hold=100       │    │
│  │  neg_streak ≥ 4 AND hold=0  ──► threat = 'none'            │    │
│  │                                                              │    │
│  │  dist < 8 m AND threat='none'  ──► threat = 'objeto' h=100 │    │
│  │  dist < 5 m (siempre)         ──► threat = 'objeto'        │    │
│  │                                                              │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─── Acción de control ─────────────────────────────────────┐    │
│  │                                                              │    │
│  │  threat != 'none'  →  setCruisingSpeed(0)                  │    │
│  │                        setBrakeIntensity(1.0)               │    │
│  │                        setHazardFlashers(objeto==True)     │    │
│  │                                                              │    │
│  │  threat == 'none'  →  setCruisingSpeed(30)                 │    │
│  │                        setBrakeIntensity(0.0)               │    │
│  │                        setSteeringAngle(PID_output)        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7. Parámetros Clave — Tabla de Referencia

### Navegación PID

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `CRUISE_SPEED` | 30 km/h | Velocidad urbana estándar |
| `Kp` | 0.28 | Corrección proporcional calibrada empíricamente |
| `Ki` | 0.01 | Corrección de deriva lenta |
| `Kd` | 0.01 | Amortiguación de oscilaciones |
| `MIN_ABS_SLOPE` | 0.6 | Filtra franjas de cruce peatonal (pendiente ~0.1) |
| `MAX_STEER_RATE` | 0.03 rad/frame | Suaviza movimientos del volante |

### SVM + HOG

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `SVM_THRESHOLD` | 0.25 | Calibrado para domain gap INRIA→Webots |
| `CONFIRM_N` | 2 | 2 scans consecutivos para confirmar peatón |
| `RELEASE_N` | 4 | 4 scans negativos para liberar el freno |
| `HOLD_FRAMES` | 100 | ~1 segundo de freno garantizado |
| `DETECT_EVERY` | 10 | SVM cada 100 ms (balance velocidad/carga CPU) |
| `HOG_WIN_W × H` | 64×128 px | Estándar Dalal & Triggs CVPR 2005 |
| `SLIDE_STEP` | 32 px | Paso del barrido = mitad de la ventana |
| ROI vertical | 59%–85% | Zona donde aparecen peatones a distancia media |
| ROI horizontal | 30%–70% | Solo el carril, no las banquetas |

### LiDAR

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `LIDAR_CONE_DEG` | 61° | Cubre ±4.7 m laterales a 8 m (ancho de carril ~3.5 m) |
| `LIDAR_MAX_M` | 8 m | Distancia de parada a 30 km/h |
| `LIDAR_OVERRIDE_M` | 5 m | Override incondicional a distancia crítica |
| `LIDAR_EVERY` | 3 frames | Lectura cada ~30 ms (estable en macOS Rosetta) |
| `LIDAR_CONFIRM` | 1 | 1 lectura basta (ya filtramos inf/nan) |

---

## 8. Cómo Ejecutar la Simulación

### Requisitos

- Webots R2025a instalado
- Python ≥ 3.9 con: `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `joblib`
- macOS Apple Silicon: **Rosetta 2 activo** (`Finder → Webots.app → Cmd+I → "Abrir con Rosetta"`)

### Pasos

1. Abrir Webots con Rosetta 2 (Apple Silicon) o normalmente (Intel/Windows/Linux)
2. `File → Open World` → seleccionar `worlds/city_2025a_lidar.wbt`
3. Verificar que el BMW tenga asignado el controlador `simple_controller_stv3`
4. Presionar el botón de Play ▶
5. El controlador carga automáticamente `pedestrian_svm.joblib` desde la raíz del proyecto
6. El display de 200×150 px muestra el estado en tiempo real

### Controles

| Tecla | Acción |
|-------|--------|
| `A` | Captura screenshot de la cámara (para debugging o recolección de datos) |

### Salida en consola (ejemplo)

```
[OK] Modelo SVM cargado
[LiDAR] FOV=3.14 rad  rayos=180  cono=±30.5° (61 rayos activos)  max=8.0m
Controlador listo — PID + SVM + LiDAR
[LIDAR] f=00003 dist=--- alert=no streak=0/1 threat=none
[SVM]   f=00010 wins=8 hits=0/1 score=0.152 thresh=0.25 pos=0/2 neg=1/4 lidar=ok(---) threat=none
[LIDAR] f=00006 dist=7.2m alert=SI streak=1/1 threat=none
[SVM]   f=00020 wins=8 hits=1/1 score=0.312 thresh=0.25 pos=1/2 neg=0/4 lidar=ALERTA(7.2m) threat=objeto
```

---

## 9. Decisiones de Diseño

### ¿Por qué LiDAR + SVM en lugar de solo SVM?

El SVM tiene un **delay de 100 ms** (DETECT_EVERY=10 frames). A 30 km/h, el auto recorre **83 cm en 100 ms**. Si el peatón aparece de repente a 2 m, el SVM podría no alcanzar a confirmar antes del impacto.

El LiDAR detecta **cualquier objeto físico** cada 30 ms sin necesidad de clasificarlo. Es la primera línea de defensa para obstáculos de última hora.

### ¿Por qué el pre-filtro amarillo en las ventanas SVM?

Sin el filtro, las franjas horizontales de los cruces peatones y las líneas del carril generaban scores de 0.20–0.28 (cerca del umbral). Con el filtro, saltamos directamente esas ventanas sin ejecutar HOG — reduciendo la carga computacional y eliminando falsos positivos.

### ¿Por qué solo canal amarillo en el PID (no gris)?

La versión anterior ejecutaba Canny sobre la imagen en escala de grises. Las siluetas oscuras de los peatones generaban bordes fuertes que el PID interpretaba como líneas del carril, desviando el volante. Con Canny solo sobre la máscara amarilla, solo los bordes de la línea amarilla afectan la dirección.

### ¿Por qué `engineSound ""`?

El PROTO BmwX5 de Webots intentaba reproducir un archivo de audio que no existía en la instalación. Cada frame, el motor de simulación generaba un warning de audio que se acumulaba en un bucle bloqueante — causa principal del "beachball" (freeze) en macOS. La solución es agregar `engineSound ""` en el nodo BmwX5 del archivo `.wbt`.

---

## 10. Resultados Esperados

| Escenario | Comportamiento esperado |
|-----------|------------------------|
| Carril libre | Auto a 30 km/h, correcciones suaves del volante |
| Peatón en carril (detección SVM) | Freno completo en <200 ms · Texto "PEATON" en display |
| Obstáculo físico < 8 m (LiDAR) | Freno completo + intermitentes · Texto "OBJETO" |
| Falsa alarma momentánea | Hold de 100 frames (~1 s) antes de liberar |
| Curva con líneas paralelas | Pre-filtro amarillo descarta ventanas → sin falsos positivos |

---

## Referencias

- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR 2005.
- INRIA Person Dataset: http://pascal.inrialpes.fr/data/human/
- Webots R2025a — Driver API: https://cyberbotics.com/doc/automobile/driver-library
- Sick LMS 291 — PROTO Webots: https://webots.cloud/run?version=R2025a&url=github.com/cyberbotics/webots/blob/released/projects/devices/sick/protos/SickLms291.proto
- Issue macOS LiDAR Apple Silicon: https://github.com/cyberbotics/webots/issues/5282
