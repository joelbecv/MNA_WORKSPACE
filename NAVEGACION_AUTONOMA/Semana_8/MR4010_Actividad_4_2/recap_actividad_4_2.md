# Actividad 4.2 — Evasión de Obstáculos con Wall-Following
**MR4010 Navegación Autónoma · Semana 8**
**Equipo 25 · Joel Arturo Becerril Balderas (A01797427)**

---

## Objetivo

Programar un vehículo BMW en Webots para que, durante el seguimiento de carril con PID, detecte autobuses estacionados al frente, los evite rodeándolos por la derecha (wall-following), y recupere la orientación original para retomar el carril — todo de forma autónoma y sin intervención humana.

---

## Mundo: `city_2025a_4_2.wbt`

El mundo fue **proporcionado por el profesor tal cual** — no realizamos modificaciones al archivo `.wbt`. Describe lo que ya incluía para entender el entorno de trabajo:

### Vehículos (ya incluidos en el mundo)

4 autobuses estáticos (controller=`<none>`) colocados en cada costado de la ciudad, frente a sus respectivas paradas:

| Nombre       | Color visual   | Posición aproximada       | recognitionColor RGB               |
|--------------|----------------|---------------------------|------------------------------------|
| `vehicle(1)` | Azul marino    | (-18.5, -104.8)           | (0.031, 0.122, 0.420)              |
| `vehicle(2)` | Rojo           | (-105.1, -44.1) rot -90°  | (1.000, 0.000, 0.000)              |
| `vehicle(3)` | Lavanda        | (10.6, 105.0) rot 180°    | (0.863, 0.541, 0.867)              |
| `vehicle(4)` | Verde          | (105.3, 40.9) rot 90°     | (0.180, 0.761, 0.494)              |

4 `BusStop` frente a cada autobús.

### Sensores del BMW (ya incluidos en el mundo)

El mundo entregaba el BMW con sensores listos para la actividad:

| Sensor          | Posición local        | Función                                 |
|-----------------|-----------------------|-----------------------------------------|
| `ds_right_front`| (1.2, -1.0, -0.3)    | Detecta bus al costado delantero        |
| `ds_right_mid`  | (0.0, -1.0, -0.3)    | Detecta bus al costado central          |
| `ds_right_rear` | (-1.2, -1.0, -0.3)   | Confirma que el bus quedó atrás (≥4.8m) |

Todos apuntan en dirección -Y (lateral derecha), rango 0–5 m. También incluía cámara 128×128 con Recognition (maxRange=30m) y LiDAR Sick LMS 291.

---

## Controlador: Máquina de Estados

```
LINE_FOLLOW ──→ WALL_FOLLOW ──→ REORIENT ──→ RECENTER ──→ LINE_FOLLOW
                                                 ↑              |
                                                 └──────────────┘ (si no se halla línea)
```

### Estado 0 — `LINE_FOLLOW`
Seguimiento de carril amarillo con PID (portado de Actividad 2.1).
- Canny + HoughLinesP sobre ROI trapezoidal
- PID: Kp=0.28, Ki=0.01, Kd=0.01
- Velocidad: **30 km/h**
- **Transición**: Recognition detecta bus (área ≥400 px²) **y** LiDAR frontal ≤14.5 m

### Estado 1 — `WALL_FOLLOW`
El vehículo rodea el autobús por la derecha manteniendo distancia fija.

- Velocidad: **15 km/h** (mitad de crucero — exigido por rúbrica)
- **Fase A** (abrirse paso): giro suave a la izquierda hasta que `ds_right_mid < DS_ENGAGE_DIST (4.5m)`
- **Fase B** (rodear): controlador P lateral → `steering = KP_WALL × (right_dist − WALL_TARGET)`
  - `WALL_TARGET = 2.9 m`, `KP_WALL = 0.10`
- **Transición**: `ds_right_rear > 4.8 m` → bus superado → guarda heading actual → pasa a REORIENT

### Estado 2 — `REORIENT`
Recupera la orientación original usando el giroscopio.

- Velocidad: **20 km/h**
- El giroscopio acumula el ángulo de yaw durante toda la simulación
- Al entrar en WALL_FOLLOW se guarda `saved_heading = gyro.getValues()[1]`
- Controlador P: `steering = KP_HEADING × (saved_heading − heading_actual)`
  - `KP_HEADING = 1.0`
- **Transición**: `|heading_error| < 0.08 rad (≈4.6°)` → pasa a RECENTER

### Estado 3 — `RECENTER`
Busca la línea amarilla y retoma el PID.

- Mismo pipeline de imagen que LINE_FOLLOW
- **Transición**: se detectan líneas válidas → vuelve a LINE_FOLLOW
- Si no se detectan líneas en 30 frames: avanza recto y vuelve a REORIENT

---

## Decisiones de calibración y por qué

### 1. Sector LiDAR lateral: 150–180 (no 135–165)
El sensor Sick LMS 291 tiene 181 rayos (índice 0 = izquierda, 90 = frente, 180 = derecha).
El sector **150–180** corresponde a 60°–90° del frente, que es la zona lateral pura.  
El sector original 135–165 (45°–75°) dejaba un ángulo mixto frente-lateral donde el bus no siempre aparecía cuando ya estaba al costado — se corrigió a 150–180 para wall-following confiable.

### 2. BUS_LIDAR_THRESH = 14.5 m (no 12 m)
El `Camera Recognition` pierde al autobús cuando está a menos de ~12 m (el objeto sale del FOV).  
Se subió el umbral a **14.5 m** para iniciar la evasión mientras el Recognition todavía confirma que es un bus, evitando arrancar la evasión ante falsos positivos del LiDAR.

### 3. Gyro index [1] (eje Y, no [2] eje Z)
`getValues()` retorna [roll_rate, pitch_rate, yaw_rate].  
En Webots con el BMW, el yaw (rotación en plano horizontal) cae en el índice **[1]** — se validó experimentalmente observando que [2] siempre era cero mientras el vehículo giraba.

### 4. BUS_CONFIRM_FRAMES = 1
Con colores de reconocimiento exactos (tolerancia ±0.05 por canal RGB) y `maxRange=30` en la cámara, **1 frame es suficiente** para confirmar detección sin falsos positivos. Incrementar a 3–5 frames causaba que el vehículo llegara demasiado cerca antes de evadir.

### 5. WALL_TARGET = 2.9 m
Calibrado experimentalmente: 2.5 m era demasiado justo (el BMW rozaba el bus), 3.5 m lo llevaba al carril contrario. **2.9 m** da separación segura manteniéndose en el carril derecho.

---

## Parámetros completos de referencia

```python
# Seguimiento de carril
SPEED_FOLLOW      = 30      # km/h
Kp, Ki, Kd        = 0.28, 0.01, 0.01

# Detección de bus
BUS_LIDAR_THRESH  = 14.5    # m — umbral LiDAR frontal
BUS_MIN_PX_AREA   = 400     # px² mínimos en Recognition
BUS_CONFIRM_FRAMES = 1      # frames consecutivos para confirmar

# Evasión (wall-following)
SPEED_EVADE       = 15      # km/h
WALL_TARGET       = 2.9     # m de separación lateral
KP_WALL           = 0.10
DS_CLEAR_DIST     = 4.8     # m — ds_right_rear para declarar bus superado
DS_ENGAGE_DIST    = 4.5     # m — ds_right_mid para pasar de fase A a B

# Recuperación de heading
SPEED_REORIENT    = 20      # km/h
KP_HEADING        = 1.0
HEADING_TOL       = 0.08    # rad ≈ 4.6°

# LiDAR sectores (índices de los 181 rayos del Sick LMS 291)
LIDAR_FRONT       = 70–110  # ±20° del frente
LIDAR_LATERAL     = 150–180 # costado derecho puro (60°–90°)
```

---

## Arquitectura de sensores

```
┌─────────────────────────────────────────────────────┐
│                    BMW X5 (Webots)                  │
│                                                     │
│  Cámara 128×128 px                                  │
│   └─ Recognition (maxRange=30m)  → detecta bus      │
│                                                     │
│  Sick LMS 291 (LiDAR 2D, 181 rayos, 180°)           │
│   ├─ Sector frontal (70-110)     → distancia bus    │
│   └─ Sector lateral (150-180)    → wall distance    │
│                                                     │
│  Giroscopio (yaw rate)                              │
│   └─ index [1]                   → heading acum.    │
│                                                     │
│  3× DistanceSensor (5m, derecha)                    │
│   ├─ ds_right_front              → bus al costado   │
│   ├─ ds_right_mid                → fase A→B trigger │
│   └─ ds_right_rear               → bus superado     │
└─────────────────────────────────────────────────────┘
```

---

## Flujo temporal de una evasión completa

```
t=0   LINE_FOLLOW: PID sigue carril a 30 km/h
      → Recognition ve bus (área >400px²) + LiDAR < 14.5m
t=1   WALL_FOLLOW inicia: velocidad baja a 15 km/h, guarda heading
      → Fase A: giro suave izquierda, abre espacio
      → Fase B: ds_right_mid < 4.5m → bus al costado → P lateral
      → ds_right_rear > 4.8m: bus superado
t=2   REORIENT: 20 km/h, giro P para recuperar heading guardado
      → |error| < 0.08 rad
t=3   RECENTER: busca línea amarilla con mismo pipeline PID
      → líneas detectadas → LINE_FOLLOW retoma el circuito
```

---

## Archivos entregados

```
MR4010_Actividad_4_2/
├── worlds/
│   └── city_2025a_4_2.wbt          ← mundo modificado (buses + sensores)
├── controllers/act_4_2/
│   ├── act_4_2.py                  ← controlador limpio
│   └── act_4_2_comentado.py        ← controlador comentado (entrega)
└── recap_actividad_4_2.md          ← este documento
```
