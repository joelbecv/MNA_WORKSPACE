# Plan de Trabajo — Proyecto Final: Conditional Imitation Learning (CIL)
### MR4010 Navegación Autónoma · Equipo 25
**Joel Becerril · 
**Fecha:** 2026-06-19

---

## Resumen ejecutivo

Implementar un sistema **Conditional Imitation Learning (CIL)** que conduzca autónomamente un vehículo en Webots recibiendo comandos de navegación en intersecciones (recto, izquierda, derecha). El sistema se entrena en el **Mundo 1** (sin tráfico) y se evalúa en el **Mundo 2** (con tráfico SUMO, peatones y vehículos estacionados), integrando los módulos de seguridad de actividades anteriores.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     MUNDO 2 — Inferencia                    │
│                                                             │
│  Cámara → CNN ──┐                                           │
│                 ├─→ Ángulo de dirección                     │
│  Comando Nav ───┘   (salida CIL)                            │
│                                                             │
│  + Radar         → Control de velocidad (distancia segura)  │
│  + Reconocimiento→ Freno de emergencia (peatones)           │
│  + LiDAR         → Evasión de obstáculos (vehículos)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Reutilización de actividades anteriores ✅

| Módulo | Fuente | Archivo | Estado |
|--------|--------|---------|--------|
| Lectura de cámara (BGRA→BGR) | Act 3.1 / 4.2 | `simple_controller_pedestrian_v2.py` | ✅ funciona |
| PID lane following (H2) | Act 2.1 / 3.1 | `simple_controller_pedestrian_v2.py` | ✅ parámetros validados |
| Detección peatones SVM + freno | Act 3.1 | `simple_controller_pedestrian_v2.py` | ✅ threshold=0.05 |
| Wall-following / evasión obstáculos | Act 4.2 | `act_4_2.py` | ✅ funciona a 15 km/h |
| Recognition de cámara (bus → peatón) | Act 4.2 | `act_4_2.py` | ✅ Webots R2023b API |
| Lectura keyboard + debounce | Act 4.2 | `act_4_2.py` | ✅ |
| Integración LiDAR frontal | Act 4.2 | `act_4_2.py` | ✅ funciona — solo NO usar enablePointCloud() |

> **Fix freeze macOS (ya aplicado en ambos .wbt):** La causa principal del freeze/beachball NO es el LiDAR — es que el PROTO BmwX5 busca un archivo de audio que no existe. Cada frame genera un warning que bloquea el hilo. Solución: `engineSound ""` en el nodo BmwX5 del .wbt. Ya aplicado en ambos mundos.
>
> **LiDAR:** funciona correctamente con `lidar.enable(timestep)` + `lidar.getRangeImage()`. Solo evitar `lidar.enablePointCloud()` (eso sí causa freeze).
>
> **Rosetta 2:** Abrir Webots con Rosetta en Apple Silicon → Finder → Webots.app → Cmd+I → "Abrir con Rosetta".

---

## Parámetros del mundo (verificados en .wbt)

| Parámetro | Valor |
|-----------|-------|
| `basicTimeStep` | 16 ms |
| Vehículo | BmwX5 (R2023b PROTO) |
| Mundos | `city_traffic_2025_01.wbt` (entrenamiento), `city_traffic_2025_02.wbt` (evaluación) |
| Velocidad de recolección | 30 km/h (constante, sin PID) |

**Conversión práctica a 30 km/h con timestep=16ms:**
- Velocidad = 8.33 m/s → 0.133 m/frame
- Capturar cada 5 frames = cada 80ms → ~12.5 fps → ~75 imágenes por minuto de conducción
- Para 10k imágenes → ~133 minutos de conducción distribuidos entre 4 integrantes

---

## FASE 1 — Recolección de datos (Mundo 1)
**Responsable sugerido:** Todos los integrantes conducen en turnos

### Controlador: `collect_cil_data.py`
**Basado en:** `act_4_2.py` (estructura de teclado + cámara) — sin PID, conducción manual.

#### Teclas de navegación (siguiendo Codevilla 2017)

| Tecla | Comando | Código interno |
|-------|---------|----------------|
| `←` Arrow Left | Girará izquierda en siguiente intersección | `CMD = 1` |
| `↑` Arrow Up | Seguirá recto | `CMD = 0` |
| `→` Arrow Right | Girará derecha en siguiente intersección | `CMD = 2` |

El comando se mantiene fijo hasta cambiar con teclado. Empieza en `CMD = 0` (recto).

#### Estructura del CSV de salida

```
image_path,steering_angle,nav_command
data/img_00001.jpg,0.042,0
data/img_00002.jpg,0.038,0
data/img_00150.jpg,-0.180,1
```

- **`steering_angle`**: valor de `driver.getSteeringAngle()` en radianes en el momento de captura
- **`nav_command`**: 0=recto, 1=izquierda, 2=derecha

#### Checklist de recolección de datos (por integrante)

- [ ] Cubrir **ambos sentidos** de circulación en todas las calles del mundo
- [ ] Incluir situaciones de **salida del carril y corrección** (zig-zag deliberado)
- [ ] Marcar `CMD=1` (izquierda) **5-10 frames antes** de llegar a la intersección y mantener durante el giro
- [ ] Marcar `CMD=2` (derecha) de la misma forma
- [ ] Volver a `CMD=0` al salir de la intersección
- [ ] Registrar al menos **2,500 imágenes por integrante** para llegar a ~10k antes de augmentation
- [ ] Verificar que el CSV no tenga filas con `steering_angle = 0.0` en tramos rectos (puede sesgar el dataset hacia 0)

#### Dataset balancing recomendado

| Segmento | Objetivo |
|----------|----------|
| Rectos (CMD=0) | ~50% del dataset |
| Curvas/giros izquierda (CMD=1) | ~25% del dataset |
| Curvas/giros derecha (CMD=2) | ~25% del dataset |

---

## FASE 2 — Entrenamiento en Google Colab
**Responsable sugerido:** Joel 

### Arquitectura CIL (Codevilla 2017)

```
Imagen (RGB, 88×200) ──→ CNN (ResNet/VGG pequeño) ──→ features
                                                         │
Comando Nav (one-hot 3) ──────────────────────────────→ concat
                                                         │
                                                    MLP (3 capas)
                                                         │
                                                  steering_angle
```

**Alternativa más simple (Bojarski 2016):** CNN directo sin branch de comando → útil como baseline para comparar.

### Pasos en Colab

```python
# Celda 1: clonar dataset desde GitHub
!git clone https://github.com/<usuario>/cil_dataset_equipo25

# Celda 2: instalar dependencias
!pip install tensorflow keras pandas opencv-python

# Celda 3: cargar CSV y normalizar
import pandas as pd
df = pd.read_csv('cil_dataset_equipo25/dataset.csv')
# Normalizar steering: dividir entre MAX_ANGLE (0.5 rad)
df['steering_norm'] = df['steering_angle'] / 0.5

# Celda 4: data augmentation
# - Flip horizontal: img_flip, steering = -steering_norm, mismo cmd
# - Brightness jitter: factor aleatorio 0.7–1.3
# - Gaussian noise: std=0.02 sobre imagen normalizada

# Celda 5: definir modelo CIL
# Celda 6: entrenar (50–100 épocas, EarlyStopping patience=10)
# Celda 7: exportar modelo
model.save('cil_model_equipo25.h5')
```

### Métricas de evaluación del entrenamiento

| Métrica | Umbral aceptable |
|---------|-----------------|
| Val MAE (steering) | < 0.05 rad (~3°) |
| Val MSE | < 0.01 |
| Loss curva | Sin overfitting visible después de epoch 30 |

---

## FASE 3 — Inferencia autónoma (Mundo 2)
**Responsable sugerido:** Joel 

### Controlador: `autonomous_cil.py`

#### Máquina de estados

```
STATES:
  CIL_DRIVE     — Inferencia normal: imagen → CNN → steering
  BRAKE_PED     — Freno de emergencia por peatón detectado
  EVADE_OBS     — Wall-following lateral (evasión de obstáculo)
  REORIENT      — Recuperación de heading post-evasión
```

#### Sensores y su función en Mundo 2

| Sensor | API Webots R2023b | Función |
|--------|------------------|---------|
| Cámara | `getDevice("camera")` | Input al modelo CIL |
| Reconocimiento cámara | `camera.recognitionEnable(ts)` | Detectar peatones → freno |
| Radar delantero | `getDevice("radar")` (sensorsSlotFront) | Distancia al vehículo más próximo → speed |
| LiDAR (si no macOS) | `getDevice("Sick LMS 291")` | Evasión de obstáculos laterales |
| Distance sensors | `ds_right_front / ds_right_mid / ds_right_rear` | Wall-following durante evasión |

#### Lógica de radar (nueva — Act Final)

```python
RADAR_THRESHOLD = 15.0  # m: distancia de seguridad
# Si distancia_radar < RADAR_THRESHOLD: frenar progresivamente
# Si distancia_radar < 5.0 m: detener completamente
speed = CRUISE_SPEED * min(1.0, dist_radar / RADAR_THRESHOLD)
```

#### Integración con módulos previos (copy-paste seguro)

```python
# ── Freno de emergencia por peatón (de simple_controller_pedestrian_v2.py) ──
# Copiar: recognitionEnable(), bucle de getRecognitionObjects(), lógica de brake
# NO copiar: el SVM (domain gap documentado — usar solo recognition de cámara)

# ── Evasión de obstáculos (de act_4_2.py) ──
# Copiar: wall_follow_step(), REORIENT state, integración de giroscopio
# Ajustar: BUS_LIDAR_THRESH → OBSTACLE_RADAR_THRESH (si se usa radar)
```

---

## Las 3 rutas requeridas (Mundo 2)

| Ruta | Origen | Destino | Comandos | Requisito especial |
|------|--------|---------|----------|-------------------|
| **A** | Gasolinera | Silos | Solo recto (CMD=0) | Freno por peatón |
| **B** | Parque infantil | Iglesia | Al menos 1 derecha (CMD=2) | Evasión de obstáculo |
| **C** | Hospital | Escuela | Al menos 1 izquierda (CMD=1) | Control de distancia radar |

> Definir los puntos exactos de origen/destino una vez abierto el Mundo 2 en Webots. Reposicionar el vehículo (Supervisor o posición inicial del .wbt) al inicio de cada ruta antes de grabar.

---

## Pre-flight checklist (protocolo obligatorio antes de cada ejecución)

Seguir este orden. **No saltarse ningún paso.**

- [ ] **1.** Mapear cada requisito de la rúbrica → función de código específica
- [ ] **2.** Abrir el mundo en Webots, leer la consola antes de ejecutar el controller
- [ ] **3.** Verificar nombres de devices: `getDevice("camera")`, `getDevice("radar")`, `getDevice("Sick LMS 291")`
- [ ] **4.** Verificar API R2023b: `camera.recognitionEnable(ts)` (NO `enableRecognition`)
- [ ] **5.** Syntax check: `python -m py_compile controller.py && echo OK`
- [ ] **6.** Primera corrida: solo prints de sensores, sin control activo
- [ ] **7.** Comunicar cualquier cambio al equipo antes de implementar

---

## Estructura de carpetas del proyecto

```
MR4010_proyecto_final_2026/
├── worlds/
│   ├── city_traffic_2025_01.wbt        ← NO MODIFICAR
│   └── city_traffic_2025_02.wbt        ← NO MODIFICAR
├── controllers/
│   ├── collect_cil_data/
│   │   └── collect_cil_data.py         ← Fase 1: recolección
│   └── autonomous_cil/
│       └── autonomous_cil.py           ← Fase 3: inferencia
├── data/
│   ├── images/                         ← imágenes capturadas (en .gitignore)
│   └── dataset.csv                     ← CSV con paths, steering, command
├── models/
│   └── cil_model_equipo25.h5           ← modelo exportado de Colab
└── docs/
    └── reporte_equipo25.pdf            ← entregable final
```

> `data/images/` debe estar en `.gitignore`. Subir el dataset a GitHub con `git-lfs` o como release asset.

---

## División de trabajo sugerida

| Integrante | Responsabilidad |
|-----------|----------------|
| **Joel** | `collect_cil_data.py` + integración final `autonomous_cil.py` |
|

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Dataset desbalanceado (mayoría ángulo=0) | Alta | Hacer zig-zags deliberados + oversample giros |
| Modelo no generaliza a Mundo 2 | Media | Data augmentation agresivo + conducir ambos sentidos |
| Freeze macOS | Alta | Ya resuelto: `engineSound ""` en .wbt aplicado. No usar `enablePointCloud()` |
| SVM domain gap (peatones) | Alta | Usar solo `recognitionEnable()` — más confiable que SVM en Webots |
| SUMO degrada performance con +30 vehículos | Media | Limitar max vehículos SUMO a 30 en Scene Tree |

---

## Entregables del proyecto

- [ ] Controlador Mundo 1 (`collect_cil_data.py`) con comentarios
- [ ] Dataset en GitHub (>10k imágenes post-augmentation + CSV)
- [ ] Notebook Google Colab con entrenamiento documentado
- [ ] Controlador Mundo 2 (`autonomous_cil.py`) con comentarios
- [ ] Video YouTube < 6 min: arquitectura + evidencia de 3 rutas
- [ ] Reporte PDF con código comentado + enlace YouTube

---

## Estado actual de archivos generados

| Archivo | Estado |
|---------|--------|
| `worlds/city_traffic_2025_01.wbt` | ✅ Modificado: `engineSound ""` + `name "display_image"` |
| `worlds/city_traffic_2025_02.wbt` | ✅ Modificado: `engineSound ""` + LiDAR + Radar + Recognition + distance sensors |
| `controllers/collect_cil_data/collect_cil_data.py` | ✅ Listo para usar |
| `controllers/autonomous_cil/autonomous_cil.py` | ✅ Listo (requiere modelo .h5 de Colab) |
| `models/cil_model_equipo25.h5` | ⏳ Pendiente — exportar de Colab después de entrenar |

## Próximo paso inmediato

1. **Abrir `city_traffic_2025_01.wbt`** en Webots con Rosetta 2 → cambiar controller del BMW a `collect_cil_data` → Play
2. **Verificar consola**: debe imprimir "Reanudando desde imagen #0" sin errores
3. **Conducción manual distribuida** — cada integrante conduce ~35 min y agrega imágenes al dataset compartido

---

*Generado: 2026-06-19 | Basado en rúbrica `rubrica.txt` + código validado de Act 2.1, 3.1, 4.2*
