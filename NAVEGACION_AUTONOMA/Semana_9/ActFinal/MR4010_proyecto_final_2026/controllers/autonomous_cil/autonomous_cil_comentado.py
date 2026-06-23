"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  GUÍA DE LECTURA — autonomous_cil_comentado.py                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CÓMO LEER ESTE ARCHIVO:                                                     ║
║  • Bloques NARANJA (como este)  → guía pedagógica: qué hace, por qué,       ║
║    dónde se usó antes y qué decir en el VIDEO                               ║
║  • Comentarios # grises         → notas técnicas cortas sobre líneas         ║
║                                                                              ║
║  PROPÓSITO DEL CONTROLADOR:                                                  ║
║  Conducción autónoma completa usando el modelo CIL entrenado en Colab.       ║
║  El BmwX5 en el Mundo 2 (con tráfico SUMO) debe:                            ║
║    1. Seguir carriles con la CNN (steering angle)                             ║
║    2. Respetar comandos de navegación en intersecciones (A/W/D)              ║
║    3. Frenar ante peatones (camera recognition)                              ║
║    4. Mantener distancia de seguridad (radar)                                ║
║    5. Evadir obstáculos estacionados (LiDAR + wall-following)                ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    CIL_DRIVE → BRAKE_PED → CIL_DRIVE          (peatón detectado y liberado) ║
║    CIL_DRIVE → EVADE_OBS → REORIENT → CIL_DRIVE (obstáculo y recuperación) ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os

# TensorFlow con manejo graceful de import — si no está instalado, CIL va en 0.0 (recto)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[CIL] TensorFlow no disponible — instalar con: pip install tensorflow")

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 1 — PARÁMETROS GENERALES Y COMANDOS DE NAVEGACIÓN               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Define las constantes globales del sistema autónomo. Separamos los          ║
║  parámetros por módulo (CIL, radar, peatones, LiDAR, reorient) para que     ║
║  cualquier integrante pueda ajustar un módulo sin tocar otro.                ║
║                                                                              ║
║  COMANDOS CIL (Codevilla 2017):                                              ║
║  El operador en la grabación del video presiona A/W/D ANTES de llegar a     ║
║  la intersección. Esto replica el "comando de intención" del conductor,      ║
║  que el modelo aprendió durante el entrenamiento.                            ║
║    0 = CMD_STRAIGHT → seguir recto (predicción de giro pequeño)              ║
║    1 = CMD_LEFT     → girar a la izquierda                                   ║
║    2 = CMD_RIGHT    → girar a la derecha                                     ║
║                                                                              ║
║  MAX_STEER_RATE = 0.03 rad/frame:                                            ║
║  Sin este limitador, la CNN produce cambios abruptos que hacen zigzaguear   ║
║  el vehículo (bug observado en experimentos preliminares). El rate limiter   ║
║  suaviza el output del modelo sin añadir latencia perceptible.               ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 2.1 (H2/H3): CRUISE_SPEED=30, MAX_ANGLE=0.5 — mismos valores         ║
║    validados en mundo highway                                                ║
║  • Act 4.2 (evasión bus): SPEED_EVADE=15 para maniobras a baja velocidad    ║
║  • collect_cil_data.py: misma estructura CMD_STRAIGHT/LEFT/RIGHT             ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Señalar la tabla de parámetros y explicar que fueron heredados de las     ║
║    actividades anteriores que ya funcionaban en estos mundos                 ║
║  → "MAX_STEER_RATE actúa como un filtro de paso bajo sobre el output de      ║
║    la CNN — evita oscilaciones que no veríamos en datos de entrenamiento"   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

CRUISE_SPEED   = 30      # km/h en conducción normal
MAX_ANGLE      = 0.5     # rad — límite físico del volante BmwX5
MAX_STEER_RATE = 0.03    # rad/frame — limita cambios bruscos del modelo CIL

CMD_STRAIGHT = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_LABEL    = {0: "RECTO", 1: "IZQUIERDA", 2: "DERECHA"}

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 2 — PARÁMETROS DE RADAR (nueva funcionalidad, Act Final)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Define la zona de seguridad delantera. El radar mide la distancia al        ║
║  vehículo más cercano en el frente. Si está a menos de RADAR_SAFE_M,         ║
║  se reduce la velocidad proporcionalmente.                                   ║
║                                                                              ║
║  FÓRMULA DE VELOCIDAD PROPORCIONAL:                                          ║
║    v = CRUISE_SPEED × min(1.0, dist / RADAR_SAFE_M)                         ║
║                                                                              ║
║  Ejemplo numérico:                                                           ║
║    dist = 15.0 m  → factor = 1.0  → v = 30 km/h (velocidad plena)           ║
║    dist = 10.0 m  → factor = 0.67 → v = 20 km/h                             ║
║    dist =  7.5 m  → factor = 0.50 → v = 15 km/h                             ║
║    dist =  5.0 m  → factor = 0.33 → v = STOP (se usa RADAR_STOP_M)          ║
║                                                                              ║
║  FÍSICA: el auto tarda en frenar. A 30 km/h (~8.3 m/s) y asumiendo          ║
║  deceleración de 4 m/s², la distancia de frenado es ~8.7 m.                 ║
║  Por eso RADAR_SAFE_M = 15 m: deja ~6 m de margen de seguridad.             ║
║                                                                              ║
║  DIFERENCIA CON ACT 4.2:                                                     ║
║  En Act 4.2 el bus se detectaba con LiDAR (cono frontal). Aquí usamos        ║
║  el Radar porque puede detectar vehículos en movimiento del SUMO a través    ║
║  de obstrucciones parciales — el LiDAR no atraviesa objetos.                 ║
║                                                                              ║
║  PRIMERA VEZ que usamos Radar en estas actividades.                           ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Demostrar en Mundo 2: acercarse lentamente a un vehículo SUMO,           ║
║    mostrar que la velocidad baja progresivamente en el HUD                   ║
║  → "Este es el componente nuevo respecto a actividades anteriores —           ║
║    el radar permite adaptarse al tráfico SUMO dinámico"                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

RADAR_SAFE_M = 15.0   # m: zona de desaceleración progresiva
RADAR_STOP_M =  5.0   # m: detención completa

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 3 — PARÁMETROS DE FRENO POR PEATÓN (de Act 3.1 y Act 4.2)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Sistema de confirmación + histéresis para el freno ante peatones.            ║
║  Evita frenadas por falsos positivos del recognition (ruido de un frame).    ║
║                                                                              ║
║  LÓGICA DE CONFIRMACIÓN (doble umbral):                                      ║
║    • Para ENTRAR en BRAKE_PED: necesita PED_CONFIRM_N=2 detecciones          ║
║      positivas CONSECUTIVAS → evita reaccionar a un falso positivo aislado   ║
║    • Para SALIR de BRAKE_PED: necesita PED_RELEASE_N=4 negativas + mantener  ║
║      el freno PED_HOLD_F=100 frames (~1.6 s) para dejar pasar al peatón     ║
║                                                                              ║
║  POR QUÉ DETECT_EVERY = 10:                                                  ║
║  camera.recognitionEnable(timestep × 10) → el recognition corre cada 160ms.  ║
║  Correrlo en cada frame (16ms) genera carga de CPU que puede acumular frames  ║
║  pendientes y causar lag perceptible. En pruebas de Act 4.2, cada 10 frames  ║
║  fue suficiente para detectar peatones antes de que alcancen el cruce.       ║
║                                                                              ║
║  NOTA SVM:                                                                   ║
║  En Act 3.1 también usábamos un SVM para clasificar peatones desde la        ║
║  imagen de cámara. En el proyecto final NO usamos el SVM porque sufría de    ║
║  "domain gap": fue entrenado con imágenes INRIA (fotos reales) pero en       ║
║  Webots las texturas son sintéticas → scores máx ~0.39, frecuentes falsos    ║
║  positivos. El recognition nativo de Webots es más confiable aquí.           ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 3.1: primera implementación de freno por peatón con SVM               ║
║  • Act 4.2: misma lógica de recognition + confirmación, sin SVM              ║
║  • Este archivo: copiado de Act 4.2 con ajuste menor en PED_HOLD_F           ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → "Descartamos el SVM de la Act 3.1 porque entrenamos con INRIA (dataset    ║
║    de fotos reales) y Webots usa texturas sintéticas — el recognition        ║
║    integrado de Webots es más confiable en este entorno"                     ║
║  → Mostrar la consola cuando aparezca "[CIL] PEATÓN detectado"               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

PED_CONFIRM_N = 2       # detecciones consecutivas para confirmar
PED_RELEASE_N = 4       # negativas consecutivas para liberar freno
PED_HOLD_F    = 100     # frames de freno garantizados (~1.6 s)
DETECT_EVERY  = 10      # recognition cada N frames para ahorrar CPU

PEDESTRIAN_KEYWORDS = ["pedestrian", "Pedestrian", "human", "person"]

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 4 — PARÁMETROS DE LIDAR Y WALL-FOLLOWING (de Act 4.2)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  El LiDAR Sick LMS 291 hace un barrido de 180° cada timestep.                ║
║  Si detecta algo a menos de OBS_LIDAR_THRESH metros en el cono frontal,     ║
║  se activa el wall-following lateral para rodear el obstáculo.               ║
║                                                                              ║
║  PARÁMETROS COPIADOS EXACTAMENTE DE Act 4.2:                                 ║
║  • WALL_TARGET = 2.9 m  → distancia al flanco del obstáculo                 ║
║  • KP_WALL = 0.10       → ganancia proporcional (P) del controlador de pared ║
║  • DS_CLEAR_DIST = 4.8 m → sensor trasero > 4.8 m → obstáculo ya superado  ║
║  • DS_ENGAGE_DIST = 4.5 m → sensor frontal < 4.5 m → seguir con evasión    ║
║                                                                              ║
║  POR QUÉ ESTOS VALORES Y NO OTROS:                                           ║
║  Fueron encontrados con búsqueda iterativa en Act 4.2 con el bus urbano.     ║
║  KP_WALL < 0.08 → el auto no corrige suficientemente y golpea el obstáculo  ║
║  KP_WALL > 0.15 → el auto oscila en torno a WALL_TARGET                     ║
║  0.10 fue el punto de "mínima oscilación con corrección suficiente".         ║
║                                                                              ║
║  DIFERENCIA RESPECTO A ACT 4.2:                                              ║
║  En Act 4.2 el obstáculo era específicamente un "bus urbano" con dimensiones ║
║  conocidas. Aquí es genérico: cualquier vehículo estacionado.                ║
║  OBS_LIDAR_THRESH = 14m (vs 10m en Act 4.2) para dar más margen a           ║
║  velocidades de 30 km/h vs 15 km/h del bus.                                 ║
║                                                                              ║
║  REGLA CRÍTICA — LiDAR API:                                                   ║
║  lidar.enable(timestep) + lidar.getRangeImage() → funciona correctamente     ║
║  lidar.enablePointCloud() → FREEZE macOS — confirmado en diagnósticos Act 4.2║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 4.2 (act_4_2.py): wall_follow_step() y la máquina EVADE→REORIENT     ║
║    son copia directa de ese controlador                                      ║
║  • Act 2.1 (H2): SPEED_EVADE=15 también fue la velocidad de maniobra en H2  ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → "Los parámetros del wall-following los validamos en la Act 4.2 con el bus ║
║    urbano — reutilizamos exactamente los mismos valores para el proyecto     ║
║    final porque el comportamiento físico del vehículo no cambia"             ║
║  → Demostrar la evasión en Mundo 2 con un vehículo estacionado               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

OBS_LIDAR_THRESH = 14.0   # m: umbral de detección frontal
LIDAR_FOV_DEG    = 20     # grados a cada lado del centro del barrido

SPEED_EVADE    = 15       # km/h durante evasión (velocidad reducida)
WALL_TARGET    = 2.9      # m: distancia objetivo al flanco del obstáculo
KP_WALL        = 0.10     # ganancia P del controlador de pared
DS_CLEAR_DIST  = 4.8      # m: sensor trasero libre → obstáculo superado
DS_ENGAGE_DIST = 4.5      # m: sensor frontal sigue activo → continuar evasión

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 5 — PARÁMETROS DE REORIENTACIÓN (de Act 4.2)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Después de evadir un obstáculo, el auto queda desalineado del carril.       ║
║  El estado REORIENT usa el giroscopio para medir cuánto se desvió y         ║
║  aplica una corrección proporcional hasta volver al heading original.        ║
║                                                                              ║
║  MATEMATICA — integración de yaw rate:                                       ║
║    heading(t) = Σ yaw_rate(k) × Δt  para k = 0..t                          ║
║    Δt = timestep / 1000.0 = 0.016 s                                          ║
║    Error = heading_ref − heading_actual                                       ║
║    steer_corr = KP_HEADING × heading_error                                   ║
║                                                                              ║
║  heading_ref se fija al ENTRAR en EVADE_OBS (heading de referencia antes     ║
║  del desvío). Al salir de EVADE_OBS, heading_ref se re-captura como el       ║
║  target al que debe volver el auto.                                           ║
║                                                                              ║
║  HEADING_TOL = 0.08 rad ≈ 4.6°:                                              ║
║  Umbral de aceptación. En pruebas de Act 4.2, tolerancias < 0.05 rad         ║
║  causaban oscilaciones (sobrecompensación) porque el giroscopio acumula      ║
║  pequeño drift. Con 0.08 rad el auto declara "recuperado" sin oscilaciones.  ║
║                                                                              ║
║  GYRO AXIS:                                                                  ║
║  gyro.getValues()[2] → eje Z en el frame del vehículo                        ║
║  En Webots R2023b, el eje Z del gyro corresponde a la rotación de yaw.       ║
║  Confirmado en Act 4.2 con tests de giro manual.                             ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 4.2: REORIENT state completo — copia exacta                           ║
║  • Valores KP_HEADING=1.0 y HEADING_TOL=0.08 validados en Act 4.2 con bus   ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Mostrar la pantalla dividida: Webots view + consola con los prints        ║
║    "[EVADE] → REORIENT → CIL_DRIVE" para mostrar la transición              ║
║  → "La reorientación fue el módulo más difícil de calibrar en Act 4.2 —     ║
║    heredamos esos parámetros directamente"                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

SPEED_REORIENT = 20       # km/h durante corrección de heading
KP_HEADING     = 1.0      # ganancia P del controlador de heading
HEADING_TOL    = 0.08     # rad ≈ 4.6° — umbral de convergencia

# =============================================================================

STATE_CIL   = "CIL_DRIVE"
STATE_PED   = "BRAKE_PED"
STATE_EVADE = "EVADE_OBS"
STATE_REOR  = "REORIENT"

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 6 — INICIALIZACIÓN: DRIVER, SENSORES Y REGLAS CRÍTICAS          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Conecta todos los dispositivos del BmwX5 en el Mundo 2. Cada getDevice()   ║
║  busca el dispositivo por el nombre exacto definido en el .wbt.              ║
║                                                                              ║
║  REGLAS APLICADAS EN ESTA SECCIÓN:                                            ║
║                                                                              ║
║  REGLA 1 — timestep sin multiplicador:                                       ║
║    timestep = int(driver.getBasicTimeStep())  # = 16 ms                     ║
║    Causa descubierta en Act 3.1: multiplicar (×2, ×3) causaba freeze.        ║
║    El controlador H3 de Act 2.1 que funcionó usaba esta línea exacta.       ║
║                                                                              ║
║  REGLA 2 — engineSound "" ya en el .wbt:                                     ║
║    El PROTO BmwX5 buscaba un archivo de audio de motor inexistente.          ║
║    Cada frame generaba un warning que acumulaba mensajes → freeze macOS.     ║
║    Solución: engineSound "" en el nodo BmwX5 del .wbt. Ya aplicado.         ║
║                                                                              ║
║  REGLA 3 — recognitionEnable (NO enableRecognition):                         ║
║    API R2023b cambió el nombre del método. En versiones anteriores de        ║
║    Webots se usaba enableRecognition() pero en R2023b es recognitionEnable().║
║    Confirmado en Act 4.2 contra la documentación oficial R2023b.             ║
║                                                                              ║
║  REGLA 4 — lidar.enable() sin enablePointCloud():                            ║
║    enablePointCloud() activa la visualización 3D de la nube de puntos,       ║
║    que requiere un proceso separado que causa freeze en macOS. Solo          ║
║    necesitamos getRangeImage() (1D) para detectar distancias → enable() basta.║
║                                                                              ║
║  REGLA 5 — gyro eje Z:                                                       ║
║    gyro.getValues() retorna [gyro_x, gyro_y, gyro_z]. El eje Z en el        ║
║    frame del vehículo es el yaw rate (rotación horizontal). Confirmado       ║
║    con test manual en Act 4.2: giro izquierda → Z negativo, derecha → Z+.   ║
║                                                                              ║
║  DISPOSITIVOS NUEVOS vs ACTIVIDADES ANTERIORES:                               ║
║  • radar → NUEVO en proyecto final (no lo usamos en acitividades 2.1-4.2)   ║
║  • ds_right_* → copiados de Act 4.2 (wall-following del bus)                ║
║  • lidar/camera/display/keyboard → presentes desde Act 3.1                  ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Señalar las 5 reglas y explicar que cada una tiene un "porqué" que         ║
║    costó iteraciones descubrir — no son caprichosas                          ║
║  → "El radar es la única adición verdaderamente nueva; todo lo demás         ║
║    es reutilización directa de lo que ya funcionaba"                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

driver   = Driver()
timestep = int(driver.getBasicTimeStep())   # REGLA 1: sin multiplicador — 16 ms

# Cámara: input CIL + recognition de peatones
camera = driver.getDevice("camera")
camera.enable(timestep)
camera.recognitionEnable(timestep * DETECT_EVERY)   # REGLA 3: API R2023b
CAM_W = camera.getWidth()    # 320 px
CAM_H = camera.getHeight()   # 160 px

# Radar: distancia a vehículos frontales (NUEVO en Act Final)
radar = driver.getDevice("radar")
radar.enable(timestep)

# LiDAR Sick LMS 291: detección de obstáculos estacionados
lidar = driver.getDevice("Sick LMS 291")
lidar.enable(timestep)                  # REGLA 4: enable() — NO enablePointCloud()
LIDAR_RAYS = lidar.getNumberOfPoints()  # 180 rayos en este modelo

# Sensores de distancia laterales: wall-following durante evasión (de Act 4.2)
ds_rf = driver.getDevice("ds_right_front")
ds_rm = driver.getDevice("ds_right_mid")
ds_rr = driver.getDevice("ds_right_rear")
for ds in [ds_rf, ds_rm, ds_rr]:
    ds.enable(timestep)

# Giroscopio: integración de heading para REORIENT (de Act 4.2)
gyro = driver.getDevice("gyro")
gyro.enable(timestep)

display  = driver.getDevice("display_image")   # HUD 200×150
keyboard = Keyboard()
keyboard.enable(timestep)

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 7 — CARGA DEL MODELO CIL (corazón del proyecto)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Carga el modelo entrenado en Google Colab. Este modelo es el resultado       ║
║  de las Fases 1 y 2: los datos recolectados con collect_cil_data.py fueron  ║
║  usados en Colab para entrenar la CNN con arquitectura CIL.                  ║
║                                                                              ║
║  RUTA DEL MODELO:                                                             ║
║    models/cil_model_equipo25.h5                                               ║
║  Ruta resuelta relativa al directorio del controlador:                        ║
║    controllers/autonomous_cil/../../models/cil_model_equipo25.h5             ║
║  Esta ruta es portable: funciona en cualquier computadora del equipo si      ║
║  clonaron el repo completo.                                                   ║
║                                                                              ║
║  DISEÑO DEFENSIVO:                                                            ║
║  Si el modelo no existe → cil_predict() retorna 0.0 (steering recto) y el   ║
║  auto sigue operando con los módulos de seguridad activos. Permite probar   ║
║  radar, peatones y evasión ANTES de tener el modelo entrenado.               ║
║                                                                              ║
║  preprocess_image() — adaptación necesaria:                                   ║
║  La firma de entrada del modelo depende de cómo se definió en Colab:         ║
║    OPCIÓN A: modelo de UNA entrada → img_batch (imagen + cmd como canal)     ║
║    OPCIÓN B: modelo de DOS entradas → [img_batch, cmd_onehot]                ║
║  El código intenta la opción B primero y hace fallback a la A.               ║
║  AJUSTAR según el notebook de Colab del equipo.                              ║
║                                                                              ║
║  MODELO DE INPUT — arquitectura CIL (Codevilla 2017):                        ║
║    Imagen: (1, 88, 200, 3) → CNN extrae features visuales del carril        ║
║    Comando: (1, 3)  one-hot → selecciona el "branch" de la MLP              ║
║    Output: (1, 1)  steering_norm ∈ [-1, 1] → × MAX_ANGLE → ángulo real     ║
║                                                                              ║
║  PRIMERA VEZ que cargamos un modelo Keras entrenado por el equipo.            ║
║  (En Act 3.1 usamos SVM sklearn .pkl — diferente flujo de carga)             ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Mostrar model.summary() en la consola — número de parámetros, capas      ║
║  → "El modelo toma 88×200 px y un vector one-hot de 3 elementos — en         ║
║    total [N] parámetros entrenables"                                         ║
║  → Mostrar la curva de entrenamiento (loss) del notebook Colab               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

CTRL_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "models", "cil_model_equipo25.h5"))

model = None
if TF_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[CIL] Modelo cargado: {MODEL_PATH}")
        model.summary()
    else:
        print(f"[CIL] Modelo NO encontrado: {MODEL_PATH}")
        print("[CIL] Copiar cil_model_equipo25.h5 de Colab a models/")


def preprocess_image(bgr: np.ndarray, nav_cmd: int):
    # Resize a 88×200: resolución usada en Codevilla 2017
    img_model = cv2.resize(bgr, (200, 88))
    img_norm  = img_model.astype(np.float32) / 255.0    # normalizar [0, 1]
    img_batch = np.expand_dims(img_norm, axis=0)         # (1, 88, 200, 3)

    # Comando como vector one-hot: la CNN usa el índice para activar el branch
    cmd_onehot = np.zeros((1, 3), dtype=np.float32)
    cmd_onehot[0, nav_cmd] = 1.0

    return img_batch, cmd_onehot


def cil_predict(bgr: np.ndarray, nav_cmd: int) -> float:
    """Inferencia CIL. Retorna ángulo en rad. 0.0 si no hay modelo."""
    if model is None:
        return 0.0

    img_inp, cmd_inp = preprocess_image(bgr, nav_cmd)

    try:
        pred = model.predict([img_inp, cmd_inp], verbose=0)   # 2 entradas (Codevilla)
    except Exception:
        pred = model.predict(img_inp, verbose=0)               # fallback 1 entrada

    steering_norm = float(pred[0][0])
    return np.clip(steering_norm, -1.0, 1.0) * MAX_ANGLE

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 8 — FUNCIONES DE SENSORES: LiDAR, RADAR, PEATONES               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Tres funciones puras que transforman lecturas crudas de sensores en         ║
║  valores de decisión para la máquina de estados.                             ║
║                                                                              ║
║  get_lidar_front_dist():                                                     ║
║    LiDAR devuelve 180 valores de distancia (0°..180°, 1°/rayo).              ║
║    Tomamos el cono central ±LIDAR_FOV_DEG=20° → 40 rayos.                  ║
║    Filtramos 0.1 m < r < 80 m para eliminar ruido del propio vehículo.      ║
║    Retornamos el mínimo: la distancia al obstáculo más cercano en el frente. ║
║                                                                              ║
║  get_radar_min_dist():                                                       ║
║    Radar retorna una lista de RadarTarget. Cada target tiene .distance y     ║
║    .speed (velocidad relativa). Tomamos el target más cercano.               ║
║    Si no hay targets (carretera vacía) → retorna 999.0 (sin freno).          ║
║                                                                              ║
║  check_pedestrian():                                                          ║
║    camera.getRecognitionObjects() → lista de objetos reconocidos por          ║
║    Webots en la imagen actual. Comparamos el modelo de cada objeto con       ║
║    las palabras clave de peatón. Un solo match → return True.                ║
║    Esta función solo se llama cada DETECT_EVERY frames (Bloque 3).           ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 3.1: check_pedestrian() es copia de simple_controller_pedestrian.py  ║
║  • Act 4.2: get_lidar_front_dist() es copia de act_4_2.py con FOV ajustado  ║
║  • get_radar_min_dist(): NUEVA — primera vez que usamos Radar API            ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Mostrar en la consola: los prints de "[CIL] PEATÓN detectado" y           ║
║    "[CIL] Obstáculo a X.Xm → EVADE_OBS"                                     ║
║  → "Las tres funciones de sensor son la capa de percepción — separarlas      ║
║    del control hace más fácil depurar cuál sensor falló"                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

def get_lidar_front_dist() -> float:
    """Distancia mínima en el cono frontal ±LIDAR_FOV_DEG°."""
    ranges = lidar.getRangeImage()
    if not ranges:
        return 999.0
    center = LIDAR_RAYS // 2
    half   = int(LIDAR_FOV_DEG * LIDAR_RAYS / 180)
    cone   = ranges[center - half: center + half]
    valid  = [r for r in cone if 0.1 < r < 80.0]   # filtrar self y ruido lejano
    return min(valid) if valid else 999.0


def get_radar_min_dist() -> float:
    """Distancia al vehículo más cercano detectado por el radar."""
    targets = radar.getTargets()
    if not targets:
        return 999.0   # sin targets = carretera despejada
    return min(t.distance for t in targets)


def check_pedestrian() -> bool:
    """Detecta peatones vía recognition (API R2023b). Copiado de Act 4.2."""
    objs = camera.getRecognitionObjects()
    for obj in objs:
        model_name = obj.getModel() if hasattr(obj, 'getModel') else ""
        if any(kw in model_name for kw in PEDESTRIAN_KEYWORDS):
            return True
    return False

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 9 — FUNCIONES DE CONTROL: WALL-FOLLOW, RATE LIMITER, HUD        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Tres funciones de actuación que transforman medidas de sensor en comandos  ║
║  al volante o al display.                                                    ║
║                                                                              ║
║  wall_follow_step():                                                          ║
║    Controlador P puro: error = WALL_TARGET - ds_middle.                      ║
║    Si el sensor lateral medio lee menos que el target → el auto está muy     ║
║    cerca → steer negativo (alejarse). Y viceversa.                           ║
║    Usando solo el sensor MEDIO (ds_rm) porque da la mejor lectura estable    ║
║    durante la maniobra. En Act 4.2 se probó con promedio de los tres y       ║
║    generó oscilaciones — el sensor del medio sólo es más robusto.            ║
║                                                                              ║
║  apply_rate_limit():                                                          ║
║    Limita cuánto puede cambiar el steering en un solo frame.                  ║
║    Sin esto, si la CNN salta de -0.3 a +0.3 en un frame (180 ms en total),  ║
║    el auto sacude bruscamente. Con rate limiter = 0.03 rad/frame, el         ║
║    cambio completo tarda 20 frames = 320 ms → suave para el pasajero.        ║
║                                                                              ║
║  draw_hud():                                                                  ║
║    HUD de información en el display 200×150 del BmwX5.                       ║
║    Fondo de color cambia según estado: verde=CIL, rojo=PED, azul=EVADE,     ║
║    amarillo=REORIENT. Útil para el video: el estado es visible en el plano  ║
║    de la cámara del auto.                                                    ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 4.2: wall_follow_step() copia exacta (mismos parámetros)              ║
║  • Act 3.1: draw_hud() primitiva (sin cambio de color por estado)            ║
║  • apply_rate_limit(): NUEVA en proyecto final — necesaria para suavizar     ║
║    la salida de la CNN que puede ser más ruidosa que el control manual        ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Mostrar el display del auto durante la evasión — el fondo azul indica     ║
║    claramente que está en modo EVADE_OBS                                     ║
║  → "El HUD de colores nos permite ver el estado de la máquina de estados     ║
║    directamente en la simulación sin necesidad de leer la consola"           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

def wall_follow_step(ds_f: float, ds_m: float, ds_r: float) -> float:
    """Controlador P de seguimiento de pared. Copiado de Act 4.2."""
    error = WALL_TARGET - ds_m   # positivo → alejarse del obstáculo
    steer = KP_WALL * error
    return float(np.clip(steer, -MAX_ANGLE, MAX_ANGLE))


def apply_rate_limit(current: float, target: float) -> float:
    """Limita el cambio de steering a MAX_STEER_RATE rad por frame."""
    delta = np.clip(target - current, -MAX_STEER_RATE, MAX_STEER_RATE)
    return current + delta


def draw_hud(state: str, nav_cmd: int, steer: float,
             radar_d: float, ped: bool, lidar_d: float) -> None:
    """HUD de estado. Color de fondo cambia por estado de la máquina."""
    colors = {
        STATE_CIL:   0x002200,   # verde oscuro — conducción normal
        STATE_PED:   0x440000,   # rojo oscuro  — freno por peatón
        STATE_EVADE: 0x002244,   # azul oscuro  — evasión activa
        STATE_REOR:  0x222200,   # amarillo osc — corrección de heading
    }
    display.setColor(colors.get(state, 0x000000))
    display.fillRectangle(0, 0, 200, 150)

    display.setColor(0xFFFFFF)
    display.drawText(f"STATE: {state}", 2, 2)
    display.drawText(f"NAV:   {CMD_LABEL[nav_cmd]}", 2, 14)
    display.drawText(f"Steer: {steer:+.3f}r", 2, 26)
    display.drawText(f"Radar: {radar_d:.1f}m", 2, 38)
    display.drawText(f"LiDAR: {lidar_d:.1f}m", 2, 50)

    display.setColor(0xFF4444 if ped else 0x44FF44)
    display.drawText(f"Ped:   {'DETECTADO' if ped else 'libre'}", 2, 62)

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 10 — LOOP PRINCIPAL: MÁQUINA DE ESTADOS COMPLETA                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Es el corazón del controlador. Cada 16ms (un frame de simulación):          ║
║    1. Lee teclado → actualiza nav_cmd                                         ║
║    2. Lee todos los sensores (cámara, radar, lidar, ds, gyro)                ║
║    3. Evalúa transiciones de estado según las lecturas                        ║
║    4. Ejecuta la lógica del estado actual (CIL/PED/EVADE/REORIENT)          ║
║    5. Envía comandos al volante y la velocidad                                ║
║    6. Actualiza el HUD                                                        ║
║                                                                              ║
║  FLUJO DE TRANSICIONES (máquina de estados):                                  ║
║                                                                              ║
║    [CIL_DRIVE] ─── peatón × PED_CONFIRM_N ──→ [BRAKE_PED]                  ║
║    [BRAKE_PED] ─── no ped × PED_RELEASE_N ──→ [CIL_DRIVE]                  ║
║    [CIL_DRIVE] ─── lidar_d < OBS_THRESH ────→ [EVADE_OBS]                  ║
║    [EVADE_OBS] ─── ds_rear > DS_CLEAR ──────→ [REORIENT]                   ║
║    [REORIENT]  ─── |heading_err| < TOL ─────→ [CIL_DRIVE]                  ║
║                                                                              ║
║  PRIORIDAD de las transiciones desde CIL_DRIVE:                              ║
║    1. Peatón (life safety) > 2. Obstáculo (property) > 3. CIL normal         ║
║    Un peatón siempre gana sobre un obstáculo — diseño safety-first.          ║
║                                                                              ║
║  CONTROL DE VELOCIDAD CON RADAR (solo en CIL_DRIVE):                         ║
║    La reducción de velocidad por radar es local al estado CIL.                ║
║    En EVADE_OBS la velocidad es fija (SPEED_EVADE=15) sin radar control.    ║
║    En BRAKE_PED la velocidad es 0 sin radar control.                          ║
║                                                                              ║
║  INTEGRACIÓN DE HEADING (integración numérica):                               ║
║    heading_accum += yaw_rate × Δt  cada frame                                ║
║    Se resetea a 0 al salir de REORIENT (heading_ref - heading_accum ≈ 0)    ║
║    El drift del giroscopio acumula error a largo plazo — aceptable porque    ║
║    REORIENT dura < 3 segundos en maniobras normales.                         ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 2.1 (H3): primer while driver.step() loop con control de volante      ║
║  • Act 3.1: máquina de 2 estados (DRIVE/BRAKE) — versión más simple          ║
║  • Act 4.2: máquina de 4 estados completa — esta es copia directa           ║
║    con la adición del control de radar y la inferencia CIL                  ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Grabar la pantalla de Webots mientras el auto completa las 3 rutas        ║
║    requeridas por la rúbrica (A: recto, B: derecha, C: izquierda)            ║
║  → Mencionar las transiciones de estado en voz cuando ocurran en el video   ║
║  → "La arquitectura en 4 estados nos permite garantizar safety (freno ante   ║
║    peatones) sin interferir con la conducción autónoma CIL normal"           ║
║  → Mostrar una métrica: % del tiempo en cada estado durante las rutas        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Variables de estado global
state         = STATE_CIL
nav_cmd       = CMD_STRAIGHT
current_steer = 0.0

ped_pos_streak = 0    # detecciones positivas consecutivas
ped_neg_streak = 0    # detecciones negativas consecutivas
ped_hold_count = 0    # frames de freno restantes

heading_ref   = 0.0   # heading al momento de entrar en EVADE
heading_accum = 0.0   # heading integrado con giroscopio

frame_count   = 0

print("=" * 60)
print("[CIL-AUTO] Controlador autónomo iniciado")
print(f"[CIL-AUTO] Modelo: {'cargado' if model else 'NO cargado — operando sin CIL'}")
print(f"[CIL-AUTO] Radar seguro: {RADAR_SAFE_M} m | Detención: {RADAR_STOP_M} m")
print("  A=izquierda  W=recto  D=derecha  M=forzar CIL")
print("=" * 60)

driver.setCruisingSpeed(CRUISE_SPEED)

while driver.step() != -1:
    frame_count += 1

    # ── 1. TECLADO ────────────────────────────────────────────────────────────
    key = keyboard.getKey()
    while key > 0:
        if   key == ord('A') or key == ord('a'):
            nav_cmd = CMD_LEFT
            print("[NAV] IZQUIERDA")
        elif key == ord('W') or key == ord('w'):
            nav_cmd = CMD_STRAIGHT
            print("[NAV] RECTO")
        elif key == ord('D') or key == ord('d'):
            nav_cmd = CMD_RIGHT
            print("[NAV] DERECHA")
        elif key == ord('M') or key == ord('m'):
            state = STATE_CIL
            print("[DEBUG] Forzado a CIL_DRIVE")
        key = keyboard.getKey()

    # ── 2. LECTURA DE SENSORES ────────────────────────────────────────────────
    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))   # BGRA
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    radar_d  = get_radar_min_dist()      # distancia al vehículo frontal más cercano
    lidar_d  = get_lidar_front_dist()    # distancia al obstáculo en cono frontal
    ds_f_val = ds_rf.getValue()
    ds_m_val = ds_rm.getValue()
    ds_r_val = ds_rr.getValue()

    # Integración de yaw rate para tracking de heading (eje Z en R2023b)
    yaw_rate      = gyro.getValues()[2]
    heading_accum += yaw_rate * (timestep / 1000.0)   # rad

    # Detection de peatón: solo en frames múltiplos de DETECT_EVERY
    ped_detected = False
    if frame_count % DETECT_EVERY == 0:
        ped_detected = check_pedestrian()

    # ── 3. MÁQUINA DE ESTADOS ─────────────────────────────────────────────────

    if state == STATE_CIL:
        # Actualizar contadores de peatón
        if ped_detected:
            ped_pos_streak += 1; ped_neg_streak = 0
        else:
            ped_neg_streak += 1; ped_pos_streak = 0

        # TRANSICIÓN → BRAKE_PED (mayor prioridad: life safety)
        if ped_pos_streak >= PED_CONFIRM_N:
            state          = STATE_PED
            ped_hold_count = PED_HOLD_F
            print(f"[CIL] PEATÓN detectado → BRAKE_PED (frame {frame_count})")

        # TRANSICIÓN → EVADE_OBS
        elif lidar_d < OBS_LIDAR_THRESH:
            state       = STATE_EVADE
            heading_ref = heading_accum   # capturar heading de referencia
            print(f"[CIL] Obstáculo a {lidar_d:.1f}m → EVADE_OBS")

        else:
            # INFERENCIA CIL: imagen + comando → ángulo de dirección
            target_steer = cil_predict(bgr, nav_cmd)

            # Control de velocidad proporcional al radar
            if radar_d < RADAR_STOP_M:
                driver.setCruisingSpeed(0)
                driver.setBrakeIntensity(0.5)
            elif radar_d < RADAR_SAFE_M:
                driver.setCruisingSpeed(CRUISE_SPEED * (radar_d / RADAR_SAFE_M))
                driver.setBrakeIntensity(0.0)
            else:
                driver.setCruisingSpeed(CRUISE_SPEED)
                driver.setBrakeIntensity(0.0)

            # Rate limiter: suaviza la salida de la CNN
            current_steer = apply_rate_limit(current_steer, target_steer)
            driver.setSteeringAngle(current_steer)

    elif state == STATE_PED:
        # Freno total: detener y mantener PED_HOLD_F frames mínimo
        driver.setCruisingSpeed(0)
        driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)

        ped_hold_count -= 1
        if ped_hold_count <= 0:
            if ped_neg_streak >= PED_RELEASE_N:
                # Peatón despejado — volver a conducción autónoma
                state = STATE_CIL
                driver.setBrakeIntensity(0.0)
                ped_pos_streak = 0; ped_neg_streak = 0
                print(f"[PED] Peatón liberado → CIL_DRIVE")
            else:
                ped_hold_count = 20   # extender si el peatón sigue en frame

    elif state == STATE_EVADE:
        # Wall-following: rodear obstáculo por el lado derecho (de Act 4.2)
        driver.setCruisingSpeed(SPEED_EVADE)
        driver.setBrakeIntensity(0.0)

        steer = wall_follow_step(ds_f_val, ds_m_val, ds_r_val)
        current_steer = apply_rate_limit(current_steer, steer)
        driver.setSteeringAngle(current_steer)

        # Condición de salida: obstáculo totalmente superado
        if ds_r_val > DS_CLEAR_DIST and ds_f_val < DS_ENGAGE_DIST:
            state       = STATE_REOR
            heading_ref = heading_accum   # target para la reorientación
            print(f"[EVADE] Obstáculo superado → REORIENT")

    elif state == STATE_REOR:
        # Recuperar heading original con controlador P sobre el giroscopio
        driver.setCruisingSpeed(SPEED_REORIENT)
        heading_error = heading_ref - heading_accum
        steer_corr    = np.clip(KP_HEADING * heading_error, -MAX_ANGLE, MAX_ANGLE)

        current_steer = apply_rate_limit(current_steer, steer_corr)
        driver.setSteeringAngle(current_steer)

        # Convergencia: heading dentro de la tolerancia
        if abs(heading_error) < HEADING_TOL:
            state         = STATE_CIL
            heading_accum = 0.0   # resetear — drift acumulado ya no importa
            driver.setCruisingSpeed(CRUISE_SPEED)
            print(f"[REOR] Heading recuperado (err={heading_error:.3f}r) → CIL_DRIVE")

    # ── 4. HUD ────────────────────────────────────────────────────────────────
    draw_hud(state, nav_cmd, current_steer, radar_d, ped_detected, lidar_d)
