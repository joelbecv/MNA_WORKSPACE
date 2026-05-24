# Problemática encontrada por versión — Actividad 3.1

## Archivos de referencia

| Archivo | Descripción |
|---|---|
| `simple_controller_pedestrian_v1.py` | Versión original — parámetros sin ajustar |
| `simple_controller_pedestrian.py` | Versión activa — con correcciones aplicadas |

---

## Mundos disponibles

| Mundo | Descripción |
|---|---|
| `city_2025a_activity_3_1.wbt` | Original — con barriles dinámicos y peatones continuos |
| `city_2025a_sin_barril.wbt` | Sin barriles — peatones siguen cruzando sin pausa |
| `city_2025a_pausa_peatones.wbt` | Sin barriles — peatones pausan 20 s en cada punto de giro |

---

## Problemas encontrados

### Problema 1 — El barril bloquea permanentemente el avance

**Mundo:** `city_2025a_activity_3_1.wbt`  
**Controlador:** `simple_controller_pedestrian_v1.py`

**Causa:**  
El `supervisor_controller` coloca el barril 10 metros frente al carro cada 60 pasos de simulación (~2 s). La posición del barril se recalcula en función de la posición actual del vehículo, por lo que siempre aparece enfrente sin importar cuánto avance el auto. El freno de emergencia del controlador dura 30 frames (~1 s), pero en ese tiempo el barril ya desapareció y vuelve a aparecer 2 s después — el auto nunca logra rebasarlo.

**Solución:**  
Se creó `city_2025a_sin_barril.wbt`: copia del mundo original sin el nodo `OilBarrel` ni el `supervisor_controller`.

---

### Problema 2 — Peatones cruzan indefinidamente sin pausa

**Mundo:** `city_2025a_sin_barril.wbt`  
**Controlador:** `simple_controller_pedestrian_v1.py`

**Causa:**  
El controlador interno del PROTO `Pedestrian` calcula la posición como `distancia = tiempo_real × velocidad`. El tiempo de simulación nunca se detiene, por lo que los peatones oscilan entre sus dos waypoints de forma continua. Cualquier intento de pausarlos desde un supervisor externo falla porque el controlador del peatón sobreescribe la posición en cada paso.

**Intento fallido — Supervisor externo:**  
Se creó `pedestrian_pause_supervisor` que intentaba congelar la posición del peatón vía `setSFVec3f`. El resultado fue un efecto "fantasma": la figura del peatón seguía moviéndose mientras aparecía una copia congelada. El controlador interno del peatón siempre gana porque calcula su posición a partir del tiempo real de simulación.

**Solución:**  
Se creó `pedestrian_with_pause` — controlador propio que replica el comportamiento del PROTO oficial pero introduce un "tiempo efectivo" pausable. Al detectar un cambio de segmento (cruce de waypoint), acumula `paused_time` y congela la posición durante 20 s. Usado en `city_2025a_pausa_peatones.wbt`.

---

### Problema 3 — LiDAR frena ante peatones en la banqueta

**Mundo:** `city_2025a_pausa_peatones.wbt`  
**Controlador:** `simple_controller_pedestrian_v1.py`

**Causa:**  
El cono de detección del LiDAR era de ±25° con un rango de 20 m. Esta combinación cubre las banquetas laterales: un peatón parado en la orilla a 15 m de distancia cae dentro del cono y activa el freno de emergencia (`estado = "BARRIL"`), aunque no represente un obstáculo real en el carril.

**Solución aplicada en `simple_controller_pedestrian.py`:**
```python
# v1 (original)
LIDAR_CONE_DEG = 25
LIDAR_DANGER_M = 20.0

# v2 (corregido)
LIDAR_CONE_DEG = 10    # solo detecta objetos directamente enfrente
LIDAR_DANGER_M = 12.0  # distancia reducida para ignorar objetos lejanos
```

---

### Problema 4 — SVM detecta peatones fuera del carril

**Mundo:** cualquiera  
**Controlador:** `simple_controller_pedestrian_v1.py`

**Causa:**  
La función `sliding_window_detect` recorría la imagen completa (100% ancho × 100% alto). Esto incluye cielo, edificios y banquetas laterales, donde pueden aparecer siluetas de peatones que activan detecciones falsas aunque no estén en el carril.

**Solución aplicada en `simple_controller_pedestrian.py`:**  
Se aplicó ROI en la ventana deslizante — solo escanea la zona central donde aparecerían peatones reales en el camino:
```python
x_start = int(w * 0.20)   # ignora 20% izquierdo
x_end   = int(w * 0.80)   # ignora 20% derecho
y_start = int(h * 0.40)   # ignora el cielo y parte alta
y_end   = int(h * 0.90)   # ignora la zona muy cercana al auto
```

---

## Cómo correr la simulación

```bash
# 1. Abrir el mundo deseado en Webots y presionar Play
# 2. En terminal:
cd ".../SDC_webots 3/controllers"
/Applications/Webots.app/Contents/MacOS/webots-controller simple_controller_pedestrian.py
```
