# Historial de Desarrollo — Actividad 3.1: Detección de Peatones con SVM
**Maestría en Inteligencia Artificial — Navegación Autónoma — Semana 5**

---

## Objetivo de la Actividad

Desarrollar un controlador para el automóvil BMW en Webots que:
1. Siga el carril amarillo de forma autónoma mediante control PID.
2. Detecte peatones en tiempo real usando un clasificador SVM + descriptor HOG.
3. Ejecute un freno de emergencia al confirmar la presencia de un peatón en el carril.

**Entorno:** Webots R2025a · macOS · BMW con `Driver()` API · Mundo `city_2025a`

---

## Archivos Finales del Proyecto

| Archivo | Rol |
|---|---|
| `controllers/simple_controller_stv2.py` | Controlador base funcional — SVM+PID sin LiDAR |
| `controllers/simple_controller_stv3.py` | **Controlador activo** — SVM+PID+LiDAR (Sesión 5) |
| `worlds/city_2025a_activity_3_1.wbt` | Mundo para stv2 (velocidades de peatones reducidas) |
| `worlds/city_2025a_lidar.wbt` | Mundo para stv3 (incluye SickLms291 en el BMW) |
| `../../pedestrian_svm.joblib` | Modelo SVM entrenado (INRIA Person Dataset) |
| `controllers/versiones_previas/` | Versiones anteriores archivadas |

---

## Arquitectura del Controlador Final

```
Cámara BGRA
    │
    ├─► [Módulo PID — Seguimiento de Carril]
    │       HSV → máscara amarilla → dilate → Canny
    │       → ROI trapezoidal → HoughLinesP
    │       → centroide izq/der → error → PID → SteeringAngle
    │
    ├─► [Módulo SVM — Detección de Peatones]  (cada 5 frames)
    │       ROI vertical 51%–71%
    │       → escala ×5 → ventana deslizante 64×128 px
    │       → HOG (924 features) → decision_function()
    │       → score ≥ 0.30 = ventana positiva
    │       → CONFIRM_N=2 scans positivos = PEATÓN CONFIRMADO
    │
    └─► [Lógica de Control]
            threat=pedestrian → freno=1.0, velocidad=0
            threat=none       → freno=0.0, velocidad=30 km/h
```

---

## Historial Cronológico de Sesiones

---

### Sesión 1 — Configuración inicial y problemas del mundo

**Controlador de partida:** `simple_controller_pedestrian_v1.py`
**Mundo:** `city_2025a_activity_3_1.wbt` (original del profesor)

#### Problema 1.1 — El barril bloquea el avance permanentemente

**Causa:** El `supervisor_controller` recalcula la posición del barril en función del vehículo cada ~2 s. El barril siempre reaparece enfrente. El auto frena, el barril desaparece, vuelve a aparecer antes de que el auto pueda avanzar → bucle infinito.

**Solución:** Se creó `city_2025a_sin_barril.wbt` eliminando el nodo `OilBarrel` y el supervisor del barril.

---

#### Problema 1.2 — Peatones cruzan indefinidamente sin posibilidad de pausarlos

**Causa:** El PROTO `Pedestrian` calcula su posición como `posición = tiempo_simulación × velocidad`. El tiempo de simulación nunca se detiene. Un supervisor externo que intentara "congelar" al peatón era sobreescrito en cada timestep por el controlador interno del PROTO.

**Intento fallido:** Se creó `pedestrian_pause_supervisor` que llamaba `setSFVec3f` para fijar la posición. Resultado: efecto "fantasma" — la figura seguía moviéndose aunque la posición física estuviera congelada.

**Solución:** Se creó `pedestrian_with_pause` — un controlador propio que replica la lógica del PROTO pero agrega un campo `paused_time`. Al cruzar un waypoint, el peatón se detiene 20 s acumulando tiempo pausado sin avanzar. Usado en `city_2025a_pausa_peatones.wbt`.

---

#### Problema 1.3 — LiDAR frena ante peatones en la banqueta

**Causa:** El cono de ±25° y rango de 20 m del Sick LMS 291 cubre las aceras laterales. Un peatón parado a 15 m de distancia en la orilla activa `estado = "BARRIL"` aunque no esté en el carril.

**Solución:** Reducir el cono a ±10° y el rango a 12 m en `simple_controller_pedestrian.py` para detectar solo objetos directamente enfrente.

---

#### Problema 1.4 — SVM detecta peatones fuera del carril

**Causa:** La ventana deslizante recorría el 100% de la imagen incluyendo cielo, edificios y banquetas. Siluetas de peatones en las aceras activaban detecciones aunque no representaran peligro real.

**Solución:** Restringir el barrido a x=20%–80%, y=40%–90% de la imagen.

---

### Sesión 2 — Diagnóstico del freeze y rediseño del controlador

**Fecha:** 2025-05-22
**Controlador:** `simple_controller_pedestrian.py` (v1 corregida)
**Mundo:** `city_2025a_v2.wbt` (4 peatones, velocidad 3.0 m/s)

#### Problema 2.1 — Freeze macOS (beachball) al habilitar el LiDAR

**Causa identificada:** Llamar `lidar.enable(timestep)` con cualquier intervalo (probados: 10 ms, 32 ms, 100 ms, 500 ms) congela Webots en macOS. El Sick LMS 291 genera 361–720 rayos por barrido; el motor de física de Webots R2025a en macOS no puede procesarlos sin bloquear el hilo principal.

**Intentos fallidos:**
- `lidar.enable(timestep)` → freeze
- `lidar.enable(100)` → freeze
- `lidar.enablePointCloud()` → freeze inmediato
- Threading para aislar la lectura → el freeze era del motor de física, no del controlador

**Solución definitiva:** `lidar = None`. El LiDAR se documenta en el código como "Sección 2-B" con instrucciones para activarlo en Linux/Windows, pero se deshabilita completamente en runtime macOS.

---

#### Problema 2.2 — Error "only one Robot instance"

**Causa:** Se intentó instanciar `Car()` y `Driver()` al mismo tiempo en el mismo proceso controlador. Webots solo permite una instancia de `Robot` (o sus subclases) por proceso.

**Solución:** Usar únicamente `Driver()`. La clase `Driver` hereda de `Car` que hereda de `Robot` — cubre motor, dirección, frenos y todos los sensores en una sola instancia.

```python
# ❌ Incorrecto
car    = Car()
driver = Driver()

# ✅ Correcto
driver = Driver()
```

---

#### Problema 2.3 — Freeze por multiplicador en el timestep

**Causa:** Usar `robot.getBasicTimeStep() * 2` o `* 3` como timestep del loop causa freeze en el mundo `city`. El controlador `simple_controller_H3.py` de semana 2 que funcionaba usaba el timestep sin multiplicador.

**Solución:**
```python
# ❌ Causa freeze
timestep = int(driver.getBasicTimeStep()) * 2

# ✅ Correcto
timestep = int(driver.getBasicTimeStep())
```

---

#### Problema 2.4 — Domain Gap INRIA vs Webots

**Descripción:** El modelo SVM fue entrenado con el INRIA Person Dataset (fotografías reales de personas). Los modelos 3D de Webots tienen texturas sintéticas planas, sin sombras ni variación de iluminación natural. Los descriptores HOG de ambos dominios son estadísticamente diferentes.

**Evidencia medida en consola:**

| Objeto | Score decision_function |
|---|---|
| Fondo, edificios | -0.40 a +0.10 |
| Postes de semáforo | +0.10 a +0.20 |
| Peatones Webots 3D | +0.10 a +0.39 |
| Personas reales (INRIA) | > +1.0 (nunca alcanzado en Webots) |

**Implicación:** El umbral estándar de 0.0 generaba decenas de falsos positivos por frame. Fue necesario calibrar empíricamente el umbral contra las distribuciones reales observadas en Webots.

**Solución temporal:** Umbral calibrado = 0.30 (separa el máximo de fondo 0.19 del mínimo de peatones 0.30).
**Solución real (pendiente):** Reentrenar el SVM con capturas del propio Webots usando `collect_data.wbt` + `retrain_svm.py`.

---

### Sesión 4 — Intento de activación del LiDAR (2026-05-23)

#### Problema 4.1 — Segmentation fault al habilitar el LiDAR

**Descripción:** Se intentó activar el sensor Sick LMS 291 en el controlador `simple_controller_stv3.py` con el mundo `city_2025a_lidar.wbt`. El controlador crashea con `segmentation fault` inmediatamente al ejecutar `lidar.enable(timestep)`.

**Evidencia en consola:**
```
[OK] Modelo SVM cargado
[LiDAR] FOV=3.14 rad  rayos=180  cono=±12.5°  max=15.0 m
zsh: segmentation fault  python
```

**Hallazgo adicional:** `rayos=180` en lugar de los 36 configurados en el .wbt confirma que el PROTO `SickLms291` **no expone `horizontalResolution` como campo editable**. El campo fue aceptado por el parser del .wbt pero ignorado en runtime — el sensor siempre inicializa con 180 rayos internamente.

**Diagnóstico:** El segfault ocurre en la capa de integración C++ de Webots antes de que Python llegue a leer ningún dato. No es un problema de volumen de cómputo ni de hilos — el sensor simplemente no es compatible con macOS en esta versión (Webots R2025a). Intentos previos en sesiones anteriores ya habían producido freeze (beachball) con `lidar.enable()`; en esta sesión el crash es más severo (segfault).

**Conclusión:** El LiDAR Sick LMS 291 no puede usarse en macOS con Webots R2025a. El código de integración queda documentado en `simple_controller_stv3.py` (Sección 2-B) para activarlo en Linux/Windows.

**Estado de stv3:** Se aplicó fallback (`lidar = None`, `lidar_reader = None`). El controlador `stv3` es funcionalmente idéntico a `stv2` en macOS, con el código LiDAR preservado y comentado para otras plataformas.

---

### Sesión 5 — Activación del LiDAR con Rosetta 2 (2026-05-23)

**Controlador:** `simple_controller_stv3.py`
**Mundo:** `city_2025a_lidar.wbt`
**Plataforma:** macOS Apple Silicon M5

---

#### Problema 5.1 — Segmentation fault con LiDAR (resuelto con Rosetta 2)

**Descripción:** `lidar.enable(timestep)` causaba segfault inmediato en macOS Apple Silicon. El crash ocurre en la capa C++ de Webots antes de que Python llegue a leer datos.

**Causa raíz:** Webots R2025a fue compilado para Intel x86. En Apple Silicon corre vía Rosetta 2 (traducción de instrucciones en tiempo real). La integración C++ del LiDAR tiene código que genera segfault bajo esa traducción.

**Solución:** Abrir Webots explícitamente con Rosetta 2:
- Finder → Webots.app → Cmd+I → ✅ "Abrir con Rosetta"
- Referencia: Issue [#5282 cyberbotics/webots](https://github.com/cyberbotics/webots/issues/5282)

Con Rosetta activo, `lidar.enable()` y `getRangeImage()` funcionan correctamente desde el loop principal.

---

#### Problema 5.2 — Beachball (freeze) con LiDAR activo

**Descripción:** Con LiDAR habilitado, el simulador se congelaba después de 30–60 segundos de operación aunque Rosetta eliminó el segfault.

**Diagnóstico:** Se identificaron **dos causas independientes**:

**Causa A — BmwX5 PROTO reproduciendo audio inválido en cada frame (causa principal):**
El PROTO del BMW intenta reproducir un archivo de sonido del motor (`engine_speaker`) cuya ruta es inválida en Webots R2025a. El error se procesaba en **cada timestep** (cada 10 ms), generando carga acumulativa en el thread principal que eventualmente bloqueaba la simulación.

```
WARNING: DEF VEHICLE BmwX5 > Speaker "engine_speaker": Impossible to play ''
WARNING: Invalid URL '/Applications/Webots.app/Contents/projects/default/libraries/vehicle/'
```

**Solución A:** Sobrescribir el campo `engineSound` en el nodo BmwX5 del .wbt:
```
DEF VEHICLE BmwX5 {
  engineSound ""   ← desactiva el speaker del motor
  ...
}
```
Aplicado en `city_2025a_lidar.wbt` y `city_2025a_activity_3_1.wbt`.

**Causa B — `getRangeImage()` bloqueante en Rosetta:**
Ocasionalmente (no en cada frame), la llamada bloquea el hilo principal más tiempo del tolerado por macOS → beachball intermitente. Mitigado con `LIDAR_EVERY=20` (leer cada 200 ms en lugar de cada frame).

**Resultado:** Con ambas correcciones el sistema corre de forma estable y continua.

---

#### Problema 5.3 — Falsos positivos SVM en líneas amarillas paralelas (curvas)

**Descripción:** Al salir de curvas, las líneas amarillas del carril aparecen en ángulo dentro del ROI y generan scores HOG que superan el umbral de 0.30, activando falsas detecciones de peatón.

**Solución:** Pre-filtro de color por ventana — antes de calcular HOG, se verifica si la ventana contiene más del 15% de píxeles amarillos (HSV). Si sí, se descarta sin correr el clasificador. Las personas no son amarillas.

```python
win_hsv = cv2.cvtColor(win_bgr, cv2.COLOR_BGR2HSV)
if cv2.inRange(win_hsv, YELLOW_LOW, YELLOW_HIGH).mean() > 15:
    continue  # ventana dominada por amarillo → no es persona
```

Beneficio adicional: reduce el número de ventanas que llegan al HOG → menor carga de cómputo.

---

#### Problema 5.4 — Peatones en cruce desvían la trayectoria del auto

**Descripción:** En los cruces con múltiples peatones, los contornos de las siluetas entraban en el Canny gris y generaban líneas Hough falsas que competían con las líneas del carril, desviando el steering hacia los peatones.

**Causa:** En stv3 se había quedado el Canny combinado (gris + amarillo), perdiendo la corrección que ya existía en stv2.

**Solución:** Revertir a Canny exclusivo sobre la máscara amarilla (igual que stv2):
```python
# ❌ stv3 inicial: capturaba bordes de peatones
edges = cv2.bitwise_or(cv2.Canny(grey, 50, 150), cv2.Canny(ymask, 50, 150))

# ✅ stv3 corregido: solo líneas amarillas del carril
edges = cv2.Canny(ymask, 50, 150)
```

---

#### Problema 5.5 — Falsos positivos LiDAR: "BARRIL" en ciudad sin barriles

**Descripción:** El texto "BARRIL" aparecía en pantalla cuando el LiDAR detectaba edificios o postes a menos de 20 m. El mundo `city_2025a_lidar.wbt` no tiene barriles activos.

**Causa A:** `LIDAR_MAX_M=20m` alcanzaba fachadas de edificios y postes de semáforo. Reducido a 10 m.
**Causa B:** El nombre "barril" era herencia del mundo con barril de sesiones anteriores.

**Solución:** Renombrar `threat='barrel'` → `threat='objeto'` y texto display `"BARRIL"` → `"OBJETO"`. A esta distancia el LiDAR detecta cualquier obstáculo real en el carril (peatón, vehículo, poste muy cercano).

---

#### Parámetros finales Sesión 5

| Parámetro | Valor anterior (stv2) | Valor final (stv3) | Razón |
|---|---|---|---|
| LiDAR | deshabilitado | **habilitado** | Rosetta 2 + engineSound fix |
| `LIDAR_EVERY` | — | **20** | Reduce llamadas bloqueantes |
| `LIDAR_MAX_M` | — | **10.0 m** | Evita edificios/postes de ciudad |
| `LIDAR_EMERGENCY_M` | — | **5.0 m** | Freno inmediato sin confirmación |
| `LIDAR_CONFIRM` | — | **2** | 2 lecturas consecutivas para activar |
| `DETECT_EVERY` | 5 | **10** | Reduce carga SVM |
| `DISPLAY_EVERY` | 1 | **3** | Reduce carga de render |
| `CONFIRM_N` | 2 | **2** | Sin cambio (pre-filtro amarillo reemplazó CONFIRM_N=3) |
| Canny PID | gris+amarillo | **solo amarillo** | Siluetas en cruces no desvían steering |
| Pre-filtro amarillo SVM | ninguno | **>15% → skip** | Elimina falsos positivos en curvas |
| `engineSound` | (no configurado) | **""** | Fix crítico del beachball |

---

### Sesión 3 — Calibración del controlador v2

**Controlador:** `simple_controller_pedestrian_v2.py`
**Mundo:** `city_2025a_activity_3_1.wbt`

Esta sesión consistió en un ciclo iterativo de prueba y ajuste para eliminar falsos positivos y alcanzar detección confiable. Se describen las problemáticas en orden de aparición.

---

#### Problema 3.1 — Falsos positivos: árboles ciprés

**Descripción:** Los cipreses al borde del camino generaban scores de ~0.25–0.30 con el umbral original de 0.80. Sus gradientes HOG (forma vertical, bordes laterales) imitan parcialmente la silueta humana.

**Solución:** Ajuste de ROI vertical para bajar la zona de escaneo. Los cipreses quedan mayormente fuera del rango de alturas del barrido.

---

#### Problema 3.2 — Falsos positivos: postes de semáforo

**Descripción:** Los postes verticales de semáforo generaban scores de ~0.20 con el umbral provisional. El HOG de un poste gris vertical es similar al del torso de una persona: gradientes fuertes en la dirección vertical.

**Evidencia visual:** Imagen capturada en sesión mostrando cuadro rojo SVM sobre un poste gris con score=0.203.

**Solución:** Reducir el ROI vertical para que solo cubra la zona media de la imagen donde aparecerían peatones reales en el carril, alejando la ventana de los postes que quedan más arriba.

---

#### Problema 3.3 — Falsos positivos: líneas amarillas del carril

**Descripción:** Las líneas pintadas en el centro del carril generaban scores de ~0.20–0.28, suficiente para activar falsos positivos cuando el umbral estaba en 0.20. Los gradientes del borde de la línea amarilla contra el asfalto crean patrones horizontales que el HOG no debería clasificar como persona, pero sí los clasifica en el dominio sintético de Webots.

**Solución:** Agregar una exclusión central en el barrido horizontal. El 42%–58% central de la imagen (donde viven las líneas amarillas) se salta completamente.

```python
if cx_skip_lo <= x < cx_skip_hi:
    continue   # saltar zona central donde están las líneas del carril
```

---

#### Problema 3.4 — Siluetas de peatones interfieren con el seguimiento de carril (PID)

**Descripción:** Cuando un peatón cruzaba frente al auto, el Canny sobre imagen en escala de grises detectaba el contorno del peatón como líneas. Estas líneas falsas entraban en el algoritmo HoughLinesP y competían con las líneas amarillas del carril, desviando el cálculo del centroide y causando que el auto girara hacia el peatón en lugar de seguir el carril.

**Solución:** Cambiar la fuente del Canny. En lugar de calcular bordes sobre la imagen completa en gris, se calcula Canny únicamente sobre la máscara HSV del color amarillo (dilatada 3×3 para rellenar gaps). Los peatones, cruces y cualquier otro elemento no-amarillo quedan completamente excluidos.

```python
# ❌ Antes: Canny sobre toda la imagen → captura peatones, cruces, todo
edges = cv2.Canny(grey, 50, 150)

# ✅ Después: Canny solo sobre máscara amarilla → solo las líneas del carril
ymask_d = cv2.dilate(ymask, np.ones((3,3), np.uint8), iterations=1)
edges   = cv2.Canny(ymask_d, 50, 150)
```

---

#### Iteraciones de calibración del ROI (resumen)

La zona de detección SVM se ajustó en múltiples ciclos. A continuación el resumen de los valores clave que evolucionaron:

| Parámetro | Valor inicial | Valor final | Razón del cambio |
|---|---|---|---|
| `SVM_THRESHOLD` | 0.80 | **0.30** | Webots scores máx ~0.39; 0.80 nunca detectaba nada |
| ROI vertical superior | 40% | **51%** | Bajar para evitar postes y edificios |
| ROI vertical inferior | 90% | **71%** | Subir para no capturar zona muy cercana (ruido) |
| ROI horizontal izq | 20% | **30%** | Reducir para excluir banquetas laterales |
| ROI horizontal der | 80% | **70%** | Reducir para excluir banquetas laterales |
| Centro excluido | ninguno | **42%–58%** | Añadido para eliminar falsos positivos de líneas amarillas |
| Scale ventana | 3.0 | **5.0** | ROI final de 26px requiere ≥128px → escala 5.0 (26×5=130) |
| `CONFIRM_N` | 4 | **2** | Con 4, el peatón salía del frame antes de confirmar |
| `HOLD_FRAMES` | 150 (~1.5 s) | **80** (~0.8 s) | Tiempo de frenado ajustado para no bloquear innecesariamente |
| Velocidad peatones | 2.5–6.0 m/s | **1.92–4.62 m/s** | Reducción ×0.9 ×0.9 ×0.95 para aumentar tiempo de detección |

---

#### Primera detección exitosa confirmada

```
[SVM] f=03245 wins=6  hits=1/1 score=0.328 thresh=0.30 pos=1/2
[SVM] f=03265 wins=6  hits=2/1 score=0.629 thresh=0.30 pos=2/2  ← CONFIRMADO
[SVM] f=03285 wins=6  hits=1/1 score=0.431 thresh=0.30 pos=2/2  threat=pedestrian
```

El auto detuvo su avance con `threat=pedestrian`, freno=1.0, velocidad=0.

---

## Estado Final (Sesión 5 — 2026-05-23)

### Funcional ✅
- Sin freeze (beachball) en macOS Apple Silicon M5
- PID sigue el carril amarillo de forma estable — Canny exclusivo sobre máscara amarilla
- SVM detecta peatones con scores 0.30–0.70, CONFIRM_N=2
- Pre-filtro amarillo elimina falsos positivos de líneas del carril en curvas
- LiDAR Sick LMS 291 activo (requiere Rosetta 2) — cono ±12.5°, máx 10 m
- Freno dual: SVM confirma peatón → `threat='pedestrian'`; LiDAR detecta obstáculo → `threat='objeto'`
- Display de diagnóstico en tiempo real (ver guía de líneas en el código)

### Controlador activo
`controllers/simple_controller_stv3.py` + `worlds/city_2025a_lidar.wbt`

### Requisito de plataforma
Webots debe abrirse con **Rosetta 2** en macOS Apple Silicon:
Finder → Webots.app → Cmd+I → ✅ "Abrir con Rosetta"

### Limitación conocida — Domain Gap INRIA vs Webots
El modelo entrenado con INRIA (fotos reales) clasifica con confianza baja en Webots (dominio sintético). Los scores rara vez superan 0.70 cuando en imágenes reales deberían ser >1.0.

**Solución pendiente:** Reentrenar con capturas del propio simulador:
```bash
# 1. Abrir collect_data.wbt en Webots → Play → esperar ~2 min
# 2. Ejecutar retrain_svm.py
cd "Actividad 3.1 - Detección de Peatones con SVM"
python retrain_svm.py
```

---

## Lecciones Aprendidas

1. **Domain gap es un problema real y medible.** Un modelo con 85% accuracy en INRIA puede no detectar nada en Webots. Los scores son comparativos solo dentro del mismo dominio.

2. **El timestep de Webots es crítico en macOS.** Multiplicadores y ciertos sensores congelan la simulación. El diagnóstico estándar es aislar si el freeze es del mundo o del controlador.

3. **Canny debe ser selectivo.** Aplicar Canny sobre imagen completa captura bordes de peatones, cruces y sombras. Para seguimiento de carril, filtrar por color amarillo antes del Canny aísla exactamente los bordes deseados.

4. **El ROI de detección requiere calibración empírica.** La relación entre tamaño del ROI, escala de upsampling y ventana HOG (128 px mínimo) define el espacio de parámetros válidos.

5. **Los sistemas de confirmación deben ser asimétricos.** Es más seguro exigir más para liberar el freno (RELEASE_N=4) que para confirmarlo (CONFIRM_N=2).

6. **Los PRO TOs de Webots pueden generar carga oculta.** El BmwX5 PROTO intentaba reproducir un audio inválido en cada timestep. Warnings repetidos por frame son señal de alerta de carga acumulativa — siempre revisar la consola de Webots antes de diagnosticar el controlador.

7. **Rosetta 2 es necesaria para el LiDAR en Apple Silicon.** Sin ella, `lidar.enable()` genera segfault en la capa C++. Con Rosetta y `engineSound ""` el LiDAR corre estable.

8. **Los pre-filtros de color son más eficientes que subir el umbral.** Filtrar ventanas SVM por contenido amarillo elimina la causa raíz de los falsos positivos en curvas sin sacrificar sensibilidad general del clasificador.

---

*Última actualización: 2026-05-23 — Sesión 5. Controlador activo: `simple_controller_stv3.py`*
