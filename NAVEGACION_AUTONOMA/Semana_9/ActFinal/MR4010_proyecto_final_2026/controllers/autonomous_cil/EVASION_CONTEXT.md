# Contexto de evasión de bus — MR4010 Proyecto Final · Equipo 25

> Archivo de referencia para no reinventar el diagnóstico en cada sesión.
> Actualizado: 2026-06-24 — dimensiones del bus corregidas desde BusSimple.proto real

---

## 1. Geometría del mundo

### Coordinate system en `city_traffic_2025_02.wbt`
El mundo usa **Z como eje vertical** (Z≈0.55 = altura de los vehículos sobre el piso).
X e Y son los dos ejes horizontales del mapa.

### BMW spawn
`translation -158.77 -195.62 0.316` → BMW arranca en **(X=-158.77, Y=-195.62)**

### Buses relevantes (vehicles con recognitionColor)
| Nombre | X | Y | Z | Rotación | Color |
|--------|---|---|---|----------|-------|
| vehicle(6) | -221.691 | -1.389 | 0.55 | 0 0 1 1.57 (90° Z) | Rojo |
| vehicle(7) | 173.496 | -60.534 | 0.55 | 0 0 1 1.57 (90° Z) | Verde |
| vehicle(8) | **-28.1** | **-195.99** | 0.55 | **sin rotación** | Naranja |

### Bus encontrado en el log: **vehicle(8)**
- Posición: X=-28.1, Y=-195.99 (aproximadamente mismo Y que el BMW)
- Sin rotación → **el bus apunta en +X** (misma dirección que el BMW)

**Dimensiones REALES del BusSimple.proto** (bounding object medido):
```
Box { size 2.64 2.9 9.73 }   ← width × height × length (en frame del bus)
translation 1.4625 0 1.35    ← centro del bbox desde origen del nodo
rotation 120° around (1,1,1)/√3 → mapea Box.Z (9.73m) → Bus.X (largo)
```
- **Largo real: 9.73m** (NO 12m — el valor anterior era incorrecto)
- **Ancho real: 2.64m** (NO 2.5m)
- Centro del bbox: +1.4625m en +X desde el origen del nodo
- **FRENTE:** X = -28.1 + 1.4625 + 4.865 = **-21.77m**
- **TRASERA:** X = -28.1 + 1.4625 - 4.865 = **-31.50m**
- **Cara izq (+Y, hacia BMW):** Y = -195.99 + 1.32 = **-194.67m**
- **Cara der (-Y):** Y = -195.99 - 1.32 = **-197.31m**
- BMW en Y=-195.62 está **dentro del ancho del bus** (−197.31 < −195.62 < −194.67) ✓
- BMW debe adelantar al bus por la **izquierda (+Y)** para salir de su camino

### Buses de intersección (NO cambian de posición)
| Bus | X | Y | Rotación | Heading |
|-----|---|---|----------|---------|
| vehicle(6) | -221.691 | -1.389 | 0 0 1 1.57 (90° Z) | +Y direction |
| vehicle(7) | 173.496 | -60.534 | 0 0 1 1.57 (90° Z) | +Y direction |

Buses de intersección están perpendiculares al BMW → si el BMW los detecta, la evasión L1→S→R1 también funciona (buscan +Y lateral izquierdo disponible).

---

## 2. Sensores y su orientación

El BMW viaja en **dirección +X**. Con este heading:
- **Izquierda** del coche = dirección **+Y** (hacia mayores Y)
- **Derecha** del coche = dirección **-Y** (hacia menores Y)
- **ds_right_front/mid/rear** apuntan en **-Y** (perpendicular derecho)
- **LiDAR frontal** (rays 70-110 = ±20° desde heading actual) mide en +X ±20°
- **LiDAR lateral** (rays 150-180) mide 60°-90° a la derecha

### Lo que cada sensor detecta durante la evasión
- `lidar_f`: esquina delantera-izquierda del bus (X=-21.77, Y=-194.67) a distancia oblicua ~10m mientras la esquina está dentro de ±20° del heading
- `ds_rr/ds_rm/ds_rf`: cara izquierda del bus (Y=-194.67) a ~2.36m lateral cuando el coche pasa junto al bus

---

## 3. Lo que muestra el log (run con código actual)

### EVADE trigger: frame 923
- `lidar_f = 26.51m` — bus a 26.51m frontal
- Todos los ds = 5.00m (fuera de rango)

### L1 (frames 923-1012, steer=-0.30 = giro **izquierda**)
- Frames 930-951: `ds_rr min = 0.38m` (primer roce — esquina trasera del bus pasa junto al coche)
- Frames 962-985: `ds_rr min = 0.63m` (balanceo trasero del coche durante el giro)
- El coche se desplaza ~4.18m en +Y (izquierda) durante L1+L2

### L2 (frames 1013-1101, steer=+0.30 = giro **derecha**, heading se restaura)
- Todos los ds = 5.00m — el bus ya pasó al lado derecho del coche
- `lidar_f` vuelve a ver el bus (que está más en frente) al final de L2: 15.26m

### Fase S (frames 1102-1321, steer=0.0)
- Coche en Y=-191.31m (desplazado 4.68m a la izquierda — confirmado por ds_rr=2.36m + cara del bus en Y=-194.67m)
- **Cara izquierda del bus** (Y=-194.67) a **2.36m** del lado derecho del coche
- `lidar_f`: empieza en 15.12m (esquina a 14.74m en X + 3.36m en Y = 15.12m oblicuo), baja y se **estabiliza en ~10m** durante TODA la S
  - La esquina delantera-izquierda del bus (X=-21.77, Y=-194.67) está a ~10m oblicuo, al borde del sector ±20°
  - El bus nunca "sale" del sector mientras el coche avanza recto
- `ds_rr` detecta características del bus cada ~35 frames: arcos de ruedas, escalones
  - Primera detección: frames 1117-1130 @ 1.35m (coche llega al nivel de la trasera del bus, X=-31.50)
  - Detecciones siguientes: frames 1157, 1192, 1228, 1263, 1298 @ 2.26-2.36m
  - Última detección: frames 1298-1307 @ 2.36m
  - Frame 1308+: todos los ds = 5.00 (bus salió del rango ds pero NO de lidar_f)
- **Frame 1322 (transición S→R1 con el bug)**: `lidar_f` salta de 10.04 a **999.00** (esquina cruza 20°)
  - Geometría: X_dist = 3.36/tan(20°) = 9.23m → **coche en X≈-31.00m** al salir S
  - Bus frente en X=-21.77m → **9.23m de bus aún por delante del coche** al salir S ← causa raíz

### Fase R1 (frames 1322-1411, steer=+0.30 = giro **derecha**)
- Frames 1322-1329: `lidar_f = 999` (bus fuera del sector, todos los ds limpios)
- **Frame 1330**: `lidar_f = 9.48m` — ¡el bus **re-entra** en el sector frontal!
  - El giro a la derecha rota el heading del coche **HACIA el bus** (el bus estaba a >20° derecha, ahora vuelve a <20°)
  - Desde frame 1330 hasta 1411: `lidar_f` cae de 9.48m → 1.02m (choque frontal)
- **Frame 1373**: `bus_area = 50721` (bus llena toda la cámara)
- **Frame 1401**: `ds_rf = 2.90m` (esquina delantera-derecha del coche toca el bus)
- **Frame 1411**: `ds_rf = 0.93m`, `lidar_f = 1.05m` (esencialmente golpeando el bus)

### Fase R2 (frames 1412-1502, steer=-0.30 = giro **izquierda**)
- Coche choca/roza la cara izquierda del bus
- Frame 1427-1428: `ds_rf = 0.08 → 0.02m` (esencialmente tocando)
- Frame 1501: `lidar_f = 999` (el coche por fin pasa la esquina del bus)
- Frame 1502: transición a RECENTER (R2 completó 90 frames)
- **Resultado**: el coche roza el bus, hay daño físico

---

## 4. Diagnóstico de la causa raíz (CONFIRMADO)

### El problema en una oración
**S sale cuando la esquina DELANTERA-IZQUIERDA del bus cruza el límite de 20°, pero el coche aún tiene 9.23m del cuerpo del bus por delante. R1 gira hacia la derecha y dirige el coche directamente contra la cara del bus.**

### Por qué la condición `lidar_f > 20m` es incorrecta como único gatillo de salida
- `lidar_f` mide la **esquina delantera-izquierda del bus** (X=-21.77, Y=-194.67) a distancia oblicua ~10m
- Al salir S (frame 1322): **X_dist = Y_dist/tan(20°) = 3.36/0.364 = 9.23m → coche en X≈-31.00m**
- Bus frontal en X=-21.77m → **9.23m de bus aún por delante del coche**
- Cuando R1 gira a la derecha (steer=+0.30), en sólo 8 frames el heading rota 3.1° y la esquina del bus re-entra en el sector ±20° a ~9.5m
- El coche entonces avanza frontalmente hacia el bus durante 61 frames más → choque a 1m

### Lo que pasó en frame 1322 (el "falso positivo")
```
Frame 1321: lidar_f=10.04, esquina del bus a 10.04m, ángulo≈20° (todavía dentro)
Frame 1322: lidar_f=999.00, esquina del bus cruzó 20° (salió del sector)
           → condición "frontal_clear" se vuelve True
           → S sale a R1 PREMATURAMENTE
```

El bus sigue estando físicamente 9.4m adelante. La condición se cumple demasiado pronto.

---

## 5. La corrección

### Qué se necesita
Cuando `lidar_f > S_CLEAR_DIST` se activa por primera vez (esquina del bus sale del sector ±20°), el coche necesita continuar **recto** por 16.7m más antes de girar a la derecha. Esto pone el coche en X≈-14.33m, con el frente del bus (X=-21.77m) 4.94m detrás antes de R1.

### Parámetro nuevo: N_EXTRA = 250 frames
- 250 frames × 0.0667 m/frame = **16.7m de recto adicional**
- Coche sale S en X≈-31.00m → entra a R1 en X≈-14.33m
- Frente del bus en X=-21.77m → **4.94m de clearance** entre trasera del coche y frente del bus ✓
- Durante R1 (90 frames girando a derecha): clearance lateral **0.61m** (cara bus Y=-194.67, lado coche Y=-194.06) ✓
- Implementado en `autonomous_cil.py` el 2026-06-24

### Lógica de salida corregida para la fase S (IMPLEMENTADA)
```python
N_EXTRA = 250  # frames rectilíneos adicionales después de lidar_f > S_CLEAR_DIST

elif _evade_sub == "S":
    steer = 0.0
    frontal_clear = lidar_f > S_CLEAR_DIST
    if frontal_clear and _evade_count >= N_STRAIGHT:
        _s_clear_count += 1
    else:
        _s_clear_count = 0    # reset si el bus vuelve al sector
    if _s_clear_count >= N_EXTRA:
        _evade_sub = "R1"; _evade_count = 0; _s_clear_count = 0
        print(f"[EVADE] S→R1  {N_EXTRA}f libres  lidar={lidar_f:.1f}m (bus superado)")
    elif _evade_count >= N_STRAIGHT_MAX:
        _evade_sub = "R1"; _evade_count = 0; _s_clear_count = 0
        print(f"[EVADE] S→R1  TIMEOUT lidar={lidar_f:.1f}m")
```

### Geometría verificada de R1 con el fix
- Coche entra a R1 en X=-14.33m (4.94m past bus front X=-21.77m, NO hay overlap en X) ✓
- R1 avanza 5.62m en X y 1.80m en -Y (hacia el bus)
- Clearance mínimo durante R1: 0.61m entre lado derecho del coche y cara del bus ✓
- R1 termina: coche en X=-8.71m, Y=-193.11m (bus frente a 13.06m atrás) ✓
- R2 espeja el movimiento, restaura Y≈-191.31m ✓

---

## 6. Estado de los 5 pasos y qué funciona

| Paso | Sub | Estado | Nota |
|------|-----|--------|------|
| 1 | L1 | ✅ Funciona | Giro izq 90 frames, desplaza ~4.18m en +Y |
| 2 | L2 | ✅ Funciona | Giro der 90 frames, restaura heading |
| 3 | S  | ✅ Fix aplicado 2026-06-24 | N_EXTRA=250 frames; coche en X=-14.33m al entrar R1 |
| 4 | R1 | ✅ Fix aplicado 2026-06-24 | Clearance 0.61m con bus, coche ya pasó el frente |
| 5 | R2 | ✅ Fix aplicado 2026-06-24 | Sin colisión esperada |
| Post | RECENTER | ✅ Funciona | Detecta línea amarilla y vuelve a CIL |

---

## 7. Parámetros completos de referencia

```python
SPEED_EVADE         = 15     # km/h
STEER_EVADE         = 0.30   # rad
N_TURN              = 90     # frames por medio-giro (L1, L2, R1, R2)
N_STRAIGHT          = 150    # frames mínimos en S antes de empezar a contar
S_CLEAR_DIST        = 20.0   # m — lidar_f debe superar esto para iniciar N_EXTRA
N_EXTRA             = 250    # frames adicionales rectilíneos DESPUÉS de que lidar_f > 20m
N_STRAIGHT_MAX      = 600    # timeout total de S (seguridad)
BUS_COOLDOWN_FRAMES = 300    # frames sin re-trigger tras EVADE
```

### Variable de estado nueva a añadir
```python
_s_clear_count = 0  # frames consecutivos con lidar_f > S_CLEAR_DIST durante S
```
Resetear en: L2→S transition, nuevo EVADE trigger.

---

## 8. Variables de estado y su resumen

```python
state            # STATE_CIL | STATE_PED | STATE_EVADE | STATE_RECENTER
_evade_sub       # "L1" | "L2" | "S" | "R1" | "R2"
_evade_count     # frames en el sub-estado actual (reset en cada transición)
_s_clear_count   # frames con frontal libre durante S (NUEVO)
_bus_cooldown    # frames de cooldown restantes post-EVADE
_bus_streak      # frames consecutivos con bus detectado por Recognition
```

---

## 9. Notas sobre el LiDAR durante las fases

| Fase | lidar_f típico | Por qué |
|------|----------------|---------|
| CIL_DRIVE (lejos) | 26-30m | Bus en sector ±20°, recto |
| L1 | 20-24m, luego 999 | Bus sale del sector al girar izq |
| L2 | 999, luego 15-23m | Bus re-entra al final al girar der |
| S | **10m constante** | Esquina delantera-izq del bus al borde del sector |
| R1 (con bug) | 999 × 8 frames, luego 9.5→1m | Giro der → bus re-entra, choque |
| R1 (con fix) | 999 continuo | Coche ya pasó el bus, no hay re-entrada |
