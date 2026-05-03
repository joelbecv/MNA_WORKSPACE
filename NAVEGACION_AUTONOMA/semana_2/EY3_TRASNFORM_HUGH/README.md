# EJ3 — Transformada de Hough para Detección de Líneas
## Navegación Autónoma — Maestría en Inteligencia Artificial

Ejercicio basado en:
- Video 1: https://www.youtube.com/watch?v=7m-RVJ6ABsY
- Video 2: https://www.youtube.com/watch?v=gbL3XKOiBvw
- Código base: https://gist.github.com/pknowledge/62ad0d100d6d4df756c0374dee501131

Tema: **Transformada de Hough** — técnica para detectar líneas rectas en una imagen, fundamental en navegación autónoma para detección de carriles.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `EJ3.py` | Detección de líneas con Transformada de Hough |
| `sudoku.png` | Imagen de prueba con muchas líneas rectas |

---

## Cómo correrlo

```bash
cd /Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/EY3_TRASNFORM_HUGH
~/miniconda3/envs/ml_env/bin/python EJ3.py
```

Muestra dos ventanas:
1. **Edges (Canny)** — los bordes detectados antes de aplicar Hough
2. **Hough Lines** — la imagen original con las líneas detectadas dibujadas en rojo

---

## Cómo funciona la Transformada de Hough

El proceso tiene 3 pasos:

**1. Escala de grises** → `cv2.cvtColor`
Simplifica la imagen a un solo canal para facilitar el procesamiento.

**2. Detección de bordes** → `cv2.Canny`
Encuentra los cambios abruptos de intensidad. Los bordes son el input de Hough.

**3. Transformada de Hough** → `cv2.HoughLines`
Cada punto de borde "vota" por todas las líneas que podrían pasar por él. Las líneas con más votos son las detectadas.

---

## Parámetros clave de HoughLines

| Parámetro | Valor usado | Significado |
|---|---|---|
| rho | 1 | Resolución de distancia en píxeles |
| theta | π/180 | Resolución angular (1 grado) |
| threshold | 200 | Mínimo de votos para considerar una línea |

Bajar el threshold detecta más líneas pero con más ruido. Subirlo detecta menos líneas pero más confiables.

---

## Aplicación en Navegación Autónoma

La Transformada de Hough es la base para:
- **Detección de carriles** en carreteras
- **Detección de cruces** y intersecciones
- **Seguimiento de líneas** en robots móviles
