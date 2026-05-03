# EY4 — Análisis de Contornos con OpenCV

Ejercicios sobre detección, comparación y segmentación de formas usando contornos en OpenCV.

---

## Paso 1 — Generar las imágenes de prueba

Las imágenes se generan con un script — no necesitas descargar nada.

```bash
cd ~/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/EY4_CONTORNOS
~/miniconda3/envs/ml_env/bin/python generate_images.py
```

Esto crea dos archivos en la misma carpeta:

| Archivo | Contenido |
|---|---|
| `ref_shape.png` | Fondo blanco con un boomerang negro (forma de referencia) |
| `input_shapes.png` | Fondo blanco con círculo, cuadrado, triángulo, 2 boomerangs y una estrella |

> Corre este script **una sola vez** antes de ejecutar cualquier ejercicio.

---

## Ejercicios

### EJ4 — Comparación de formas (`EJ4_contour_matching.py`)

Encuentra cuál de las formas en `input_shapes.png` se parece más al boomerang de `ref_shape.png` usando **Hu Moments**.

**Muestra 3 ventanas:**
- `1 - Referencia` — el boomerang que se usa como molde (contorno verde)
- `2 - Todos los contornos` — todas las formas detectadas en azul
- `3 - Mejor match` — la forma más parecida resaltada en **rojo** con etiqueta "MATCH"

**Cómo funciona:**
- `matchShapes` compara formas sin importar tamaño, orientación ni rotación
- Valor cercano a **0** = muy similar; valor alto = muy diferente
- En la terminal se imprime la distancia de cada contorno para ver cuál ganó

```bash
~/miniconda3/envs/ml_env/bin/python EJ4_contour_matching.py
```

---

### EJ5 — Aproximación de contornos (`EJ5_contour_approx.py`)

Reduce el número de puntos de un contorno manteniendo la forma general usando **`approxPolyDP`**.

**Muestra 2 ventanas:**
- `1 - Originales` — contornos en rojo con todos los puntos detectados
- `2 - Simplificados` — contornos en azul con menos puntos; los vértices que quedaron se marcan con puntos verdes

**Cómo funciona:**
- `factor = 0.02` controla cuánto se simplifica
  - `0.001` = casi igual al original
  - `0.05` = muy simplificado (pocos puntos)
- En la terminal se imprime cuántos puntos tenía cada versión y el % de reducción

```bash
~/miniconda3/envs/ml_env/bin/python EJ5_contour_approx.py
```

---

### EJ6 — Defectos de convexidad (`EJ6_convexity.py`)

Detecta las **cavidades** de una forma — los huecos que quedan entre el borde real y su "envoltura convexa" (como si le rodearas la forma con una liga elástica).

**Muestra 2 ventanas:**
- `1 - ANTES` — las formas originales con contornos en gris
- `2 - DESPUÉS` — líneas verdes (casco convexo) + círculos azules en el fondo de cada cavidad

**Cómo funciona:**
- El **casco convexo** es la envoltura convexa mínima de la forma
- Un **defecto** es el espacio entre el contorno y ese casco
- Solo se marcan defectos mayores a 10px para ignorar ruido
- Se aplica `approxPolyDP` antes para evitar falsos positivos
- Las formas convexas (círculo, cuadrado, triángulo) no muestran ningún círculo

```bash
~/miniconda3/envs/ml_env/bin/python EJ6_convexity.py
```

---

### EJ7 — Censura de formas con K-means (`EJ7_censoring.py`)

Identifica y censura automáticamente las formas no convexas usando el **factor de solidez** y **K-means**.

**Muestra 2 ventanas:**
- `Detected shapes` — contornos de las formas no convexas detectadas
- `Censored` — las mismas formas cubiertas con un rectángulo negro

**Cómo funciona:**
- **Solidez** = área del contorno / área del casco convexo
  - Valor cercano a **1** = forma sólida (círculo, cuadrado)
  - Valor **bajo** = forma con cavidades (boomerang, estrella)
- K-means (K=2) agrupa automáticamente las formas en dos clusters
- El cluster con menor solidez promedio = formas no convexas → se censuran

```bash
~/miniconda3/envs/ml_env/bin/python EJ7_censoring.py
```

---

### EJ8 — Segmentación con GrabCut (`EJ8_grabcut.py`)

Segmenta interactivamente un objeto del fondo usando **GrabCut**.

Usa la foto de la tortuga marina (`capitulo_1/images/input.jpg`) — fondo azul vs animal = caso ideal.

**Pasos:**
1. Dibuja un rectángulo alrededor del objeto con el mouse
2. GrabCut se aplica **automáticamente** al soltar el mouse
3. Presiona `ESC` para salir

**Muestra 2 ventanas:**
- `Input` — la foto original; mientras arrastras los colores se invierten para ver qué seleccionas
- `Output — objeto segmentado` — el objeto recortado con el fondo en negro

**Cómo funciona:**
- GrabCut usa modelos de color Gaussianos (GMM) para separar primer plano del fondo
- Es más preciso que un umbral simple porque aprende los colores del objeto
- Itera 5 veces para refinar la segmentación
- En navegación autónoma este principio se automatiza con YOLO o Mask R-CNN (sin rectángulo manual)

```bash
~/miniconda3/envs/ml_env/bin/python EJ8_grabcut.py
```

---

---

### EJ9 — Detección de obstáculos en escena sintética (`EJ9_nav_obstacles.py`)

Simula lo que hace la cámara de un vehículo autónomo: detecta vehículos, peatones y conos automáticamente usando segmentación por color HSV + contornos.

**Muestra 3 ventanas:**
- `1 - Cámara vehicular` — escena generada con asfalto, carriles y obstáculos
- `2 - Obstáculos detectados` — bounding boxes automáticos con etiqueta y distancia simulada
- `3 - Máscaras HSV` — las 3 máscaras binarias (una por tipo de obstáculo)

**Cómo funciona:**
- Convierte BGR → HSV para separar color de iluminación
- Aplica un rango HSV por cada tipo de obstáculo (vehículo, peatón, cono)
- Encuentra contornos en cada máscara y dibuja bounding boxes
- Verifica si el carril propio tiene obstáculos

```bash
~/miniconda3/envs/ml_env/bin/python EJ9_nav_obstacles.py
```

---

### EJ10 — Detección por clic en foto real (`EJ10_click_detect.py`)

Detecta objetos en cualquier foto real haciendo clic sobre ellos — el sistema encuentra automáticamente todo lo que tenga ese color.

**Pasos:**
1. Se abre la foto de la tortuga
2. Haz clic en cualquier parte de la imagen
3. El sistema detecta y enmarca en verde todo lo que tenga ese color
4. Haz clic en otra zona para buscar un color diferente
5. Presiona `ESC` para salir

**Muestra 2 ventanas al hacer clic:**
- `Resultado` — foto con bounding boxes verdes sobre los objetos detectados
- `Mascara de color` — la máscara binaria del color seleccionado

**Cómo funciona:**
- Lee el valor HSV del píxel donde hizo clic el usuario
- Crea un rango de tolerancia de ±25 alrededor de ese color
- Aplica la máscara, limpia ruido y detecta contornos
- En la terminal imprime el valor HSV exacto — útil para calibrar rangos en EJ9

```bash
~/miniconda3/envs/ml_env/bin/python EJ10_click_detect.py
```

---

## Conceptos clave

| Concepto | Descripción |
|---|---|
| **Contorno** | Curva que conecta los puntos del borde de una forma |
| **Casco convexo** | La "liga elástica" alrededor de la forma — la envoltura convexa mínima |
| **Defecto de convexidad** | Cavidad entre el contorno y el casco convexo |
| **Solidez** | Qué tan "llena" es una forma: área real / área del casco convexo |
| **Hu Moments** | 7 valores invariantes a traslación, escala y rotación — usados para comparar formas |
| **approxPolyDP** | Reduce puntos de un contorno manteniendo la forma general |
| **GrabCut** | Algoritmo iterativo que segmenta objetos usando modelos de color |
| **HSV** | Espacio de color que separa tono (Hue) de brillo — más robusto que RGB para detectar colores |
| **Segmentación por color** | Crear una máscara binaria basada en rango de color HSV |
| **Bounding box** | Rectángulo mínimo que encierra un objeto detectado |

---

## Aplicaciones en Navegación Autónoma

| Ejercicio | Aplicación |
|---|---|
| **EJ4** | Reconocimiento de señales de tráfico o marcadores desde distintos ángulos |
| **EJ5** | Preprocesamiento de contornos para reducir ruido antes de análisis |
| **EJ6** | Detectar obstáculos con formas irregulares (huecos, concavidades) |
| **EJ7** | Clasificar y filtrar objetos por forma sin entrenamiento previo |
| **EJ8** | Segmentar peatones, vehículos u objetos del entorno |
| **EJ9** | Simular percepción de cámara vehicular con detección automática de obstáculos |
| **EJ10** | Calibrar rangos de color HSV haciendo clic en objetos de fotos reales |
