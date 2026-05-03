# EJ2 — Detección de Bordes en OpenCV
## Navegación Autónoma — Maestría en Inteligencia Artificial

Ejercicio basado en:
- Video: https://www.youtube.com/watch?v=aDY4aBLFOIg
- Código base: https://gist.github.com/pknowledge/2402470cd53d15639c67522c4d4f4868

Tema: **Edge Detection** — detección de bordes, uno de los pasos fundamentales en visión por computadora para identificar objetos, formas y límites en una imagen.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `EJ2.PY` | Comparación de técnicas de detección de bordes |
| `lena.jpg` | Imagen clásica de referencia en procesamiento de imágenes |

---

## Cómo correrlo

```bash
cd /Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/EY2_DETBORDES
~/miniconda3/envs/ml_env/bin/python EJ2.PY
```

Muestra una cuadrícula con 5 imágenes comparando cada técnica de detección de bordes.

---

## Técnicas de detección de bordes comparadas

| Técnica | Función | Cómo funciona |
|---|---|---|
| Original | — | Imagen en escala de grises sin procesar |
| Laplacian | `cv2.Laplacian` | Segunda derivada — detecta bordes en todas las direcciones |
| Sobel X | `cv2.Sobel(1,0)` | Primera derivada horizontal — detecta bordes verticales |
| Sobel Y | `cv2.Sobel(0,1)` | Primera derivada vertical — detecta bordes horizontales |
| Sobel Combinado | `cv2.bitwise_or` | Une Sobel X y Y — detecta bordes en ambas direcciones |

---

## Conceptos clave

- **Derivada**: mide qué tan rápido cambia la intensidad de los píxeles — donde hay un borde, el cambio es brusco
- **cv2.CV_64F**: se usa precisión de 64 bits porque los valores de la derivada pueden ser negativos
- **np.absolute**: convierte los valores negativos a positivos para poder visualizarlos
- **bitwise_or**: combina dos imágenes tomando el valor máximo de cada píxel

---

## Aplicación en Navegación Autónoma

La detección de bordes se usa para:
- Detectar carriles en la carretera
- Identificar obstáculos y objetos
- Reconocer señales de tráfico
- Determinar los límites del camino
