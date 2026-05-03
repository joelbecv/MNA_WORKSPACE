# Ejercicios YouTube — Filtros de Suavizado en OpenCV
## Navegación Autónoma — Maestría en Inteligencia Artificial

Ejercicios basados en el video: https://www.youtube.com/watch?v=u3poUhCxx4k

Tema: **Image Smoothing / Blurring** — técnicas para reducir ruido en imágenes, muy usadas en visión por computadora antes de detectar bordes u objetos.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `ej1.py` | Comparación de 5 técnicas de suavizado sobre una imagen |
| `water.png` | Imagen de prueba usada en el ejercicio |
| `lena.jpg` | Imagen clásica de referencia en procesamiento de imágenes |
| `opencv-logo.png` | Logo de OpenCV |
| `Halftone_Gaussian_Blur.jpg` | Ejemplo de imagen con patrón halftone |

---

## Cómo correrlo

```bash
cd /Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/ejercicio_youtube
~/miniconda3/envs/ml_env/bin/python ej1.py
```

Muestra una ventana con 6 imágenes comparando cada filtro lado a lado.

---

## Técnicas de suavizado comparadas

| Técnica | Función | Característica |
|---|---|---|
| Original | — | Imagen sin procesar |
| 2D Convolution | `cv2.filter2D` | Aplica un kernel personalizado de promedio 5x5 |
| Blur | `cv2.blur` | Promedio simple de píxeles vecinos |
| Gaussian Blur | `cv2.GaussianBlur` | Promedio ponderado — más peso al centro, más suave |
| Median Blur | `cv2.medianBlur` | Reemplaza cada píxel con la mediana de sus vecinos — ideal para eliminar ruido tipo sal y pimienta |
| Bilateral Filter | `cv2.bilateralFilter` | Suaviza preservando bordes — el más lento pero mejor calidad |

---

## Cuándo usar cada filtro

- **Gaussian Blur** — uso general, antes de detectar bordes (Canny)
- **Median Blur** — cuando la imagen tiene ruido tipo "sal y pimienta"
- **Bilateral Filter** — cuando se necesita suavizar sin perder bordes importantes
- **2D Convolution** — cuando se quiere un kernel personalizado

---

## Aplicación en Navegación Autónoma

El suavizado es un paso de **preprocesamiento** previo a la detección de objetos, carriles o señales — reduce el ruido del sensor de cámara para que los algoritmos de detección funcionen mejor.
