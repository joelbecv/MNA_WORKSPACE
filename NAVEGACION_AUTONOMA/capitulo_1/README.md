# Capítulo 1 — Introducción a OpenCV
## Navegación Autónoma — Maestría en Inteligencia Artificial

Este capítulo cubre los fundamentos de procesamiento de imágenes con OpenCV en Python.
Cada programa es independiente y puede correrse con:

```bash
~/miniconda3/envs/ml_env/bin/python <nombre_programa>.py
```

> Asegúrate de correr cada programa desde la carpeta `capitulo_1` para que encuentre la imagen en `images/input.jpg`.

---

## 1. Carga y visualización de imágenes

| Programa | Descripción |
|---|---|
| `1_program.py` | Cargar y mostrar una imagen |
| `2_program.py` | Convertir a escala de grises y guardar como JPG |
| `3_program.py` | Guardar una imagen cambiando su formato a PNG |

---

## 2. Espacios de color

| Programa | Descripción |
|---|---|
| `4_program.py` | Convertir imagen de color a escala de grises con `cvtColor` |
| `5_program.py` | Convertir imagen al espacio de color YUV |
| `6_program.py` | Separar canales Y, U, V usando `cv2.split` y slicing NumPy |
| `7_program.py` | Separar y recombinar canales BGR para obtener efectos de color |

---

## 3. Transformaciones geométricas

### Traslación
| Programa | Descripción |
|---|---|
| `8_program.py` | Traslación con recorte de bordes |
| `9_program.py` | Traslación agrandando el lienzo para evitar recorte |
| `10_program.py` | Dos traslaciones consecutivas para centrar la imagen |
| `11_program.py` | Traslación con relleno de bordes en mosaico (`BORDER_WRAP`) |

### Rotación
| Programa | Descripción |
|---|---|
| `12_program.py` | Rotación alrededor del centro con reducción de escala |
| `13_program.py` | Rotación combinada con traslación en lienzo doble |

### Escalado
| Programa | Descripción |
|---|---|
| `14_program.py` | Escalado con interpolación lineal, cúbica y por área |

### Transformaciones afines
| Programa | Descripción |
|---|---|
| `15_program.py` | Transformación afín para deformar la imagen en paralelogramo |
| `16_program.py` | Imagen espejo horizontal usando transformación afín |

### Transformaciones proyectivas (Homografía)
| Programa | Descripción |
|---|---|
| `17_program.py` | Perspectiva comprimiendo la parte inferior de la imagen |
| `18_program.py` | Efecto de perspectiva comprimiendo los bordes verticales |

---

## 4. Efectos de deformación (Image Warping)

| Programa | Descripción |
|---|---|
| `19_program.py` | Onda vertical — desplazamiento horizontal con función seno |
| `20_program.py` | Onda horizontal — desplazamiento vertical con función seno |
| `21_program.py` | Onda multidireccional — combinación de seno y coseno |
| `22_program.py` | Efecto cóncavo — curva la imagen hacia adentro |

---

## Imagen utilizada
La imagen de prueba `images/input.jpg` debe estar en la carpeta `capitulo_1/images/`.
