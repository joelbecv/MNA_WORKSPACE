# OBJETIVO: Aplicar un efecto de onda en ambas direcciones combinando seno y coseno simultáneamente

import cv2
import numpy as np
import math

# Lee la imagen en escala de grises
img = cv2.imread('images/input.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# Crea una imagen de salida vacía del mismo tamaño
img_output = np.zeros(img.shape, dtype=img.dtype)

for i in range(rows):
    for j in range(cols):
        # Desplazamiento horizontal con seno sobre filas
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
        # Desplazamiento vertical con coseno sobre columnas — crea efecto ondulado en dos ejes
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
        if i + offset_y < rows and j + offset_x < cols:
            img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
        else:
            img_output[i, j] = 0

cv2.imshow('Input', img)
cv2.imshow('Multidirectional wave', img_output)
cv2.waitKey()
