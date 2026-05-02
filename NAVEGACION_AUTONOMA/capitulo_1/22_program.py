# OBJETIVO: Aplicar un efecto cóncavo curvando la imagen hacia adentro con un seno de gran amplitud

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
        # Amplitud grande (128px) y período largo (2*cols) — curva suave que dobla la imagen
        offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2 * cols)))
        offset_y = 0
        if j + offset_x < cols:
            img_output[i, j] = img[i, (j + offset_x) % cols]
        else:
            img_output[i, j] = 0

cv2.imshow('Input', img)
cv2.imshow('Concave', img_output)
cv2.waitKey()
