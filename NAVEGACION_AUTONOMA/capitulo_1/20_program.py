# OBJETIVO: Aplicar un efecto de onda horizontal desplazando píxeles verticalmente con una función seno

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
        offset_x = 0
        # Calcula el desplazamiento vertical usando una función seno sobre las columnas
        # Amplitud=16px, período=150 columnas — crea ondas horizontales
        offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
        if i + offset_y < rows:
            img_output[i, j] = img[(i + offset_y) % rows, j]
        else:
            img_output[i, j] = 0

cv2.imshow('Input', img)
cv2.imshow('Horizontal wave', img_output)
cv2.waitKey()
