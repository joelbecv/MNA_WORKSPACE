# OBJETIVO: Aplicar un efecto de onda vertical desplazando píxeles horizontalmente con una función seno

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
        # Calcula el desplazamiento horizontal usando una función seno sobre las filas
        # Amplitud=25px, período=180 filas — crea ondas verticales
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        offset_y = 0
        if j + offset_x < rows:
            img_output[i, j] = img[i, (j + offset_x) % cols]
        else:
            img_output[i, j] = 0

cv2.imshow('Input', img)
cv2.imshow('Vertical wave', img_output)
cv2.waitKey()
