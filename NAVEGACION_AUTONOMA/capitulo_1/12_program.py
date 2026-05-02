# OBJETIVO: Rotar una imagen alrededor de su centro con reducción de escala

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Crea la matriz de rotación
# Argumentos: (punto central de rotación, ángulo en grados, escala)
# Centro = mitad de la imagen, 30 grados, escala 0.7 (reduce la imagen al 70%)
rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 0.7)

# Aplica la rotación manteniendo el mismo tamaño de imagen
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

cv2.imshow('Rotation', img_rotation)
cv2.waitKey()
