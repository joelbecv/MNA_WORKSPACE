# OBJETIVO: Aplicar un efecto visual con transformación proyectiva comprimiendo los bordes verticales
# Las líneas paralelas dejan de serlo — característica exclusiva de la transformación proyectiva

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

# Puntos de control: toman la mitad izquierda de la imagen
# Los bordes superior e inferior se comprimen 100px hacia el centro en el eje Y
src_points = np.float32([[0, 0], [0, rows - 1], [cols / 2, 0], [cols / 2, rows - 1]])
dst_points = np.float32([[0, 100], [0, rows - 101], [cols / 2, 0], [cols / 2, rows - 1]])

# Calcula la matriz de homografía a partir de los 4 pares de puntos
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Aplica la transformación proyectiva
img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
