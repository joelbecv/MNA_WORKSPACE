# OBJETIVO: Aplicar una transformación afín para deformar la imagen en forma de paralelogramo usando 3 puntos de control

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

# Define 3 puntos de control en la imagen original
# Esquina superior izquierda, superior derecha e inferior izquierda
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])

# Define a dónde se mapean esos 3 puntos en la imagen de salida
# La esquina superior derecha se mueve hacia adentro, creando efecto de paralelogramo
dst_points = np.float32([[0, 0], [int(0.6 * (cols - 1)), 0], [int(0.4 * (cols - 1)), rows - 1]])

# Calcula la matriz de transformación afín a partir de los 3 pares de puntos
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

# Aplica la transformación afín — las líneas se mantienen rectas pero los ángulos cambian
img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
