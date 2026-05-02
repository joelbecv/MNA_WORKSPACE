# OBJETIVO: Obtener la imagen espejo horizontal usando transformación afín con 3 puntos de control

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

# Puntos de control en la imagen original: esquinas superior izquierda, superior derecha e inferior izquierda
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])

# Los puntos destino invierten horizontalmente: la esquina izquierda va a la derecha y viceversa
# Esto produce el efecto espejo sin cambiar la posición vertical
dst_points = np.float32([[cols - 1, 0], [0, 0], [cols - 1, rows - 1]])

# Calcula la matriz afín a partir del mapeo de los 3 pares de puntos
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

# Aplica la transformación — la imagen se voltea horizontalmente
img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))

cv2.imshow('Input', img)
cv2.imshow('Mirror', img_output)
cv2.waitKey()
