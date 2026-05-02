# OBJETIVO: Aplicar una transformación proyectiva (homografía) para simular perspectiva usando 4 puntos de control
# Aplicaciones: realidad aumentada, rectificación de imágenes, navegación con cámara

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

# Define 4 puntos de control en la imagen original (las 4 esquinas)
# A diferencia de la transformación afín que usa 3, la proyectiva necesita 4
src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])

# Los puntos destino comprimen la parte inferior de la imagen hacia el centro
# Esto crea el efecto de perspectiva (como ver la imagen desde abajo)
dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33 * cols), rows - 1], [int(0.66 * cols), rows - 1]])

# Calcula la matriz de transformación proyectiva (homografía) a partir de los 4 pares de puntos
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Aplica la perspectiva — usa warpPerspective en lugar de warpAffine
img_output = cv2.warpPerspective(img, projective_matrix, (cols, rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
