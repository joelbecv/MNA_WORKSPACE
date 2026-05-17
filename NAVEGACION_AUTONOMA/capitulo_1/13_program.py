# OBJETIVO: Combinar traslación y rotación en un lienzo doble para evitar recorte de la imagen

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Primero traslada la imagen al centro de un lienzo el doble de grande
# Así hay espacio suficiente para que la rotación no recorte el contenido
translation_matrix = np.float32([[1, 0, int(0.5 * num_cols)], [0, 1, int(0.5 * num_rows)]])

# Rota 30 grados usando la esquina inferior derecha como punto de pivote
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)

# Aplica traslación sobre un lienzo 2x más grande que la imagen original
img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_cols, 2 * num_rows))

# Aplica la rotación sobre la imagen ya trasladada
img_rotation = cv2.warpAffine(img_translation, rotation_matrix, (num_cols * 2, num_rows * 2))

cv2.imshow('Rotation', img_rotation)
cv2.waitKey()
