# OBJETIVO: Trasladar una imagen rellenando el borde vacío con repetición de la imagen (efecto mosaico)

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('./images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Matriz de traslación: mueve la imagen 70px a la derecha y 110px hacia abajo
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])

# Aplica la traslación con relleno de bordes
# cv2.BORDER_WRAP: rellena el espacio vacío repitiendo la imagen (efecto mosaico)
# El último argumento (1) es borderValue, usado cuando el modo es BORDER_CONSTANT
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR, cv2.BORDER_WRAP, 1)

cv2.imshow('Translation', img_translation)
cv2.waitKey()
