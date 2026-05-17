# OBJETIVO: Aplicar dos traslaciones consecutivas para centrar la imagen en un marco más grande

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Primera traslación: mueve la imagen 70px a la derecha y 110px hacia abajo
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70, num_rows + 110))

# Segunda traslación: mueve la imagen 30px a la izquierda y 50px hacia arriba
# Los valores negativos invierten la dirección del desplazamiento
# El lienzo crece para acomodar ambas traslaciones y centrar la imagen
translation_matrix = np.float32([[1, 0, -30], [0, 1, -50]])
img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + 70 + 30, num_rows + 110 + 50))

cv2.imshow('Translation', img_translation)
cv2.waitKey()
