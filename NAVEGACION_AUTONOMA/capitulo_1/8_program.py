# OBJETIVO: Trasladar una imagen dentro del mismo marco (la parte que sale del borde se recorta)

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Crea la matriz de traslación: mueve la imagen 70px a la derecha y 110px hacia abajo
# Formato: [[1, 0, tx], [0, 1, ty]]  donde tx=desplazamiento horizontal, ty=vertical
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])

# Aplica la traslación manteniendo el mismo tamaño de imagen
# La parte de la imagen que se sale del borde se recorta y se pierde
# cv2.INTER_LINEAR define el método de interpolación para rellenar los píxeles nuevos
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR)

cv2.imshow('Translation', img_translation)
cv2.waitKey()
