# OBJETIVO: Trasladar una imagen agrandando el lienzo para evitar que se recorte el contenido

import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Obtiene el número de filas y columnas de la imagen
num_rows, num_cols = img.shape[:2]

# Matriz de traslación: mueve la imagen 70px a la derecha y 110px hacia abajo
translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])

# Agranda el lienzo sumando el desplazamiento al tamaño original
# Así la tortuga no se recorta — todo cabe en el nuevo marco
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70, num_rows + 110))

cv2.imshow('Translation without cropping', img_translation)
cv2.waitKey()
