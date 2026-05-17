# OBJETIVO: Guardar una imagen cambiando su formato de JPG a PNG con compresión

import cv2

# Lee la imagen en color
img = cv2.imread('images/input.jpg')
# Guarda la imagen en formato PNG con nivel de compresión 3 (escala 0-9)
# El parámetro de compresión debe ir en par: [flag, valor]
cv2.imwrite('images/output.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
