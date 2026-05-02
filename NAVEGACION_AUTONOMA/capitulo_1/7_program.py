# OBJETIVO: Separar y recombinar canales de color para obtener diferentes efectos visuales

import cv2

# Lee la imagen en color (formato BGR: Azul, Verde, Rojo)
img = cv2.imread('./images/input.jpg', cv2.IMREAD_COLOR)

# Separa los 3 canales de color en variables individuales
g, b, r = cv2.split(img)

# Recombina los canales en orden diferente para obtener distintos efectos de color
gbr_img = cv2.merge((g, b, r))  # Verde-Azul-Rojo
rbr_img = cv2.merge((r, b, r))  # Rojo-Azul-Rojo (elimina el canal verde)

# Muestra las 3 versiones para comparar
cv2.imshow('Original', img)
cv2.imshow('GBR', gbr_img)
cv2.imshow('RBR', rbr_img)
cv2.waitKey()
