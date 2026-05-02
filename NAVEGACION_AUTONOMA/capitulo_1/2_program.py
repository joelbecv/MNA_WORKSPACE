# OBJETIVO: Cargar una imagen en escala de grises y guardarla como archivo

import cv2

# Lee la imagen directamente en modo escala de grises
gray_img = cv2.imread('images/input.jpg', cv2.IMREAD_GRAYSCALE)
# Muestra la imagen en gris en una ventana
cv2.imshow('Grayscale', gray_img)
# Guarda la imagen en gris como output.jpg
cv2.imwrite('images/output.jpg', gray_img)
cv2.waitKey()
