# OBJETIVO: Convertir una imagen de color a escala de grises usando cvtColor

import cv2

# Lee la imagen en formato color BGR
img = cv2.imread('./images/input.jpg', cv2.IMREAD_COLOR)
# Convierte el espacio de color de RGB a escala de grises
# cvtColor permite convertir entre cualquier par de espacios de color disponibles en OpenCV
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey()
