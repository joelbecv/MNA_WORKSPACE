# OBJETIVO: Cargar y mostrar una imagen usando OpenCV

import cv2

# Lee la imagen del archivo y la guarda en memoria
img = cv2.imread('./images/input.jpg')
# Muestra la imagen en una ventana y espera a que el usuario presione una tecla
cv2.imshow('Input image', img)
cv2.waitKey()
