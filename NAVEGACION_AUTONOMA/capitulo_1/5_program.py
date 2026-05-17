# OBJETIVO: Convertir una imagen al espacio de color YUV (brillo separado del color)

import cv2

# Lee la imagen en formato BGR
img = cv2.imread('./images/input.jpg')
# Convierte de BGR a YUV: Y=luminancia (brillo), U y V=crominancia (color)
# Útil en visión por computadora para separar información de brillo e iluminación
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('YUV image', yuv_img)
cv2.waitKey()
