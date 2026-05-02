# OBJETIVO: Separar los canales Y, U, V de una imagen YUV usando dos métodos (split y slicing NumPy)

import cv2

# Lee la imagen y la convierte de BGR a YUV (Y=brillo, U y V=color)
img = cv2.imread('./images/input.jpg')
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Alternative 1: separa los 3 canales usando cv2.split (crea copias, más lento)
y, u, v = cv2.split(yuv_img)
cv2.imshow('Y channel', y)
cv2.imshow('U channel', u)
cv2.imshow('V channel', v)
cv2.waitKey()

# Alternative 2: accede a los canales directamente como array NumPy (más rápido)
cv2.imshow('Y channel', yuv_img[:, :, 0])
cv2.imshow('U channel', yuv_img[:, :, 1])
cv2.imshow('V channel', yuv_img[:, :, 2])
cv2.waitKey()
