# OBJETIVO: Redimensionar una imagen usando tres métodos de interpolación diferentes y comparar resultados

import cv2

# Lee la imagen
img = cv2.imread('images/input.jpg')

# Escala la imagen 1.2x con interpolación lineal
# Recomendada para agrandar: rápida pero menor calidad que cúbica
img_scaled = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)

# Escala la imagen 1.2x con interpolación cúbica
# Recomendada para agrandar: más lenta pero mayor calidad que lineal
img_scaled = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)

# Redimensiona a un tamaño fijo (450x400 píxeles) con interpolación por área
# Recomendada para reducir imágenes: toma el valor más representativo de cada zona
img_scaled = cv2.resize(img, (450, 400), interpolation=cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)

cv2.waitKey()
