# OBJETIVO: Detectar bordes en una imagen usando tres técnicas: Laplacian, Sobel X y Sobel Y
# Referencia: https://www.youtube.com/watch?v=aDY4aBLFOIg
# Código base: https://gist.github.com/pknowledge/2402470cd53d15639c67522c4d4f4868

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Lee la imagen en escala de grises — la detección de bordes trabaja sobre un solo canal
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)

# Laplacian: detecta bordes en todas las direcciones calculando la segunda derivada de la imagen
# ksize=3 define el tamaño del kernel — detecta cambios abruptos de intensidad
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
# Se usa valor absoluto porque los bordes pueden tener valores negativos
# Se convierte a uint8 para poder mostrar la imagen
lap = np.uint8(np.absolute(lap))

# Sobel X: detecta bordes verticales — calcula la primera derivada en dirección horizontal (1, 0)
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)

# Sobel Y: detecta bordes horizontales — calcula la primera derivada en dirección vertical (0, 1)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

# Convierte a valores absolutos y tipo uint8 para visualización
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

# Combina Sobel X y Sobel Y con OR bit a bit — obtiene bordes en ambas direcciones
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

# Muestra las 5 imágenes en una cuadrícula 2x3 para comparar cada técnica visualmente
titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined']
images = [img, lap, sobelX, sobelY, sobelCombined]

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
