# OBJETIVO: Comparar 5 técnicas de suavizado (blurring) en OpenCV para reducir ruido en imágenes
# Referencia: https://www.youtube.com/watch?v=u3poUhCxx4k

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Lee la imagen y convierte de BGR a RGB para mostrarla correctamente con matplotlib
img = cv2.imread('water.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2D Convolution: aplica un kernel personalizado de promedio 5x5
# Cada píxel se reemplaza por el promedio de sus 25 vecinos — suavizado básico
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)

# Blur simple: promedio de píxeles en una ventana 5x5 — equivalente al kernel anterior pero más rápido
blur = cv2.blur(img, (5, 5))

# Gaussian Blur: promedio ponderado — más peso al centro, menos en los bordes
# Más natural visualmente que el blur simple — el más usado como preprocesamiento
gblur = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur: reemplaza cada píxel con la mediana de sus vecinos
# Ideal para eliminar ruido tipo "sal y pimienta" sin afectar bordes
median = cv2.medianBlur(img, 5)

# Bilateral Filter: suaviza preservando bordes — considera tanto distancia espacial como similitud de color
# El más lento pero el que mejor conserva los detalles importantes
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

# Muestra las 6 imágenes en una cuadrícula 2x3 para comparar visualmente cada técnica
titles = ['image', '2D Convolution', 'blur', 'GaussianBlur', 'median', 'bilateralFilter']
images = [img, dst, blur, gblur, median, bilateralFilter]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
