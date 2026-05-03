# OBJETIVO: Detectar líneas en una imagen usando la Transformada de Hough
# Referencia: https://www.youtube.com/watch?v=7m-RVJ6ABsY
#             https://www.youtube.com/watch?v=gbL3XKOiBvw
# Código base: https://gist.github.com/pknowledge/62ad0d100d6d4df756c0374dee501131

import cv2
import numpy as np

# Lee la imagen — se usa sudoku porque tiene muchas líneas rectas, ideal para Hough
img = cv2.imread('sudoku.jpg')

if img is None:
    print("ERROR: no se encontró sudoku.png — corre el programa desde la carpeta EY3_TRASNFORM_HUGH")
    exit()

# Convierte a escala de grises — necesario para aplicar Canny
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny: detecta bordes usando dos umbrales (50 y 150)
# Los bordes son el input que Hough usa para encontrar líneas
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('Edges (Canny)', edges)

# HoughLines: encuentra líneas en la imagen de bordes
# Parámetros: resolución de rho=1px, resolución de theta=1 grado, umbral=200 votos mínimos
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Dibuja cada línea detectada sobre la imagen original en color rojo
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    # Calcula el punto central de la línea
    x0 = a * rho
    y0 = b * rho
    # Extiende la línea 1000px en cada dirección para que cruce toda la imagen
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Hough Lines', img)

# HoughLinesP: versión probabilística — detecta segmentos con inicio y fin definidos
# minLineLength=50: descarta segmentos más cortos de 50px
# maxLineGap=10: une segmentos separados por menos de 10px
img_p = cv2.imread('sudoku.png')
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

# Dibuja cada segmento detectado en verde — solo donde realmente existe el trazo
for line in linesP:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_p, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Hough Lines P (segmentos)', img_p)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
