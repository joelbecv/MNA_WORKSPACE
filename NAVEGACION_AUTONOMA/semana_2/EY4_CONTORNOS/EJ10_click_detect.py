# OBJETIVO: Detectar objetos en una foto real haciendo clic sobre ellos
# El usuario hace clic en cualquier objeto y el sistema detecta todo lo que tenga ese color
# Así funciona la percepción básica en robótica antes de usar redes neuronales

import cv2
import numpy as np

# Cambia esta ruta por cualquier foto que tengas
IMG_INPUT = '../../capitulo_1/images/input.jpg'

img = None
img_orig = None

def on_click(event, x, y, flags, _):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Toma el color HSV del píxel donde hizo clic el usuario
    hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[y, x]
    print(f"Clic en ({x},{y}) — HSV: ({h}, {s}, {v})")

    # Define un rango de tolerancia alrededor del color seleccionado
    tolerancia = 25
    lower = np.array([max(h - tolerancia, 0),   max(s - 60, 30),  max(v - 60, 30)])
    upper = np.array([min(h + tolerancia, 179),  min(s + 60, 255), min(v + 60, 255)])

    # Crea máscara binaria: blanco donde el color está en ese rango
    mask = cv2.inRange(hsv, lower, upper)

    # Limpia ruido con operaciones morfológicas
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # Encuentra contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img_orig.copy()
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        x0, y0, w, h0 = cv2.boundingRect(contour)
        cv2.rectangle(result, (x0, y0), (x0 + w, y0 + h0), (0, 255, 0), 3)
        cv2.putText(result, 'objeto', (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        count += 1

    cv2.putText(result, f'{count} objeto(s) detectado(s) — clic para buscar otro color',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    cv2.imshow('Resultado', result)
    cv2.imshow('Mascara de color', mask)

if __name__ == '__main__':
    img_orig = cv2.imread(IMG_INPUT)

    if img_orig is None:
        print("ERROR: no se encontró la imagen")
        exit()

    img = img_orig.copy()

    print("Haz clic en cualquier objeto de la imagen para detectarlo")
    print("Presiona ESC para salir")

    cv2.namedWindow('Foto original — haz clic en un objeto')
    cv2.setMouseCallback('Foto original — haz clic en un objeto', on_click)

    while True:
        cv2.imshow('Foto original — haz clic en un objeto', img_orig)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
