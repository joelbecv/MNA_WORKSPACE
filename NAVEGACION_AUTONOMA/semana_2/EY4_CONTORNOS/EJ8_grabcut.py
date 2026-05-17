# OBJETIVO: Segmentar un objeto del fondo usando GrabCut
# El usuario dibuja un rectángulo alrededor del objeto y el algoritmo lo recorta automáticamente
# GrabCut aprende los colores del objeto y del fondo usando modelos Gaussianos (GMMRF)
# Referencia: http://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf

import cv2
import numpy as np

# Foto real con fondo distinto al objeto — ideal para GrabCut
IMG_INPUT = '../../capitulo_1/images/input.jpg'

# Variables globales del mouse
drawing = False
x_init, y_init = -1, -1
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
img_orig = None
img = None

def run_grabcut(img_orig, rect_final):
    """Aplica GrabCut en la región seleccionada y muestra el resultado"""
    mask = np.zeros(img_orig.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut con el rectángulo como región de interés inicial
    cv2.grabCut(img_orig, mask, rect_final, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # mask: 0=fondo, 1=primer plano, 2=probable fondo, 3=probable primer plano
    # Se considera primer plano si el valor es 1 o 3
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Aplica la máscara — el fondo queda negro, el objeto permanece
    result = img_orig * mask2[:, :, np.newaxis]
    cv2.imshow('Output — objeto segmentado', result)

def draw_rectangle(event, x, y, flags, _):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt, img_orig, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
            # Invierte los colores dentro del área seleccionada para dar feedback visual
            img[y_init:y, x_init:x] = 255 - img_orig[y_init:y, x_init:x]
            cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt, bottom_right_pt = (x_init, y_init), (x, y)
        # Restaura los píxeles originales dentro del área y dibuja el rectángulo final
        img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
        cv2.rectangle(img, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

        rect_final = (x_init, y_init, x - x_init, y - y_init)

        # Ejecuta GrabCut automáticamente al soltar el mouse
        run_grabcut(img_orig, rect_final)

if __name__ == '__main__':
    img_orig = cv2.imread(IMG_INPUT)

    if img_orig is None:
        print("ERROR: no se encontró la imagen")
        exit()

    img = img_orig.copy()

    print("Instrucciones:")
    print("  Dibuja un rectángulo alrededor del objeto con el mouse")
    print("  GrabCut se aplica automáticamente al soltar")
    print("  Presiona ESC para salir")

    cv2.namedWindow('Input — dibuja un rectangulo')
    cv2.setMouseCallback('Input — dibuja un rectangulo', draw_rectangle)

    while True:
        cv2.imshow('Input — dibuja un rectangulo', img)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()
