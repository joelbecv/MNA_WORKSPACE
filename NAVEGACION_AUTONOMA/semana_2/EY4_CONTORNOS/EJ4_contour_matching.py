# OBJETIVO: Detectar contornos en una imagen y encontrar el que más se parece a una forma de referencia
# Usa Hu Moments para comparar formas independientemente de tamaño, orientación y rotación
# Ventana 1: imagen de referencia | Ventana 2: todos los contornos | Ventana 3: mejor match en rojo

import cv2
import numpy as np

IMG_REF = 'ref_shape.png'
IMG_INPUT = 'input_shapes.png'

def get_all_contours(img):
    """Extrae todos los contornos de una imagen binarizada"""
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_ref_contour(img):
    """Extrae el contorno principal de la imagen de referencia ignorando el borde de la imagen"""
    contours = get_all_contours(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        img_area = img.shape[0] * img.shape[1]
        # Solo toma contornos que ocupen entre 5% y 80% de la imagen
        if 0.05 < area / float(img_area) < 0.8:
            return contour

if __name__ == '__main__':
    img1 = cv2.imread(IMG_REF)
    img2 = cv2.imread(IMG_INPUT)

    if img1 is None or img2 is None:
        print("ERROR: Corre primero generate_images.py para crear las imágenes")
        exit()

    # ── Ventana 1: imagen de referencia ──────────────────────────────────────
    # Muestra la forma que se usará como "molde" para buscar en la otra imagen
    ref_contour = get_ref_contour(img1)
    ref_display = img1.copy()
    cv2.drawContours(ref_display, [ref_contour], -1, (0, 200, 0), 3)
    cv2.imshow('1 - Referencia (forma a buscar)', ref_display)

    # ── Ventana 2: todos los contornos detectados ─────────────────────────────
    # Filtra contornos: descarta el borde de la imagen (muy grande) y ruido (muy pequeño)
    img_area = img2.shape[0] * img2.shape[1]
    input_contours = [
        c for c in get_all_contours(img2)
        if 0.001 < cv2.contourArea(c) / float(img_area) < 0.8
    ]
    all_contours_img = img2.copy()
    cv2.drawContours(all_contours_img, input_contours, -1, color=(255, 100, 0), thickness=3)
    cv2.imshow('2 - Todos los contornos detectados (azul)', all_contours_img)

    # ── Comparación con matchShapes ──────────────────────────────────────────
    # matchShapes usa Hu Moments — valor cercano a 0 = muy similar a la referencia
    closest_contour = None
    min_dist = None

    for i, contour in enumerate(input_contours):
        ret = cv2.matchShapes(ref_contour, contour, 3, 0.0)
        print(f"Contorno {i} — distancia Hu Moments: {ret:.4f}")
        if min_dist is None or ret < min_dist:
            min_dist = ret
            closest_contour = contour

    print(f"\nMejor match — distancia: {min_dist:.4f} (más cerca de 0 = más similar)")

    # ── Ventana 3: mejor match resaltado ─────────────────────────────────────
    # Dibuja en ROJO solo el contorno que más se parece a la referencia
    result_img = img2.copy()
    cv2.drawContours(result_img, input_contours, -1, color=(200, 200, 200), thickness=2)
    cv2.drawContours(result_img, [closest_contour], 0, color=(0, 0, 255), thickness=4)

    # Escribe una etiqueta sobre el contorno encontrado
    x, y, w, h = cv2.boundingRect(closest_contour)
    cv2.putText(result_img, 'MATCH', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('3 - Mejor match (rojo)', result_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
