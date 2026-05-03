# OBJETIVO: Suavizar contornos ruidosos usando aproximación poligonal (approxPolyDP)
# Factor alto (0.05) = muy simplificado (pocos puntos) | Factor bajo (0.00001) = casi igual al original
# Ventana 1: contorno ORIGINAL en rojo | Ventana 2: contorno SIMPLIFICADO en azul

import cv2
import numpy as np

IMG_INPUT = 'input_shapes.png'

def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    img = cv2.imread(IMG_INPUT)

    if img is None:
        print("ERROR: Corre primero generate_images.py para crear las imágenes")
        exit()

    # Filtra contornos válidos (descarta borde de imagen y ruido)
    img_area = img.shape[0] * img.shape[1]
    contours = [
        c for c in get_all_contours(img)
        if 0.001 < cv2.contourArea(c) / float(img_area) < 0.8
    ]

    # ── Ventana 1: contornos ORIGINALES ───────────────────────────────────────
    # Muestra cuántos puntos tiene cada contorno antes de simplificar
    original_img = np.ones_like(img) * 255
    for c in contours:
        cv2.drawContours(original_img, [c], -1, (0, 0, 200), 2)  # rojo

    total_pts = sum(len(c) for c in contours)
    cv2.putText(original_img, f'Original: {total_pts} puntos totales',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    cv2.imshow('1 - Contornos originales (rojo)', original_img)

    # ── Ventana 2: contornos SIMPLIFICADOS ────────────────────────────────────
    # approxPolyDP reduce los puntos manteniendo la forma general
    # epsilon = tolerancia máxima de distancia entre el contorno original y el simplificado
    factor = 0.02   # prueba con 0.001 (casi igual) o 0.05 (muy simplificado)

    simplified = []
    for c in contours:
        epsilon = factor * cv2.arcLength(c, True)
        simplified.append(cv2.approxPolyDP(c, epsilon, True))

    simplified_img = np.ones_like(img) * 255
    for c in simplified:
        cv2.drawContours(simplified_img, [c], -1, (200, 0, 0), 2)  # azul
        # Dibuja un punto en cada vértice que quedó después de simplificar
        for pt in c:
            cv2.circle(simplified_img, tuple(pt[0]), 4, (0, 150, 0), -1)

    total_pts_simplified = sum(len(c) for c in simplified)
    cv2.putText(simplified_img, f'Simplificado (factor={factor}): {total_pts_simplified} puntos',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    cv2.imshow('2 - Contornos simplificados (azul)', simplified_img)

    print(f"Puntos originales : {total_pts}")
    print(f"Puntos simplificados: {total_pts_simplified}")
    print(f"Reducción: {100 - total_pts_simplified * 100 // total_pts}%")
    print(f"\nCambia 'factor' en el código para ver más o menos simplificación")

    cv2.waitKey()
    cv2.destroyAllWindows()
