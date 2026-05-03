# OBJETIVO: Detectar formas no convexas (como una pizza con un trozo sacado) usando convexityDefects
# Una forma convexa es aquella donde cualquier línea entre dos puntos internos permanece dentro de la forma
# Una forma no convexa tiene "cavidades" — eso es un defecto de convexidad
# Ventana 1: imagen original | Ventana 2: cavidades marcadas con círculos azules

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

    # ── Ventana 1: ANTES — imagen original con contornos en gris ─────────────
    before = img.copy()
    cv2.drawContours(before, contours, -1, (150, 150, 150), 2)
    cv2.putText(before, 'ANTES: formas originales',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
    cv2.imshow('1 - ANTES', before)

    # ── Ventana 2: DESPUÉS — cavidades marcadas ───────────────────────────────
    after = img.copy()
    defect_count = 0

    for contour in contours:
        # Suaviza el contorno para evitar falsos defectos por ruido en los bordes
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        # Calcula el casco convexo (la "liga elástica" alrededor de la forma)
        hull = cv2.convexHull(contour, returnPoints=False)

        # Detecta los defectos: espacios entre el contorno y el hull
        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            continue

        for i in range(defects.shape[0]):
            start_idx, end_idx, far_idx, depth = defects[i, 0]
            start = tuple(contour[start_idx][0])
            end   = tuple(contour[end_idx][0])
            far   = tuple(contour[far_idx][0])

            # depth viene en unidades de 1/256 — convertir a píxeles
            depth_px = depth / 256.0

            # Solo marca defectos profundos (> 10px) para ignorar imperfecciones pequeñas
            if depth_px > 10:
                # Líneas del hull (la envoltura convexa) en verde
                cv2.line(after, start, end, (0, 200, 0), 2)
                # Círculo azul en el punto más profundo de la cavidad
                cv2.circle(after, far, 7, (200, 0, 0), -1)
                defect_count += 1

        cv2.drawContours(after, [contour], -1, (100, 100, 100), 2)

    cv2.putText(after, f'DESPUES: {defect_count} cavidades detectadas (azul)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)
    cv2.imshow('2 - DESPUES: cavidades detectadas', after)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
