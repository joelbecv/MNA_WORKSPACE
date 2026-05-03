# OBJETIVO: Identificar y censurar formas específicas usando el factor de solidez y K-means clustering
# Solidez = área del contorno / área del casco convexo — valor bajo indica forma no convexa (como un boomerang)
# K-means agrupa las formas en 2 clusters y selecciona el de menor solidez para censurar

import cv2
import numpy as np

IMG_INPUT = 'input_shapes.png'

def get_all_contours(img):
    """Extrae todos los contornos de una imagen binarizada"""
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

if __name__ == '__main__':
    img = cv2.imread(IMG_INPUT)

    if img is None:
        print("ERROR: no se encontró la imagen de entrada")
        exit()

    img_orig = np.copy(img)
    input_contours = get_all_contours(img)
    solidity_values = []

    # Calcula el factor de solidez de cada contorno
    # Solidez baja = forma con cavidades (boomerang) | Solidez alta = forma sólida (círculo, cuadrado)
    for contour in input_contours:
        area_contour = cv2.contourArea(contour)
        convex_hull = cv2.convexHull(contour)
        area_hull = cv2.contourArea(convex_hull)
        if area_hull == 0:
            continue
        solidity = float(area_contour) / area_hull
        solidity_values.append(solidity)

    # K-means con K=2: separa las formas en dos grupos (convexas y no convexas)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    solidity_array = np.array(solidity_values).reshape((len(solidity_values), 1)).astype('float32')
    compactness, labels, centers = cv2.kmeans(solidity_array, 2, None, criteria, 10, flags)

    # El cluster con menor centro de solidez contiene las formas no convexas (boomerangs)
    closest_class = np.argmin(centers)
    output_contours = []
    for i in np.where(labels == closest_class)[0]:
        output_contours.append(input_contours[i])

    # Dibuja los contornos identificados
    cv2.drawContours(img, output_contours, -1, (0, 0, 0), 3)
    cv2.imshow('Detected shapes', img)

    # Censura las formas detectadas con un rectángulo negro
    for contour in output_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_orig, [box], 0, (0, 0, 0), -1)

    cv2.imshow('Censored', img_orig)
    cv2.waitKey()
    cv2.destroyAllWindows()
