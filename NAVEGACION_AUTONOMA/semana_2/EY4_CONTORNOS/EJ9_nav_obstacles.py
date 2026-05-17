# OBJETIVO: Simular detección automática de obstáculos en navegación autónoma
# Un vehículo autónomo recibe frames de su cámara y debe identificar qué está en su camino
# Sin intervención humana: el sistema segmenta, detecta contornos y clasifica obstáculos
# Técnica: segmentación por color HSV + análisis de contornos + bounding boxes

import cv2
import numpy as np

def generate_road_scene():
    """
    Genera una escena sintética de vista frontal de cámara vehicular:
    - Fondo gris (asfalto)
    - Líneas blancas de carril
    - Obstáculos de colores (peatón=naranja, cono=rojo, vehículo=azul)
    """
    scene = np.full((480, 640, 3), (80, 80, 80), dtype=np.uint8)  # asfalto gris

    # Líneas de carril blancas
    cv2.rectangle(scene, (190, 0),   (210, 480), (200, 200, 200), -1)
    cv2.rectangle(scene, (430, 0),   (450, 480), (200, 200, 200), -1)

    # Vehículo adelante (azul)
    cv2.rectangle(scene, (250, 150), (420, 280), (180, 80,  20), -1)
    cv2.rectangle(scene, (270, 100), (400, 155), (180, 80,  20), -1)

    # Peatón cruzando (naranja)
    cv2.ellipse(scene,  (130, 290), (22, 28), 0, 0, 360, (20, 120, 220), -1)  # cabeza
    cv2.rectangle(scene,(110, 318), (150, 410), (20, 120, 220), -1)            # cuerpo

    # Cono de tráfico (rojo)
    cone = np.array([[530, 400], [510, 320], [550, 320]], np.int32)
    cv2.fillPoly(scene, [cone], (30, 30, 200))
    cv2.rectangle(scene, (505, 400), (555, 415), (30, 30, 200), -1)

    return scene

def detect_obstacles(frame):
    """
    Detecta obstáculos automáticamente usando segmentación por color en espacio HSV.
    Devuelve la imagen anotada con bounding boxes y etiquetas.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = frame.copy()

    # Cada obstáculo tiene un rango de color diferente en HSV
    # HSV permite separar color (Hue) de iluminación (Value) — más robusto que RGB
    obstacles = [
        {
            'name':  'VEHICULO',
            'color': (180, 80, 20),          # azul oscuro en BGR para dibujar
            'lower': np.array([100, 80, 50]),
            'upper': np.array([130, 255, 255]),
        },
        {
            'name':  'PEATON',
            'color': (0, 165, 255),          # naranja en BGR
            'lower': np.array([5, 100, 100]),
            'upper': np.array([25, 255, 255]),
        },
        {
            'name':  'CONO',
            'color': (0, 0, 200),            # rojo en BGR
            'lower': np.array([0, 150, 100]),
            'upper': np.array([10, 255, 255]),
        },
    ]

    for obs in obstacles:
        # Genera máscara binaria: blanco donde el color está en el rango HSV
        mask = cv2.inRange(hsv, obs['lower'], obs['upper'])

        # Elimina ruido pequeño con operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encuentra contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:          # ignora detecciones muy pequeñas (ruido)
                continue

            # Bounding box alrededor del obstáculo detectado
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), obs['color'], 3)

            # Etiqueta con nombre y área
            label = f"{obs['name']} ({int(area)}px)"
            cv2.putText(result, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, obs['color'], 2)

            # Distancia simulada: objetos más grandes = más cerca
            # En un sistema real se usaría LIDAR o cámara estéreo
            dist_sim = int(10000 / area * 10)
            cv2.putText(result, f'~{dist_sim}m', (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obs['color'], 1)

    # Indicador de carril libre
    lane_free = not any(
        cv2.countNonZero(cv2.inRange(hsv, obs['lower'], obs['upper'])[200:400, 215:430]) > 300
        for obs in obstacles
    )
    status = 'CARRIL LIBRE' if lane_free else 'OBSTACULO EN CARRIL'
    color  = (0, 200, 0)       if lane_free else (0, 0, 255)
    cv2.putText(result, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return result

if __name__ == '__main__':
    scene = generate_road_scene()

    # ── Ventana 1: escena original (lo que ve la cámara) ─────────────────────
    cv2.imshow('1 - Camara vehicular (entrada)', scene)

    # ── Ventana 2: detección automática de obstáculos ─────────────────────────
    detected = detect_obstacles(scene)
    cv2.imshow('2 - Obstaculos detectados (salida sistema)', detected)

    # ── Ventana 3: máscaras HSV de cada obstáculo ─────────────────────────────
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    mask_vehiculo = cv2.inRange(hsv, np.array([100, 80,  50]), np.array([130, 255, 255]))
    mask_peaton   = cv2.inRange(hsv, np.array([5,   100, 100]), np.array([25,  255, 255]))
    mask_cono     = cv2.inRange(hsv, np.array([0,   150, 100]), np.array([10,  255, 255]))

    # Une las 3 máscaras lado a lado para visualizar
    masks_row = np.hstack([mask_vehiculo, mask_peaton, mask_cono])
    cv2.putText(masks_row, 'VEHICULO', (10, 25),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
    cv2.putText(masks_row, 'PEATON',  (660, 25),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
    cv2.putText(masks_row, 'CONO',    (1310, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 128, 2)
    cv2.imshow('3 - Mascaras HSV por tipo de obstaculo', masks_row)

    print("Lo que hace el sistema en cada frame de la cámara:")
    print("  1. Convierte BGR -> HSV para separar color de iluminación")
    print("  2. Aplica máscara por rango de color para cada tipo de obstáculo")
    print("  3. Encuentra contornos en cada máscara")
    print("  4. Dibuja bounding box + etiqueta + distancia simulada")
    print("  5. Verifica si el carril propio tiene obstáculos")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
