# OBJETIVO: Generar las imágenes de prueba necesarias para los ejercicios EJ4-EJ8
# Crea ref_shape.png (boomerang solo) e input_shapes.png (varias formas)
# Corre este script UNA VEZ antes de ejecutar los demás ejercicios

import cv2
import numpy as np

def create_boomerang(center, size):
    """Crea puntos de un boomerang (forma no convexa con cavidad)"""
    cx, cy = center
    s = size
    pts = np.array([
        [cx,      cy - s],
        [cx + s,  cy + s//2],
        [cx + s//3, cy],
        [cx - s//3, cy],
        [cx - s,  cy + s//2],
    ], dtype=np.int32)
    return pts

# ── ref_shape.png ─────────────────────────────────────────────────────────────
# Fondo blanco con un único boomerang negro — sirve como referencia para EJ4/EJ5
ref = np.ones((300, 300, 3), dtype=np.uint8) * 255

boomerang_ref = create_boomerang((150, 150), 80)
cv2.fillPoly(ref, [boomerang_ref], (0, 0, 0))

cv2.imwrite('ref_shape.png', ref)
print("ref_shape.png creado")

# ── input_shapes.png ───────────────────────────────────────────────────────────
# Fondo blanco con varias formas: círculo, cuadrado, triángulo y dos boomerangs
img = np.ones((500, 700, 3), dtype=np.uint8) * 255

# Círculo — forma convexa, solidez ≈ 1
cv2.circle(img, (90, 90), 60, (0, 0, 0), -1)

# Cuadrado — forma convexa, solidez = 1
cv2.rectangle(img, (200, 40), (340, 160), (0, 0, 0), -1)

# Triángulo — forma convexa
tri = np.array([[450, 160], [390, 40], [510, 40]], dtype=np.int32)
cv2.fillPoly(img, [tri], (0, 0, 0))

# Boomerang 1 — forma NO convexa, solidez baja
bm1 = create_boomerang((130, 320), 70)
cv2.fillPoly(img, [bm1], (0, 0, 0))

# Boomerang 2 — forma NO convexa, solidez baja (girado)
bm2 = np.array([
    [430, 260],
    [580, 300],
    [500, 330],
    [500, 370],
    [580, 400],
    [430, 440],
    [470, 350],
], dtype=np.int32)
cv2.fillPoly(img, [bm2], (0, 0, 0))

# Estrella — forma NO convexa
cx, cy, r1, r2 = 580, 130, 70, 30
star_pts = []
for i in range(10):
    angle = np.pi / 2 + i * 2 * np.pi / 10
    r = r1 if i % 2 == 0 else r2
    star_pts.append([int(cx + r * np.cos(angle)), int(cy - r * np.sin(angle))])
star = np.array(star_pts, dtype=np.int32)
cv2.fillPoly(img, [star], (0, 0, 0))

cv2.imwrite('input_shapes.png', img)
print("input_shapes.png creado")

print("\nListo. Ahora puedes correr EJ4, EJ5, EJ6, EJ7 y EJ8.")
