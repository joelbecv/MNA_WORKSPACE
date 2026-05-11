# Semana 2 — Controlador Simple con Cámara en Webots
## Navegación Autónoma — Maestría en Inteligencia Artificial

Simulación de un vehículo autónomo en Webots controlado por teclado, con procesamiento de imagen en tiempo real usando OpenCV.

---

## Archivo

| Archivo | Descripción |
|---|---|
| `simple_controller_act_2_1.py` | Controlador del vehículo con cámara, teclado y procesamiento de imagen |

---

## Qué hace el controlador

- Obtiene imágenes en tiempo real desde la cámara del vehículo
- Convierte la imagen a escala de grises usando OpenCV y la muestra en el display del robot
- Permite controlar el vehículo con el teclado:
  - `↑` — Aumentar velocidad
  - `↓` — Reducir velocidad
  - `→` — Girar a la derecha
  - `←` — Girar a la izquierda
  - `A` — Capturar imagen y guardarla con timestamp

---

## Cómo correrlo en Mac

Requiere Webots instalado en `/Applications/Webots.app`. Abre la simulación en Webots primero, luego ejecuta en terminal:

```bash
export WEBOTS_HOME=/Applications/Webots.app/Contents
export DYLD_LIBRARY_PATH=/Applications/Webots.app/Contents/lib/controller
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
/Applications/Webots.app/Contents/MacOS/webots-controller "/Users/joelbecerril/Library/CloudStorage/GoogleDrive-baldjoel@gmail.com/Mi unidad/maestria/IA/Tec/Navegación autonoma/Sem_2/simple_controller_act_2_1.py"
```

---

## Dependencias

- Webots (con módulos `controller` y `vehicle`)
- Python 3.11 (entorno `ml_env` de conda)
- OpenCV (`cv2`)
- NumPy
