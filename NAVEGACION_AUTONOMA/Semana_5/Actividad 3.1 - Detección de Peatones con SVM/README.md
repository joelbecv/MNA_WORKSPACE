# Actividad 3.1 — Detección de Peatones con SVM + LiDAR
## Navegación Autónoma — Maestría en Inteligencia Artificial

Controlador de vehículo autónomo con seguimiento de carril PID, detección de peatones por visión (HOG + SVM) y detección de obstáculos por LiDAR (Sick LMS 291).

---

## Archivos

| Archivo | Descripción |
|---|---|
| `pedestrian_svm_training.ipynb` | Notebook de entrenamiento — genera `pedestrian_svm.joblib` |
| `pedestrian_svm.joblib` | Modelo entrenado (se crea al correr el notebook) |
| `SDC_webots 3/worlds/city_2025a_activity_3_1.wbt` | Mundo de Webots con peatones y obstáculos |
| `SDC_webots 3/controllers/simple_controller_pedestrian.py` | Controlador integrado |
| `3.4_SVM_pedestrianArturo.ipynb` | Notebook de referencia (Arturo) |
| `human detection dataset/` | Dataset local: `1/` 1828 personas, `0/` 1274 no personas — **no se sube a git** |

---

## Flujo de trabajo

### Paso 1 — Entrenar el modelo SVM

1. Abre `pedestrian_svm_training.ipynb` en Jupyter
2. Verifica que el kernel de Python tenga instalados: `scikit-learn`, `scikit-image`, `opencv-python`, `joblib`
3. Ejecuta todas las celdas — el modelo se guarda como `pedestrian_svm.joblib` en la misma carpeta

```bash
# Si faltan paquetes:
/Users/joelbecerril/miniconda3/bin/python3 -m pip install scikit-learn scikit-image opencv-python joblib
```

### Paso 2 — Correr el controlador en Webots

1. Abre Webots y carga `SDC_webots 3/worlds/city_2025a_activity_3_1.wbt`
2. Espera a que la simulación esté lista (robot en pausa)
3. Ejecuta en terminal:

```bash
export WEBOTS_HOME=/Applications/Webots.app/Contents
export DYLD_LIBRARY_PATH=/Applications/Webots.app/Contents/lib/controller
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
/Applications/Webots.app/Contents/MacOS/webots-controller \
  "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/Semana_5/Actividad 3.1 - Detección de Peatones con SVM/SDC_webots 3/controllers/simple_controller_pedestrian.py"
```

---

## Comportamiento del controlador

| Situación | Acción |
|---|---|
| Carril libre | PID lane following a 30 km/h |
| Peatón detectado (SVM) | Frenazo de emergencia |
| Barril/obstáculo (LiDAR) | Frenazo de emergencia + intermitentes |

**Display del robot:**
- Imagen de cámara en escala de grises con líneas Hough superpuestas
- HUD con velocidad, ángulo y estado actual (`PID` / `PEATON` / `BARRIL`)

---

## Arquitectura técnica

```
Cámara 256×128
  ↓ resize → display (200×150)
  ↓ HSV amarillo + Canny + ROI + HoughLinesP
  ↓ PID → ángulo de dirección

  ↓ ventana deslizante 64×128 (cada 3 frames)
  ↓ HOG (924 features) → StandardScaler → SVM (RBF, GridSearchCV)
  ↓ peatón confirmado → frenazo

LiDAR Sick LMS 291
  ↓ cono frontal ±25°
  ↓ distancia < 20 m → barril → frenazo + intermitentes
```

### Modelo SVM
- **Accuracy:** 84% (test set)
- **Pipeline:** StandardScaler → SVC (parámetros optimizados con GridSearchCV)
- **Dataset:** 3,102 imágenes — Kaggle human detection dataset + INRIA Person Dataset
- `class_weight = {0:1, 1:3}` — penaliza más los falsos negativos en peatones

### Parámetros HOG
- `target_size = (64, 128)` px
- `orientations = 11`
- `pixels_per_cell = (16, 16)`
- `cells_per_block = (2, 2)`
- Vector de características: **924 valores**

---

## Dependencias

Python en `/Users/joelbecerril/miniconda3/bin/python3` (Python 3.13) con:
- `numpy`, `opencv-python`, `scikit-learn`, `scikit-image`, `joblib`

Webots R2025a con módulos `controller` y `vehicle`.
