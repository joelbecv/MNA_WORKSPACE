# Semana 2 — Controlador Simple con Cámara en Webots
## Navegación Autónoma — Maestría en Inteligencia Artificial

Simulación de un vehículo autónomo en Webots controlado por teclado, con procesamiento de imagen en tiempo real usando OpenCV.

---

## Archivos

| Archivo | Descripción | Estado |
|---|---|---|
| `simple_controller_H3.py` | Detección de carriles con PID autopilot + modo manual | ✅ **Versión actual** |
| `simple_controller_H2.py` | Versión anterior con Hough mejorado | Histórico |
| `simple_controller_H.py` | Primera versión funcional | Histórico |
| `simple_controller_act_2_1.py` | Versión inicial de actividad | Histórico |

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

> ⚠️ **Nota importante sobre la ruta del archivo**
> La ruta al script cambia dependiendo del equipo y de dónde esté guardado el archivo.
> Usa la ruta donde tengas el archivo en **tu** máquina — puede ser en el workspace local,
> en Google Drive o en cualquier otra carpeta.

**Pasos:**
1. Abre Webots y carga `city_2025a.wbt`
2. Espera a que la simulación esté en pausa con el robot listo
3. Ejecuta en terminal los 4 comandos siguientes:

```bash
export WEBOTS_HOME=/Applications/Webots.app/Contents
export DYLD_LIBRARY_PATH=/Applications/Webots.app/Contents/lib/controller
export PYTHONPATH=/Applications/Webots.app/Contents/lib/controller/python
/Applications/Webots.app/Contents/MacOS/webots-controller "RUTA_AL_ARCHIVO/simple_controller_H3.py"
```

### Ejemplos de rutas según equipo

**Si el archivo está en el workspace local (recomendado):**
```bash
/Applications/Webots.app/Contents/MacOS/webots-controller "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/semana_2/SIMULATOR1/simple_controller_H3.py"
```

**Si el archivo está en Google Drive:**
```bash
/Applications/Webots.app/Contents/MacOS/webots-controller "/Users/joelbecerril/Library/CloudStorage/GoogleDrive-baldjoel@gmail.com/Mi unidad/maestria/IA/Tec/Navegación autonoma/Sem_2/simple_controller_H3.py"
```

---

## Dependencias

- Webots R2025a (con módulos `controller` y `vehicle`)
- Python configurado en Webots Preferences → debe tener `numpy` y `opencv-python` instalados
  - Verificar con: `<python-de-webots> -m pip install numpy opencv-python`
- El Python que usa Webots se puede revisar/cambiar en: **Webots → Cmd+, → Python command**
