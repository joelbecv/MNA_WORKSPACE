# Log de continuación — MR4010 Proyecto Final — Equipo 25
**Fecha:** 2026-06-23 | **Sesión anterior cierre:** ~2026-06-18  
**Autor:** Joel Arturo Becerril Balderas — A01797427

---

## Estado del proyecto al momento de este log

### Lo que está LISTO

| Componente | Archivo | Estado |
|---|---|---|
| Dataset limpio | `data/dataset_clean.csv` | ✅ 10,883 filas |
| Dataset fuente | `data/dataset_merged.csv` | ✅ 19,500 filas |
| Imágenes | `data/images/` | ✅ 19,500 imágenes |
| Script de limpieza | `code/clean_dataset.py` | ✅ Funcional |
| Notebook de entrenamiento | `code/train_cil.ipynb` | ✅ Reescrito en Keras |
| Controlador autónomo | `controllers/autonomous_cil/autonomous_cil.py` | ✅ Keras puro, sin PID |
| Controlador recolección | `controllers/collect_cil_data/collect_cil_data.py` | ✅ OK |

### Lo que está PENDIENTE

| Tarea | Prioridad | Notas |
|---|---|---|
| Entrenar modelo Keras en Colab | 🔴 CRÍTICO | Ver instrucciones abajo |
| Probar `autonomous_cil.py` en World 2 | 🔴 CRÍTICO | Necesita .h5 entrenado |
| Grabar video 3 recorridos | 🔴 Entrega | Con tráfico SUMO activo |
| Recorridos adicionales | 🟡 Mejora | Ver mensaje a compañeros |
| Redactar informe / Notion | 🟡 Entrega | Pendiente |

---

## Próximos pasos concretos

### 1. Entrenar el modelo en Google Colab

```
1. Subir a Google Drive (o montar):
   - code/train_cil.ipynb
   - data/dataset_clean.csv
   - data/images/ (ZIP de ~2 GB)

2. En Colab, ejecutar todas las celdas en orden:
   - Celda 1: ajustar ruta CSV_PATH y IMG_DIR
   - Celda 4: CILSequence (augmentations, batch 120)
   - Celda 5: build_cil_model() — arquitectura Codevilla 8-conv
   - Celda 6: entrenamiento 40 epochs, ModelCheckpoint guarda .h5
   - Celda 7: ver curvas de entrenamiento
   - Celda 8: verificar dummy inference

3. Descargar: models/cil_model_equipo25.h5
```

**Checklist de validación del modelo entrenado:**
- [ ] val_mse < 0.05 al final del entrenamiento
- [ ] Celda 8 imprime predicción con imagen dummy sin error
- [ ] Archivo .h5 pesa entre 30–80 MB

### 2. Probar en Webots (World 2)

```
1. Copiar .h5 a:  models/cil_model_equipo25.h5

2. Abrir Webots → MR4010_proyecto_final_2026.wbt
   (confirmar que World 2 tiene tráfico SUMO activo)

3. Presionar Play. El controlador carga el modelo automáticamente.
   Verificar en consola:  "[CIL] Modelo Keras cargado"

4. En ventana 3D del simulador:
   s = CONTINUE (seguir carretera)
   w = RECTO    (cruzar recto)
   a = IZQUIERDA
   d = DERECHA

5. Smoke test:  carretera recta con s → vehículo debe mantenerse en carril
```

### 3. Grabación del video de entrega

Tres escenas mínimas (rúbrica lo indica):
1. Ruta recta / carretera + tráfico → CMD_CONTINUE durante ~60 s
2. Intersección con giro izquierda → CMD_LEFT  
3. Intersección con giro derecha  → CMD_RIGHT

Requisito: tráfico SUMO visible, peatón presente para mostrar frenado automático.

---

## Arquitectura del modelo (resumen rápido)

```
Imagen (88×200×3) → 8 Conv (VALID, BN, Dropout, ReLU)
→ Flatten → Dense(512) × 2
Speed (norm/30) → Dense(128) × 2
Concat → Dense(512)
→ 4 ramas: Dense(256)→Dropout→Dense(256)→Dense(1, tanh)
Selección: one-hot(cmd) × branch_outs → reduce_sum → steering
```

Pérdida: MSE ponderada (5× si |steer_norm| > 0.2)  
Dataset: 10,883 filas — CONT 58%, RECTO 5%, IZQ 19%, DER 18%

---

## Dataset — situación actual

| Archivo | Filas | Descripción |
|---|---|---|
| `dataset_merged.csv` | 19,500 | Joel + Alberto fusionados |
| `dataset_clean.csv` | 10,883 | Filtrado de ruido IZQ/DER, sub-muestreo CONTINUE |
| `dataset_balanced.csv` | 13,148 | Versión vieja — NO usar |

**Imágenes:**
- `img_000000` – `img_009186`: Joel (JPEG)
- `img_009187` – `img_019499`: Alberto (PNG convertido a JPEG)

---

## Comandos útiles

```bash
# Verificar dataset limpio
cd code && python clean_dataset.py

# Compilar controlador (verificar sintaxis)
python -m py_compile controllers/autonomous_cil/autonomous_cil.py && echo OK

# Ver distribución del dataset
python -c "
import csv; from collections import Counter
r=list(csv.DictReader(open('data/dataset_clean.csv')))
c=Counter(int(x['nav_command']) for x in r)
labels={0:'CONT',1:'RECT',2:'IZQ',3:'DER'}
[print(f'{labels[k]}: {v:,} ({v/len(r)*100:.1f}%)') for k,v in sorted(c.items())]
"
```
