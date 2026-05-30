# Proceso: Análisis de Sentimiento con Vectores Embebidos
## NLP — Semanas 4 y 5 | MNA Tec de Monterrey

**Autor:** Joel Arturo Becerril Balderas — A01797427  
**Fecha:** Mayo 2026  
**Repositorio:** https://github.com/joelbecv/MNA_WORKSPACE

---

## 1. Objetivo

Construir y comparar modelos de análisis de sentimiento (positivo/negativo) usando vectores embebidos de HuggingFace sobre 3,000 comentarios de Amazon, Yelp e IMDb. Se contrastan dos enfoques: (I) promedio de embeddings de palabras, y (II) embedding directo de la oración completa.

---

## 2. Pipeline Completo

```
Datos crudos (.txt)
      │
      ▼
[1] Carga y fusión ──────────────── df: 3,000 filas × {text, label, source}
      │
      ▼
[2] Limpieza de texto ───────────── Xclean: minúsculas, sin HTML, solo letras
      │
      ▼
[3] Partición 70/15/15 ─────────── Xtrain (2,100) | Xval (450) | Xtest (450)
      │
      ├──── PARTE I ────────────────────────────────────────────────────────────
      │         │
      │         ▼
      │   [4] Vocabulario (desde train)───── min_freq≥2, min_len≥2
      │         │
      │         ▼
      │   [5] Modelo HuggingFace ──────────── bge-base-en-v1.5 (768d)
      │         │
      │         ▼
      │   [6] Diccionario {palabra → vector}─ guardado en pickle
      │         │
      │         ▼
      │   [7] Promedio de vectores ─────────── trainEmb, valEmb, testEmb
      │         │
      │         ▼
      │   [9] Clasificadores ───────────────── LR + RF → accuracy, F1
      │
      └──── PARTE II ───────────────────────────────────────────────────────────
                │
                ▼
          [10] Embed oraciones completas ─── X_train_emb, X_val_emb, X_test_emb
                │
                ▼
          [10] Clasificadores ─────────────── LR + RF → accuracy, F1
                │
                ▼
          [11] Mejor modelo → Test set ──── Matriz de confusión + report
```

---

## 3. Decisiones Técnicas y Justificación

| Decisión | Valor | Justificación |
|---|---|---|
| Split | 70 / 15 / 15 | Balance entre aprendizaje y evaluación con confianza estadística |
| Semilla | 42 | Reproducibilidad total |
| `min_freq` | 2 | Elimina errores tipográficos y ruido estadístico |
| `min_len` | 2 | Elimina residuos de la limpieza (chars sueltos) |
| Modelo | `bge-base-en-v1.5` | Mejor ratio rendimiento/recursos del grupo; 768 dims; top MTEB |
| `normalize_embeddings` | `True` | Habilita similitud coseno; mejora clasificadores lineales |
| Batch size (vocab encode) | 256 | Optimizado para RAM disponible |
| Batch size (sentence encode) | 64 | Menor por ser secuencias más largas |
| `n_estimators` RF | 200 | Suficiente para convergencia sin overfitting excesivo |
| `max_iter` LR | 1000 | Asegura convergencia en espacio de 768 dimensiones |

---

## 4. Fix Técnico: Archivo IMDb

**Problema:** `pd.read_csv('imdb_labelled.txt', sep='\t')` devuelve 748 filas en lugar de 1,000.

**Causa:** El archivo contiene comillas dobles internas en los textos (ej: `The "acting" was great`). El parser CSV de pandas las interpreta como delimitadores de campo, agrupando múltiples líneas en una sola fila.

**Solución:**
```python
imdb = pd.read_csv('imdb_labelled.txt', sep='\t', header=None,
                   names=['text', 'label'], quoting=csv.QUOTE_NONE)
```
`csv.QUOTE_NONE` desactiva el procesamiento especial de comillas → 1,000 filas correctas.

---

## 5. Modelos HuggingFace Evaluados

| Modelo | Dimensión | Parámetros | Tamaño aprox. | Organización |
|---|---|---|---|---|
| `bge-base-en-v1.5` ✓ | 768 | ~110M | ~438 MB | BAAI |
| `bge-large-en-v1.5` | 1024 | ~335M | ~1.3 GB | BAAI |
| `e5-base-v2` | 768 | ~110M | ~438 MB | Microsoft |

**Seleccionado:** `bge-base-en-v1.5` — optimiza rendimiento vs costo. `bge-large` ofrece mejor accuracy a costa de ~3× más memoria. `e5-base-v2` requiere prefijos `"query: "` / `"passage: "` para máximo rendimiento.

---

## 6. Casos de Uso en Negocio

### 6.1 E-commerce (contexto Amazon)
- Clasificar millones de reseñas automáticamente sin intervención humana
- Alertar a equipos de producto cuando un artículo acumula negativos en <24h (detección de defectos)
- Mostrar reseñas representativas en páginas de producto (positivas verificadas / negativas constructivas)
- Alimentar sistemas de recomendación: si el sentimiento de reseñas de un producto es consistentemente negativo en una característica, recomendarle al usuario uno que no lo tenga

### 6.2 Restaurantes y Servicios (contexto Yelp)
- Dashboard de satisfacción por sucursal en tiempo real
- Detectar quejas recurrentes temáticas (servicio lento, porciones pequeñas, precio alto) mediante clustering de comentarios negativos
- Comparar sentimiento vs competidores en la misma zona geográfica
- Monitorear impacto de cambios de menú o política de servicio

### 6.3 Entretenimiento (contexto IMDb / Streaming)
- Predicción de éxito comercial de contenido basándose en early reviews
- Sistema de recomendación: encontrar películas con comentarios similares a las que el usuario ya valoró positivamente
- Detección de *review bombing* (ataques coordinados de calificaciones negativas)
- Análisis de sentimiento por personaje o aspecto (actuación, guion, cinematografía)

### 6.4 Otros Sectores
| Sector | Aplicación |
|---|---|
| Banca | Monitoreo de redes sociales sobre nuevos productos financieros |
| Salud | Satisfacción de pacientes en portales médicos post-consulta |
| Recursos Humanos | Análisis de encuestas de clima laboral a escala |
| Gobierno | Análisis de sentimiento en comentarios ciudadanos sobre políticas públicas |
| Turismo | Clasificación automática de reseñas de hoteles para sistemas de calidad |

---

## 7. Archivos del Proyecto

| Archivo | Descripción |
|---|---|
| `MNA_NLP_semanas_4y5_Actividad_Embeddings_2026_HF.ipynb` | Notebook solución entregable |
| `semana05Embeddings_explicado.ipynb` | Notebook con explicaciones detalladas por sección |
| `vocab_embeddings_bge_base.pkl` | Diccionario `{palabra: vector 768d}` (caché generado en Q6) |
| `proceso_embeddings_nlp.md` | Este documento |
| `data/sentiment labelled sentences/` | Dataset original de la UCI |

**Ubicación en GitHub:**  
`https://github.com/joelbecv/MNA_WORKSPACE/tree/main/NLP/Tareasem4y5/`

---

## 8. Resultados Esperados

| Modelo | Parte | Train Acc | Val Acc | Gap | Status |
|---|---|---|---|---|---|
| Regresión Logística | I (avg embed) | ~85% | ~80-83% | <3% | ✓ No sobreentrenado |
| Random Forest | I (avg embed) | ~99% | ~77-80% | >3% | ✗ Sobreentrenado |
| Regresión Logística | II (full embed) | ~87% | ~83-87% | <3% | ✓ No sobreentrenado |
| Random Forest | II (full embed) | ~99% | ~80-84% | >3% | ✗ Sobreentrenado |

**Conclusión esperada:** Parte II > Parte I porque el modelo captura contexto completo (negaciones, modificadores). Regresión Logística es más estable; Random Forest requiere ajuste de hiperparámetros para no sobreentrenar.

---

## 9. Reproducibilidad

```bash
# 1. Activar entorno conda
conda activate ml_env

# 2. Abrir notebook en VS Code
# Seleccionar kernel: ml_env

# 3. Ejecutar todas las celdas (Run All)
# Primera ejecución:
#   - Descarga bge-base-en-v1.5 (~438MB desde HuggingFace)
#   - Genera vocab_embeddings_bge_base.pkl (~N × 768 × 4 bytes)
# Ejecuciones posteriores:
#   - Carga pkl directamente, sin re-descargar vectores
```

---

## 10. Limitaciones y Mejoras Posibles

| Limitación | Mejora posible |
|---|---|
| Promedio destruye orden de palabras (Parte I) | Usar LSTM o Transformer sobre secuencia de vectores |
| Random Forest sobreentrenado | Ajustar `max_depth`, `min_samples_leaf` |
| Dataset pequeño (3,000) para fine-tuning | Aumentar con datos de UCI u otros datasets públicos |
| Solo inglés | Usar modelos multilingüe: `bge-m3` o `multilingual-e5-base` |
| Sin manejo de ironía/sarcasmo | Fine-tuning con ejemplos de ironía etiquetada |
