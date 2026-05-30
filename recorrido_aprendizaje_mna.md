# Recorrido de Aprendizaje MNA — Joel Arturo Becerril Balderas
> **Matrícula:** A01797427 | **Repo:** https://github.com/joelbecv/MNA_WORKSPACE

---

## El patrón universal que conecta todo

En **todos** los cursos haces exactamente lo mismo:

```
Datos crudos → [Transformar a números con significado] → Clasificador/Regresor → Predicción
```

El clasificador (LR, RF, SVM, red neuronal) es siempre el mismo tipo.
Lo que cambia entre materias es **cómo representas tu tipo de dato como números**.

---

## Mapa completo por materia

### 1. 📊 Ciencia de Datos
**Dato:** Tabular (salario, edad, cantidad comprada)
**Transform:** Normalización, one-hot encoding, PCA
**Modelo:** LinearRegression, LogisticRegression, SVC
**Reto:** EDA, limpieza, selección de features
**¿Necesita HuggingFace?** ❌ — los datos ya son números con significado natural

Notebooks clave:
- [Regresión Lineal](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/CIENCIA%20DE%20DATOS_WS/Actividad_Semana8/Actividad8RLinealEQUIPO34.ipynb)
- [Regresión Logística](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/CIENCIA%20DE%20DATOS_WS/Actividad9/Actividad9RLogEquipo34.ipynb)
- [PCA](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/CIENCIA%20DE%20DATOS_WS/Actividad7/Actividad7PCAEQUIPO34.ipynb)

---

### 2. ⏱️ IAYML — Series de Tiempo
**Dato:** Números con dependencia temporal
**Transform:** Lags, diferenciación, descomposición estacional
**Modelo:** SARIMA, Prophet, LSTM
**Reto:** Capturar que t+1 depende de t-1, t-2...
**¿Necesita HuggingFace?** ❌ — los datos ya son números, el reto es el tiempo

Notebooks clave:
- [SARIMA + Prophet + LSTM](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/IAYML_WS/SEMANA8aCT/notebooks/MNA_IAyAA_Series_de_Tiempo_SARIMA_Prophet_LSTM_2024.ipynb)
- [Actividad Pronósticos](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/IAYML_WS/SEMANA8TAREA/notebooks/MNA_IAyAA_Actividad_Pronosticos_Series_de_Tiempo_2025.ipynb)

---

### 3. 🤖 Navegación Autónoma
**Dato:** Imágenes (píxeles 0-255) + datos de sensores
**Transform:** Edge detection, HOG features, transformada de Hough
**Modelo:** SVM (sobre HOG features), CNN (end-to-end)
**Reto:** Extraer características espaciales relevantes
**¿Necesita HuggingFace?** ❌ — los píxeles ya son números; la estructura es espacial

Notebooks clave:
- [SVM para peatones](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/Semana_5/Actividad%203.1%20-%20Detecci%C3%B3n%20de%20Peatones%20con%20SVM/pedestrian_svm_training.ipynb)
- [SVM ejercicios](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NAVEGACION_AUTONOMA/semana_3_5MACHINELEARNING%20EXERCI/03_Machine_Learning_rec_apoyo_sem4/3.4_SVM_a.ipynb)

---

### 4. 🧠 Advanced ML (Deep Learning)
**Dato:** Imágenes, texto, secuencias
**Transform:** El modelo aprende su propia representación (CNN aprende filtros, Transformer aprende atención)
**Modelo:** CNN, ResNet, Transformer
**Reto:** Arquitectura, regularización, transfer learning
**¿Necesita HuggingFace?** Parcialmente — el curso Nvidia de NLP (06_nlp.ipynb) usa embeddings básicos

Notebooks clave:
- [CNN MNIST](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/ADVANCEDMLMETHODS/Curso%20Nvidia/01_mnist%20(run).ipynb)
- [CNN ASL](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/ADVANCEDMLMETHODS/Curso%20Nvidia/03_asl_cnn%20(RUN).ipynb)
- [NLP con embeddings](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/ADVANCEDMLMETHODS/Curso%20Nvidia/06_nlp.ipynb)
- [Transformer actividad](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/ADVANCEDMLMETHODS/activity4/A4_DL_TC5033_Transformer-2.ipynb)
- [Entregable final](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/ADVANCEDMLMETHODS/activity2b/A2b_Final_Entregable.ipynb)

---

### 5. 📝 NLP — Sem 1-3 (TF-IDF)
**Dato:** Texto (reseñas Amazon/Yelp/IMDb)
**Transform:** TF-IDF → vector sparse (1 si la palabra está, 0 si no)
**Modelo:** LogisticRegression, RandomForest, SVM — **los mismos de Ciencia de Datos**
**Reto:** Texto no tiene orden numérico natural. TF-IDF cuenta palabras, sin semántica
**¿Necesita HuggingFace?** ❌ — TF-IDF es suficiente como baseline

Notebooks clave:
- [Actividad 2 - Análisis de Sentimiento TF-IDF](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NLP/tareassem3y4_actividad2/MNA21_NLP_Actividad2_Analisis_de_Sentimiento_A01797427.ipynb)

---

### 6. 📝 NLP — Sem 4-5 (HuggingFace Embeddings) ← AQUÍ ESTÁS
**Dato:** Texto (mismas reseñas)
**Transform:** HuggingFace bge-base-en-v1.5 → vector denso 768d con significado semántico
**Modelo:** LogisticRegression, RandomForest — **exactamente los mismos de Ciencia de Datos**
**Reto:** Texto necesita semántica. "great" y "excellent" son palabras distintas para TF-IDF, pero vectores cercanos para HuggingFace

| Parte | Cómo convierte texto a 768 números |
|---|---|
| Parte I | Vectoriza cada palabra del vocabulario → promedia |
| Parte II | Vectoriza la oración completa → un vector con contexto |

**¿Necesita HuggingFace?** ✅ — para capturar significado real, no solo frecuencias

Notebooks clave:
- [Actividad sem4y5 — Solución](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NLP/Tareasem4y5/MNA_NLP_semanas_4y5_Actividad_Embeddings_2026_HF.ipynb)
- [Actividad sem4y5 — Explicado](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NLP/Tareasem4y5/semana05Embeddings_explicado.ipynb)
- [Proceso documentado](https://github.com/joelbecv/MNA_WORKSPACE/blob/main/NLP/Tareasem4y5/proceso_embeddings_nlp.md)

---

## Por qué nunca necesitaste HuggingFace antes

| Tipo de dato | ¿Los números tienen significado natural? | Solución |
|---|---|---|
| Tabular (salario=50,000) | ✅ Sí — 50k > 30k tiene sentido | Usar directo |
| Imagen (pixel=255) | ✅ Sí — 255 es más brillante que 0 | Usar directo / CNN |
| Serie de tiempo | ✅ Sí — t+1 es después de t | Usar directo / LSTM |
| Texto ("great") | ❌ No — ¿great=47? ¿excellent=103? No dice nada | Necesitas TF-IDF o embeddings |

Con texto, asignar números arbitrarios a palabras le dice al modelo mentiras (que "bad"=3 está más cerca de "good"=2 que de "great"=1). TF-IDF resuelve el problema parcialmente. HuggingFace lo resuelve bien.

---

## El arco de lo que estás aprendiendo

```
Nivel 1: "Los datos son números"          → Ciencia de Datos
Nivel 2: "Los números tienen tiempo"      → IAYML
Nivel 3: "Los números tienen espacio"     → Navegación Autónoma
Nivel 4: "El modelo aprende su propia     → Advanced ML (CNNs, Transformers)
          representación"
Nivel 5: "El texto necesita semántica"    → NLP TF-IDF → Embeddings
Nivel 6 (próximo): "Adapta el modelo      → Fine-tuning en tu dominio
          a tu dominio específico"           (ej: reseñas Rotoplas en español)
```

**La pregunta que toda la MNA te enseña a responder:**
> ¿Cómo le doy a un algoritmo la mejor representación posible de cualquier tipo de dato para resolver un problema de negocio?

---

## Aplicación a Rotoplas — Market Intelligence

| Caso de uso | Tipo de dato | Técnica aplicable |
|---|---|---|
| Clasificar reseñas Mercado Libre de tinacos | Texto | NLP Embeddings (este ejercicio) |
| Pronosticar ventas por región | Series de tiempo | IAYML (SARIMA/LSTM) |
| Detectar defectos visuales en productos | Imágenes | Nav Autónoma (CNN) |
| Segmentar distribuidores por comportamiento | Tabular | Ciencia de Datos (clustering/LR) |
| Monitorear sentimiento en noticias hídricas | Texto | NLP Embeddings |
| Clasificar reportes de campo de vendedores | Texto | NLP Embeddings |

---

*Generado: Mayo 2026 | Claude Code — MNA Tec de Monterrey*
