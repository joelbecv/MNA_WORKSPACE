# NLP Semana 2 — Actividad Evaluada
## Maestría en Inteligencia Artificial Aplicada (MNA)
### Prof. Luis Eduardo Falcón Morales

**Alumno:** Joel Arturo Becerril Balderas  
**Matrícula:** A01797427  
**Notebook:** `MNA_NLP_semana_02_Actividad.ipynb`  
**Dataset:** 1,000 reseñas de restaurantes en inglés — Yelp (UCI Sentiment Labelled Sentences)

---

## Por qué importa esto en el negocio

Antes de que cualquier modelo de IA pueda analizar texto — clasificar opiniones de clientes, detectar quejas, identificar tendencias — el texto debe pasar por un proceso de limpieza y extracción. Esta actividad cubre exactamente ese pipeline:

> **Texto crudo → Limpieza → Tokenización → Vocabulario**

Empresas como Amazon, Uber Eats o cualquier negocio con reseñas en línea aplican este proceso a millones de comentarios diarios para extraer señales accionables: ¿qué productos gustan más?, ¿dónde hay quejas recurrentes?, ¿el sentimiento mejoró este mes?

---

## Dataset

- **Fuente:** Yelp Reviews — UCI Machine Learning Repository
- **Tamaño:** 1,000 comentarios en inglés sobre servicios de comida
- **Formato:** un comentario por línea en archivo `.txt`
- **Uso:** corpus de práctica para aplicar regex y técnicas de limpieza de texto

---

## Contenido del notebook

### Parte 1 — Carga de datos
Lectura del archivo con `readlines()` para obtener una lista de 1,000 strings listos para procesar.

---

### Parte 2 — Expresiones Regulares (15 ejercicios)

| # | Qué busca | Patrón regex | Aplicación de negocio |
|---|---|---|---|
| P1 | Eliminar `\n` al final de cada comentario | `re.sub(r'\n$', '')` | Normalizar datos al ingestar de APIs o archivos |
| P2 | Palabras con 2+ signos de admiración (`!!!`) | `\w+!{2,}` | Detectar comentarios con alta carga emocional |
| P3 | Palabras totalmente en mayúsculas | `[A-Z]{2,}` | Identificar énfasis o gritos en reseñas — señal de enojo o euforia |
| P4 | Comentarios donde TODAS las letras son mayúsculas | `fullmatch([^a-z]*)` | Filtrar comentarios con tono agresivo para moderación |
| P5 | Palabras con vocal acentuada (á é í ó ú) | `[áéíóú]` | Detectar texto en español mezclado con inglés |
| P6 | Cantidades monetarias (`$4.99`) | `\$\d+(?:\.\d+)?` | Extraer menciones de precios para análisis de percepción de valor |
| P7 | Variantes de "love" | `\w*love\w*` | Medir sentimiento positivo — palabra clave de lealtad de cliente |
| P8 | Variantes de "sooo" / "goood" | `so{2,}` / `go{3,}d` | Detectar lenguaje informal — señal de autenticidad en reseñas |
| P9 | Palabras con más de 10 caracteres | `[a-zA-Z]{11,}` | Identificar términos técnicos o especializados en el dominio |
| P10 | Palabras con mayúscula inicial que no son inicio de oración | `(?<=\s)[A-Z]\w*[a-z]` | Detectar nombres propios: restaurantes, platillos, marcas |
| P11 | Palabras compuestas con guion (Go-Kart) | `\w+(?:-\w+)+` | Extraer nombres propios compuestos y términos del negocio |
| P12 | Palabras terminadas en "ing" o "ed" | `\w+ing` / `\w+ed` | Identificar verbos en progreso o pasado — análisis de acciones del cliente |
| P13 | Limpieza completa del corpus | Pipeline regex | Preparación del texto para modelos de ML |
| P14 | Tokenización por palabras | `split()` | Convertir texto a unidades procesables por algoritmos |
| P15 | Eliminación de stopwords | Filtro por lista | Reducir ruido y quedarse solo con palabras con significado |

---

### Parte 3 — Pipeline de limpieza

El pipeline completo que transforma texto crudo en datos listos para un modelo:

```
Texto original
    ↓ Eliminar puntuación y caracteres especiales
    ↓ Convertir a minúsculas
    ↓ Eliminar espacios extra
    ↓ Tokenizar (split por palabras)
    ↓ Eliminar stopwords
Vocabulario limpio
```

**Resultado final:**
- Tokens totales antes de stopwords: ~8,000-10,000
- Tokens únicos (vocabulario): reducción significativa, solo palabras con carga semántica

---

## Conceptos clave

| Concepto | Definición |
|---|---|
| **Regex** | Patrones para buscar, extraer y limpiar texto con una sola línea de código |
| **Tokenización** | Dividir texto en unidades mínimas (palabras) procesables por algoritmos |
| **Stopwords** | Palabras de alta frecuencia sin carga semántica (the, is, and) que se eliminan para reducir ruido |
| **Vocabulario** | Conjunto de tokens únicos que representan el universo de palabras del corpus |
| **Corpus** | Colección de documentos de texto usada para entrenar o analizar modelos de NLP |

---

## Referencia

- Dataset: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
- Fuente original: https://www.yelp.com/
