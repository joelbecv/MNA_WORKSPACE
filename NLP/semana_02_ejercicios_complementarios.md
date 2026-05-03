# NLP Semana 2 — Ejercicios Complementarios
## Maestría en Inteligencia Artificial Aplicada (MNA)
### Prof. Luis Eduardo Falcón Morales

Notebook: `MNA_NLP_semana_02_ejercicios_complementarios.ipynb`

> Ejercicios de práctica (no entregables). Cubren las herramientas base para manipular texto en Python antes de aplicar modelos de NLP.

---

## Por qué importa esto en el negocio

Antes de aplicar cualquier modelo de lenguaje, los datos de texto llegan "sucios": con espacios extra, puntuación irregular, contracciones, mayúsculas inconsistentes, o mezclados con caracteres especiales. Estas técnicas son el **paso de limpieza y preparación** que determina la calidad de todo lo que viene después — análisis de sentimiento, clasificación de documentos, chatbots, búsqueda semántica. Un modelo entrenado con texto mal procesado produce resultados poco confiables, independientemente de su complejidad.

---

## Contenido

### 1. Cadena de caracteres (String)

El tipo de dato base para NLP en Python.

| Operación | Ejemplo | Resultado |
|---|---|---|
| Indexar un caracter | `doc[3]` | Caracter en posición 3 |
| Slicing | `doc[:9]` | Primeros 9 caracteres |
| Índice negativo | `doc[-1]` | Último caracter |
| Longitud | `len(doc)` | Total de caracteres (incluye espacios) |
| Minúsculas | `doc.lower()` | Todo en minúsculas |
| Mayúsculas | `doc.upper()` | Todo en mayúsculas |
| Concatenar | `x + ' ' + y` | Une dos strings |

Ejemplo con frase de *Pedro Páramo* de Juan Rulfo para ilustrar indexación.

---

### 2. split() y strip()

#### split() — segmentar texto en tokens

```python
doc.split()           # separa por espacios en blanco y \n
doc.split('\n')       # separa solo por saltos de línea
doc.split(' ')        # separa solo por espacios
doc.split('el')       # separa por cualquier string
doc.split(' ', 4)     # máximo 4 separaciones
doc.upper().split()   # combinación de métodos
```

#### strip() — limpiar bordes de un string

```python
txt.strip(' ')         # elimina espacios al inicio y al final
'!!hola!!!'.strip('¡!')  # elimina caracteres específicos de los bordes
```

> `strip()` solo actúa en los extremos, no en el interior del string.

#### splitlines() — separar por saltos de línea

```python
txt.splitlines()   # alternativa a split('\n'), más semántico
```

---

### 3. replace()

Sustituir un substring por otro — útil para normalizar texto.

```python
doc.replace('a', 'A').split()   # reemplaza y luego tokeniza
```

Referencia: [w3schools replace](https://www.w3schools.com/python/ref_string_replace.asp)

---

### 4. join()

Une una lista de strings en un solo string con un separador.

```python
'-'.join('Hola')              # → 'H-o-l-a'
' '.join(['Mucha', 'luz'])    # → 'Mucha luz'
```

Operación inversa de `split()` — se usa para reconstruir texto después de procesarlo.

---

### 5. find() y rfind()

Buscar la posición de un substring.

```python
op.find('mucha')           # primera aparición (case-sensitive)
op.lower().find('mucha')   # normalizar antes de buscar
op.lower().find('mucha', 1)  # buscar desde el índice 1
op.lower().rfind('mucha')  # búsqueda en orden inverso
```

> Retorna `-1` si no encuentra el substring.

---

### 6. Expresiones Regulares (re)

Módulo `re` para patrones más flexibles de búsqueda y segmentación.

#### re.split() — separar con patrones

```python
re.split(r' ', X)         # por espacio
re.split(r'[\n]', X)      # por salto de línea
re.split(r'[ \n]', X)     # espacio O salto de línea
re.split(r'\W+', X)       # todo lo que NO sea letra o número
```

#### re.findall() — extraer tokens

```python
re.findall(r'\w+', X)                   # solo palabras
re.findall(r"\w+|\S\w*", Y)             # palabras + no-espacios
re.findall(r"\w+(?:['-]\w+)*", Y)       # contracciones y nombres compuestos (Lao-Tzu, don't)
```

#### Grupo no-capturante (?:)

```python
re.findall(r"\w+(ing|ed)", "cat playing red")    # → ['ing']  (solo captura el grupo)
re.findall(r"\w+(?:ing|ed)", "cat playing red")  # → ['playing'] (captura todo el match)
```

`?:` cancela la extracción del grupo y devuelve el match completo. Fundamental para tokenizar contracciones en inglés sin romperlas.

---

## Métodos clave — resumen rápido

| Método | Tipo entrada | Tipo salida | Uso principal |
|---|---|---|---|
| `split()` | str | list | Tokenizar |
| `strip()` | str | str | Limpiar bordes |
| `replace()` | str | str | Normalizar |
| `join()` | list | str | Reconstruir texto |
| `find()` | str | int (índice) | Localizar substring |
| `re.split()` | str | list | Tokenizar con patrones |
| `re.findall()` | str | list | Extraer con patrones |

---

## Referencia

- Notebook en GitHub: `NLP/MNA_NLP_semana_02_ejercicios_complementarios.ipynb`
- Docs `split()`: https://python-reference.readthedocs.io/en/latest/docs/str/split.html
- Docs `strip()`: https://python-reference.readthedocs.io/en/latest/docs/str/strip.html
- Docs `replace()`: https://www.w3schools.com/python/ref_string_replace.asp
- Docs `find()` (pandas): https://pandas.pydata.org/docs/reference/api/pandas.Series.str.find.html
