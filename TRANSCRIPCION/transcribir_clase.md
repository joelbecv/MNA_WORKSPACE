# Transcribir Clase — Whisper (OpenAI)

Convierte grabaciones de audio de clases a texto usando el modelo `large` de Whisper.

---

## Requisitos

- Entorno conda `ml_env` con `openai-whisper` instalado
- `ffmpeg` instalado via Homebrew (`/opt/homebrew/bin/ffmpeg`)
- Modelo `large-v3` (~2.9 GB) descargado en `~/.cache/whisper/`

---

## Uso

```bash
conda activate ml_env
python /Users/joelbecerril/MNA_WORKSPACE/transcribir_clase.py <archivo_audio>
```

### Ejemplo

```bash
python /Users/joelbecerril/MNA_WORKSPACE/transcribir_clase.py ~/Downloads/GrabacionSem1.m4a
```

---

## Formatos de audio soportados

`mp3`, `mp4`, `m4a`, `wav`, `ogg`, `webm`

---

## Salida

- La transcripción se guarda en la **misma carpeta que el audio**, con sufijo `_large.txt`
- Ejemplo: `GrabacionSem1.m4a` → `GrabacionSem1_large.txt`
- Al terminar imprime el tiempo total de procesamiento:

```
Listo en 18m 23s
Transcripción guardada en: /Users/joelbecerril/Downloads/GrabacionSem1_large.txt
```

---

## Tiempo de procesamiento estimado

| Duración del audio | Tiempo aproximado (CPU) |
|---|---|
| 30 minutos | ~18 minutos |
| 60 minutos | ~35-40 minutos |
| 90 minutos | ~55-65 minutos |

> El modelo `large` es el más lento pero da la mejor calidad de transcripción en español.

---

## Cómo funciona el modelo

Whisper `large` es una **red neuronal de tipo Transformer** entrenada por OpenAI con ~680 millones de parámetros y 680,000 horas de audio.

### Proceso interno de transcripción

1. **Audio → Espectrograma** — `ffmpeg` convierte el audio en un mel spectrogram: una representación visual de las frecuencias del sonido a lo largo del tiempo.
2. **Encoder** — una red neuronal analiza el espectrograma y extrae patrones del habla: fonemas, palabras, acentos, pausas.
3. **Decoder** — otro Transformer genera el texto palabra por palabra, condicionado al idioma indicado (`language="es"`).
4. **Ventanas de 30 segundos** — el audio se procesa en bloques de 30s, por eso el progreso aparece con timestamps `[00:00 --> 00:30]` en la terminal.

### Por qué pesa 2.9 GB

Son los **pesos** de la red — millones de números que representan todo lo que el modelo aprendió durante el entrenamiento. Se descargan una sola vez y se reutilizan en cada transcripción desde `~/.cache/whisper/large-v3.pt`.

### En términos de negocio

Es equivalente a un transcriptor humano muy experimentado entrenado con cientos de miles de horas de audio en 99 idiomas — pero que trabaja **offline en tu máquina** sin enviar el audio a ningún servidor externo.

---

## Notas

- La primera vez descarga el modelo automáticamente (~2.9 GB). Las siguientes veces lo carga desde disco en segundos, el mensaje "Cargando modelo..." no implica descarga.
- El modelo se guarda en `~/.cache/whisper/large-v3.pt` y se reutiliza en ejecuciones posteriores.
- Si aparece el error `ffmpeg not found`, abrir una nueva terminal para recargar el PATH.
