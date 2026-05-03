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

## Notas

- La primera vez descarga el modelo automáticamente (~2.9 GB).
- El modelo se guarda en `~/.cache/whisper/large-v3.pt` y se reutiliza en ejecuciones posteriores.
- Si aparece el error `ffmpeg not found`, abrir una nueva terminal para recargar el PATH.
