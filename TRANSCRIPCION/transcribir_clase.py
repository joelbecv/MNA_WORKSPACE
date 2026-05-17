import sys
import os
import time
import subprocess
import threading
from faster_whisper import WhisperModel
from tqdm import tqdm

# Uso: python transcribir_clase.py archivo_audio.mp3
# Formatos soportados: mp3, mp4, m4a, wav, ogg, webm

if len(sys.argv) < 2:
    print("Uso: python transcribir_clase.py <archivo_audio>")
    print("Ejemplo: python transcribir_clase.py clase_semana1.mp3")
    sys.exit(1)

audio = sys.argv[1]

if not os.path.exists(audio):
    print(f"ERROR: no se encontró el archivo '{audio}'")
    sys.exit(1)

def get_duration(path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

print(f"Cargando modelo...")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

duracion_audio = get_duration(audio)
print(f"Transcribiendo '{audio}'...")
if duracion_audio:
    mins = int(duracion_audio // 60)
    secs = int(duracion_audio % 60)
    print(f"Duración del audio: {mins}m {secs}s")

inicio = time.time()
transcripcion_lista = threading.Event()
result_container = {}

def transcribir():
    segments, info = model.transcribe(audio, language="es", beam_size=1)
    result_container['text'] = ''.join(seg.text for seg in segments)
    transcripcion_lista.set()

hilo = threading.Thread(target=transcribir)
hilo.start()

total_estimado = int(duracion_audio * 0.5) if duracion_audio else 120

with tqdm(total=total_estimado, unit="s", desc="Transcribiendo",
          bar_format="{l_bar}{bar}| {elapsed} transcurrido") as pbar:
    while not transcripcion_lista.is_set():
        time.sleep(1)
        if pbar.n < total_estimado - 1:
            pbar.update(1)
    pbar.update(total_estimado - pbar.n)

hilo.join()
fin = time.time()

output_file = os.path.splitext(audio)[0] + "_large.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result_container['text'])

minutos = int((fin - inicio) // 60)
segundos = int((fin - inicio) % 60)
print(f"\nListo en {minutos}m {segundos}s")
print(f"Transcripción guardada en: {output_file}")
