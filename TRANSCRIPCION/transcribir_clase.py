import whisper
import sys
import os
import time

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

print(f"Cargando modelo... (solo la primera vez tarda en descargar)")
model = whisper.load_model("large")

print(f"Transcribiendo '{audio}'...")
inicio = time.time()
result = model.transcribe(audio, language="es", verbose=True)
fin = time.time()

# Guarda la transcripción incluyendo el modelo en el nombre del archivo
output_file = os.path.splitext(audio)[0] + "_large.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

minutos = int((fin - inicio) // 60)
segundos = int((fin - inicio) % 60)
print(f"\nListo en {minutos}m {segundos}s")
print(f"Transcripción guardada en: {output_file}")
