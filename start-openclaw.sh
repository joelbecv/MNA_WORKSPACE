#!/bin/bash
# start-openclaw.sh — Prende todo el stack de OpenClaw
# Comenta con # lo que no necesites hoy.
# Uso: ~/MNA_WORKSPACE/start-openclaw.sh

echo "🚀 Iniciando stack OpenClaw..."

# --- 1. SIEMPRE: VM del agente ---
echo "  📦 Prendiendo VM..."
multipass start openclaw-lab
echo "  ✅ VM lista"

# --- 2. MODELOS LOCALES: Ollama ---
# Descomentar si vas a usar modelos locales (qwen3, deepseek-r1, etc.)
# Si solo usas DeepSeek API o Claude, puedes dejarlo comentado.
echo "  🧠 Prendiendo Ollama..."
brew services start ollama
launchctl setenv OLLAMA_HOST 0.0.0.0
echo "  ✅ Ollama listo"

# --- 3. WHISPER: Transcripción de audio ---
# Descomentar cuando necesites transcribir audio.
# echo "  🎙️ Prendiendo Whisper server..."
# cd ~/MNA_WORKSPACE && conda run --no-capture-output -n ml_env python whisper-server.py &
# echo "  ✅ Whisper listo en http://0.0.0.0:5555"

# --- 4. [FUTURO] Nuevo servicio ---
# Agrega aquí nuevos servicios conforme los necesites.
# Mismo patrón: echo + comando + echo confirmación.
# echo "  🔧 Prendiendo [servicio]..."
# [comando para prender]
# echo "  ✅ [servicio] listo"

echo ""
echo "🦞 Stack listo. Opciones:"
echo "   Terminal:  multipass shell openclaw-lab → openclaw tui"
echo "   Telegram:  mándale mensaje a tu bot"
echo "   Sync:      ~/MNA_WORKSPACE/sync-openclaw.sh"
