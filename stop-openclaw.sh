#!/bin/bash
# stop-openclaw.sh — Apaga todo el stack de OpenClaw
# Uso: ~/MNA_WORKSPACE/stop-openclaw.sh

echo "🔒 Apagando stack OpenClaw..."

# --- Whisper server ---
if pgrep -f whisper-server > /dev/null; then
    echo "  🎙️ Apagando Whisper..."
    pkill -f whisper-server
    echo "  ✅ Whisper apagado"
fi

# --- Ollama ---
if brew services list | grep ollama | grep started > /dev/null 2>&1; then
    echo "  🧠 Apagando Ollama..."
    brew services stop ollama
    echo "  ✅ Ollama apagado"
fi

# --- VM (siempre al final) ---
echo "  📦 Apagando VM..."
multipass stop openclaw-lab
echo "  ✅ VM apagada"

echo ""
echo "🔒 Todo apagado. Puedes apagar la Mac tranquilo."
