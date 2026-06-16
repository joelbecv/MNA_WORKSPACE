#!/bin/bash
# sync-openclaw.sh — Sync incremental de OpenClaw a Google Drive
# Uso: ./sync-openclaw.sh          (solo lo nuevo)
#       ./sync-openclaw.sh --todo   (sync completo)

VM="openclaw-lab"
DRIVE="$HOME/Library/CloudStorage/GoogleDrive-baldjoel@gmail.com/Mi unidad/OpenClawdocs"
WORKSPACE="/home/ubuntu/.openclaw/workspace"
MARKER="$DRIVE/.last-sync"
MAX_SIZE_MB=10

STATUS=$(multipass info "$VM" --format csv 2>/dev/null | tail -1 | cut -d',' -f2)
if [ "$STATUS" != "Running" ]; then
    echo "⚠️  La VM '$VM' no está corriendo. Préndela con: multipass start $VM"
    exit 1
fi

mkdir -p "$DRIVE"

if [ "$1" = "--todo" ] || [ ! -f "$MARKER" ]; then
    echo "🔄 Sync completo"
    TIME_FILTER=""
else
    LAST_SYNC=$(cat "$MARKER")
    echo "🔄 Sync incremental (cambios desde $LAST_SYNC)"
    TIME_FILTER="-newermt '$LAST_SYNC'"
fi

FIND_CMD="find $WORKSPACE -type f \( -name '*.txt' -o -name '*.ipynb' -o -name '*.py' -o -name '*.csv' -o -name '*.xlsx' -o -name '*.pdf' -o -name '*.zip' -o -name '*.ics' -o -name '*.html' -o -name '*.sh' -o -name '*.png' -o -name '*.jpg' -o -name '*.mp3' -o -name '*.wav' -o -name '*.docx' -o -name '*.pptx' \) -not -name 'package*.json' $TIME_FILTER 2>/dev/null"

FILES=$(multipass exec "$VM" -- bash -c "$FIND_CMD")

FIND_HOME="find /home/ubuntu -maxdepth 1 -type f \( -name '*.txt' -o -name '*.ipynb' -o -name '*.py' -o -name '*.csv' \) $TIME_FILTER 2>/dev/null"
HOME_FILES=$(multipass exec "$VM" -- bash -c "$FIND_HOME")

ALL_FILES=$(echo -e "$FILES\n$HOME_FILES" | sort -u | grep -v '^$')

if [ -z "$ALL_FILES" ]; then
    echo "📭 No hay archivos nuevos."
    exit 0
fi

echo ""
echo "📂 Archivos encontrados:"
echo "$ALL_FILES"
echo ""
echo "📥 Copiando a Google Drive..."

COUNT=0
SKIPPED=0
while IFS= read -r file; do
    REL_PATH="${file#$WORKSPACE/}"
    if [[ "$REL_PATH" == "$file" ]]; then
        REL_PATH="${file#/home/ubuntu/}"
    fi

    # Verificar tamaño del archivo
    SIZE_BYTES=$(multipass exec "$VM" -- stat -c%s "$file" 2>/dev/null </dev/null)
    SIZE_MB=$((SIZE_BYTES / 1048576))

    if [ "$SIZE_MB" -ge "$MAX_SIZE_MB" ]; then
        echo ""
        echo "  🔒 $REL_PATH pesa ${SIZE_MB}MB (mayor a ${MAX_SIZE_MB}MB)"
        read -p "     ¿Sincronizar este archivo? (s/n): " RESPUESTA </dev/tty
        if [ "$RESPUESTA" != "s" ] && [ "$RESPUESTA" != "S" ]; then
            echo "     ⏭️  Saltado."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    DIR_PATH=$(dirname "$REL_PATH")
    mkdir -p "$DRIVE/$DIR_PATH"
    multipass transfer "$VM:$file" "$DRIVE/$REL_PATH" 2>/dev/null </dev/null
    if [ $? -eq 0 ]; then
        COUNT=$((COUNT + 1))
        echo "  ✅ $REL_PATH"
    fi
done <<< "$ALL_FILES"

date '+%Y-%m-%d %H:%M:%S' > "$MARKER"

echo ""
echo "✅ $COUNT archivos sincronizados."
if [ "$SKIPPED" -gt 0 ]; then
    echo "⏭️  $SKIPPED archivos saltados por tamaño."
fi
echo "   📁 $DRIVE"