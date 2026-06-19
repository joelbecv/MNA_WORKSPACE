# OpenClaw — Documentación de Scripts

Scripts de administración del stack OpenClaw ubicados en `~/MNA_WORKSPACE/`.  
Todos requieren que **Multipass** esté instalado (`brew install --cask multipass`).

---

## Flujo de trabajo normal

```
start-openclaw.sh  →  (trabajas)  →  sync-openclaw.sh  →  stop-openclaw.sh
```

---

## start-openclaw.sh — Encender el stack

**Qué hace:**  
Prende la VM `openclaw-lab` y los servicios que necesitas para trabajar.  
Ollama (modelos locales) está activado por defecto; Whisper está comentado.

**Cuándo usarlo:** al comenzar una sesión de trabajo con OpenClaw.

```bash
~/MNA_WORKSPACE/start-openclaw.sh
```

**Servicios que levanta (en orden):**

| # | Servicio | Estado por defecto | Para qué sirve |
|---|----------|--------------------|----------------|
| 1 | VM `openclaw-lab` | ✅ Activo | Agente principal — siempre necesario |
| 2 | Ollama | ✅ Activo | Modelos locales (`qwen3`, `deepseek-r1`, etc.) |
| 3 | Whisper server | 💤 Comentado | Transcripción de audio (puerto 5555) |

**Cómo activar Whisper:**  
Descomentar las líneas del bloque `# --- 3. WHISPER ---` en el script.

**Salida esperada al terminar:**
```
🦞 Stack listo. Opciones:
   Terminal:  multipass shell openclaw-lab → openclaw tui
   Telegram:  mándale mensaje a tu bot
   Sync:      ~/MNA_WORKSPACE/sync-openclaw.sh
```

---

## sync-openclaw.sh — Sincronizar archivos a Google Drive

**Qué hace:**  
Copia archivos del workspace de la VM (`/home/ubuntu/.openclaw/workspace`) y del home  
de la VM (`/home/ubuntu/`) hacia Google Drive en `OpenClawdocs`.

**Cuándo usarlo:** antes de apagar, o cuando quieras respaldar avances en Drive.

```bash
# Solo archivos nuevos/modificados desde el último sync (modo normal)
~/MNA_WORKSPACE/sync-openclaw.sh

# Todos los archivos sin importar fecha (primer uso o para forzar rebuild)
~/MNA_WORKSPACE/sync-openclaw.sh --todo
```

**Extensiones sincronizadas:**  
`.txt` `.ipynb` `.py` `.csv` `.xlsx` `.pdf` `.zip` `.ics` `.html` `.sh`  
`.png` `.jpg` `.mp3` `.wav` `.docx` `.pptx`

**Archivos ≥ 10 MB:** el script pregunta interactivamente `(s/n)` antes de copiarlos.

**Destino en Drive:**
```
~/Library/CloudStorage/GoogleDrive-baldjoel@gmail.com/Mi unidad/OpenClawdocs/
```

**Requisitos:**  
- La VM `openclaw-lab` debe estar corriendo.  
  Si no está, el script sale con: `⚠️  La VM 'openclaw-lab' no está corriendo.`  
  Prenderla con: `multipass start openclaw-lab`
- Google Drive for Desktop instalado y montado en la ruta de arriba.

**Nota técnica (fix 2026-06-15):**  
Los comandos `multipass exec` y `multipass transfer` dentro del loop  
usan `</dev/null` para no interferir con el `while read` que itera la  
lista de archivos. Sin este fix, solo se copiaba el primer archivo.

---

## stop-openclaw.sh — Apagar el stack

**Qué hace:**  
Detiene Whisper (si corre), para Ollama vía `brew services`, y apaga la VM.

**Cuándo usarlo:** al terminar la sesión. Siempre **después** del sync.

```bash
~/MNA_WORKSPACE/stop-openclaw.sh
```

**Orden de apagado:**

| # | Servicio | Cómo lo apaga |
|---|----------|---------------|
| 1 | Whisper server | `pkill -f whisper-server` (solo si está corriendo) |
| 2 | Ollama | `brew services stop ollama` (solo si está activo) |
| 3 | VM `openclaw-lab` | `multipass stop openclaw-lab` — **siempre al final** |

**Salida esperada:**
```
🔒 Todo apagado. Puedes apagar la Mac tranquilo.
```

---

## Referencia rápida

```bash
# Flujo completo de una sesión
~/MNA_WORKSPACE/start-openclaw.sh        # encender

multipass shell openclaw-lab             # entrar a la VM
# → openclaw tui                         # iniciar el agente dentro de la VM

~/MNA_WORKSPACE/sync-openclaw.sh         # respaldar antes de cerrar
~/MNA_WORKSPACE/stop-openclaw.sh         # apagar todo

# Diagnóstico rápido
multipass list                           # ver estado de VMs
multipass info openclaw-lab              # detalle de la VM
brew services list | grep ollama         # estado de Ollama
```

---

*Última actualización: 2026-06-17*
