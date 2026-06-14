# OpenClaw — Guía completa (mi setup)

## Las 3 capas (entender esto lo aclara todo)
- **VM (Multipass):** la máquina. Se prende/apaga.
- **Gateway:** el motor del agente. Corre de fondo (servicio systemd) dentro de la VM. Atiende Telegram. Sigue vivo aunque cierres la terminal.
- **TUI (`openclaw tui`):** la ventana de chat. Abrirla o cerrarla NO afecta al gateway.

## La regla de oro: ¿dónde corro cada comando?
- Comandos **`multipass ...`** → en la **Mac** (afuera de la VM).
- Comandos **`openclaw ...`** o de Linux (`apt`, etc.) → **dentro** de la VM (después de `multipass shell`).

---

## ▶️ INICIAR / ABRIR (empezar a usar)
```
      # solo si la VM está apagada
multipass start openclaw-lab

      # entrar a la VM
multipass shell openclaw-lab

                      # abrir el chat con el agente
openclaw tui

```
- Si `openclaw` dice **"command not found"**: corre `source ~/.bashrc` y reintenta.
- Recuerda: `start` = prender la VM (botón de encendido); `shell` = entrar a la VM ya prendida.

## 💬 Hablarle al agente
- **En la Mac:** `openclaw tui` (dentro de la VM)
- **Desde cualquier lado:** por **Telegram** (mensaje a tu bot)

## ⏸️ CERRAR el chat (dejando Telegram vivo)
- Salir del TUI: `Ctrl + C` (o `/quit`). El gateway sigue → Telegram responde mientras la VM esté prendida.

## ⏹️ APAGAR (rutina segura) — IMPORTANTE
Para terminar por hoy, **apaga la VM tú primero, limpio, ANTES de apagar la Mac**:
```
exit                              # salir del shell (si estás dentro)

       # apagar la VM LIMPIO
multipass stop openclaw-lab
# recién entonces apaga la Mac
```
⚠️ **Nunca apagues la Mac con la VM corriendo.** Eso puede dejar la VM en estado "Unknown"/atorado y corromper la imagen (fue lo que nos costó horas). Apagar la VM tú primero lo evita.

Si solo **cierras la tapa** (la Mac se duerme), la VM se pausa y se reanuda al despertar — ok para ratos cortos, pero para descansos largos mejor `multipass stop`.

---

## 🆘 Si la VM se atora (Unknown / Starting eterno / timeout)
En orden, hasta que arranque:
1. Forzar apagado y reintentar:
   ```
   multipass stop --force openclaw-lab
   multipass start openclaw-lab
   ```
2. Reiniciar el demonio de Multipass (en la Mac):
   ```
   sudo launchctl unload /Library/LaunchDaemons/com.canonical.multipassd.plist
   sudo launchctl load -w /Library/LaunchDaemons/com.canonical.multipassd.plist
   multipass start openclaw-lab
   ```
3. **Reiniciar el Mac**, luego `multipass start openclaw-lab`.
4. Último recurso: **restaurar el snapshot** (ver abajo).

Nota: "cannot connect to the multipass socket" = el demonio se cayó → reiniciar el Mac lo arregla.

## 🛟 Snapshots (red de seguridad)
- Listar (en la Mac): `multipass list --snapshots`
- Crear (VM apagada):
  ```
  multipass stop openclaw-lab
  multipass snapshot --name <nombre> openclaw-lab
  multipass start openclaw-lab
  ```
- Restaurar:
  ```
  multipass stop --force openclaw-lab
  multipass restore openclaw-lab.<snapshot>
  multipass start openclaw-lab
  ```
- ⚠️ Restaurar te regresa al estado del snapshot — **todo lo posterior se pierde**. Por eso conviene tener un snapshot "todo funcionando".

**Mis snapshots:**
- `snapshot1` = Ubuntu limpio (base, ANTES de OpenClaw)
- `<crear>` = con OpenClaw + Telegram montado ← tomar este después de reinstalar

---

## 🔧 Reinstalar OpenClaw (dentro de la VM)
```
sudo apt update && sudo apt install -y curl
curl -fsSL https://openclaw.ai/install.sh | bash
```
**Onboarding seguro:**
- Advertencia de seguridad → **Yes**
- Modo → **QuickStart**
- Binding de red → **Local (this machine)** / `127.0.0.1`
- Modelo → **Sonnet** (`claude-sonnet-4-6`) o **Skip**
- API key → la del tope de gasto (**sin screenshot** de la key)

Post-install útil: regenerar el gateway token →
```
openclaw doctor --generate-gateway-token
```

## 📲 Reconfigurar Telegram (dentro de la VM, UNO POR UNO)
```
openclaw config set channels.telegram.enabled true
openclaw config set channels.telegram.botToken "TU_TOKEN"
openclaw config set channels.telegram.dmPolicy "allowlist"
openclaw config set channels.telegram.allowFrom '["tg:6942720774"]'
openclaw config set commands.ownerAllowFrom '["telegram:6942720774"]'
openclaw config set plugins.entries.telegram.enabled true
```
⚠️ Córrelos **uno a la vez** y cuida la **comilla de cierre** en los de corchetes (`'[...]'`).
Luego reinicia el gateway y verifica con `openclaw status` (Telegram debe salir **ON**).

---

## 🧠 Modelos: nube vs local
- **Por defecto:** Sonnet (`claude-sonnet-4-6`) por API → de pago, lo más capaz.
- **Local con Ollama** (en el Mac) → gratis y privado. **(Pendiente de conectar.)**
  - Modelos ya bajados: `gpt-oss:20b`, `deepseek-r1:14b`, `qwen2.5-coder:7b`
  - Pendiente de bajar: `qwen3.6:27b`
  - Conexión: abrir Ollama hacia la VM + configurar el provider con la **URL nativa (sin `/v1`)** para que el tool calling funcione.
- **Cambiar de modelo en el chat:** `/model <modelo>`
  - Simple/rápido → `ollama/gpt-oss:20b`
  - Complejo → `ollama/qwen3.6:27b`
  - Lo más difícil → `claude-sonnet-4-6` (nube, de pago)
- ⚠️ **Memoria:** `gpt-oss:20b` (~16 GB) + la VM en 24 GB queda apretado. Para convivir bien: usar un modelo más chico, o bajarle RAM a la VM.
- Ollama en el Mac: `brew services start ollama` lo prende; `ollama list` muestra modelos.

---

## 🔒 Seguridad — recordatorios
- **Nunca** compartir ni capturar en screenshot: la **API key** (`sk-ant-...`) ni el **token de Telegram**. (El user ID sí, no es secreto.)
- Telegram candadeado: **solo tu ID** (allowlist + owner).
- Gateway solo en **localhost (127.0.0.1)** — no expuesto.
- Tope de gasto: API key con spend limit + créditos prepagados.
- **Pendiente a futuro:** blindar secretos (SecretRefs) si manejas datos reales/sensibles, respaldas/compartes la VM, o pasas a producción.
- Datos **no sensibles** mientras sea experimental.

## 📌 Mi setup (referencia)
- **VM:** openclaw-lab (Ubuntu 26.04 LTS, Multipass, ARM64) — 4 GB RAM / 2 CPU
- **IPv4 de la VM:** 192.168.252.2 · **Mac (gateway):** 192.168.252.1
- **Modelo nube:** claude-sonnet-4-6
- **Mac:** MacBook Air M5, 24 GB RAM
- **Telegram user ID:** 6942720774
- **Gateway/Dashboard:** http://127.0.0.1:18789 (solo dentro de la VM)
