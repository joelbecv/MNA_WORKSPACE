# OpenClaw — Guía rápida (mi setup)

## Las 3 capas (entender esto lo aclara todo)
- **VM (Multipass):** la máquina. Se prende/apaga.
- **Gateway:** el motor del agente. Corre de fondo (servicio systemd) dentro de la VM. Es el que atiende Telegram. Sigue vivo aunque cierres la terminal.
- **TUI (`openclaw tui`):** la ventana de chat en la terminal. Abrirla o cerrarla **NO** afecta al gateway.

---

## ▶️ INICIAR / ABRIR (empezar a usar)
Desde una terminal en la Mac:
```
multipass start openclaw-lab
      # solo si la VM está apagada
multipass shell openclaw-lab
      # entrar a la VM
openclaw tui
                      # abrir el chat con el agente
```
- El gateway y Telegram **arrancan solos** con la VM.
- Si `openclaw tui` dice "command not found": corre `source ~/.bashrc` y reintenta.

## 💬 Hablarle al agente
- **En la Mac:** `openclaw tui` (dentro de la VM)
- **Desde cualquier lado:** por **Telegram** (mensaje a tu bot)

## ⏸️ CERRAR el chat (dejando Telegram vivo)
- Salir del TUI: `Ctrl + C`  (o `/quit`)
- El gateway sigue corriendo → Telegram sigue respondiendo mientras la VM esté prendida.

## ⏹️ APAGAR todo (lo más seguro, cuando termines del todo)
```
exit                              # salir de la VM (si estás dentro)
multipass stop openclaw-lab       # apagar la VM
```
- Agente offline, Telegram offline, **cero superficie de ataque, cero gasto**.
- Este es el modelo "on-demand": cuando no lo uses, apágalo.

---

## 🔧 Comandos útiles (dentro de la VM)
```
openclaw status                   # estado general
openclaw status --deep            # estado real de los canales (Telegram OK)
openclaw logs --follow            # ver logs en vivo (salir con Ctrl+C)
openclaw security audit           # reporte de seguridad
openclaw security audit --deep    # auditoría de seguridad profunda
openclaw doctor                   # diagnóstico y arreglos
openclaw config get <clave>       # ver un valor de config
source ~/.bashrc                  # si "openclaw" no se encuentra
```

## 🖥️ Manejo de la VM (desde la Mac, FUERA de la VM)
```
multipass list                    # listar VMs y su estado
multipass info openclaw-lab       # ver CPU/RAM/disco que usa
multipass start openclaw-lab      # prender
multipass stop openclaw-lab       # apagar
```

## 🛟 Red de seguridad (snapshot)
Restaurar al punto limpio si algo se rompe:
```
multipass stop openclaw-lab
multipass restore openclaw-lab.snapshot1
multipass start openclaw-lab
```
Crear un nuevo snapshot (con la VM apagada):
```
multipass stop openclaw-lab
multipass snapshot openclaw-lab
```

---

## 🔒 Recordatorios de seguridad
- **Nunca** compartir ni capturar en screenshot: la **API key** (`sk-ant-...`) ni el **token de Telegram**. (El user ID sí se puede, no es secreto.)
- Telegram está candadeado: **solo tu ID** (allowlist + owner). Nadie más le puede hablar ni aprobar acciones.
- El gateway solo escucha en **localhost (127.0.0.1)** — no expuesto a internet.
- **Pendiente a futuro:** blindar secretos (SecretRefs) si algún día manejas datos reales/sensibles, respaldas o compartes la VM, o pasas a producción.
- Gasto topado: API key con spend limit + créditos prepagados.

## 📌 Mi setup (referencia)
- **VM:** openclaw-lab (Ubuntu, Multipass)
- **Modelo:** claude-sonnet-4-6 (Sonnet)
- **Snapshot de respaldo:** openclaw-lab.snapshot1
- **Gateway/Dashboard:** http://127.0.0.1:18789 (solo dentro de la VM)
