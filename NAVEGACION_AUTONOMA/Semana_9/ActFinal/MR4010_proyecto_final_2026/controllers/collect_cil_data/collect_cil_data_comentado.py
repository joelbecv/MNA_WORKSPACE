"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  GUÍA DE LECTURA — collect_cil_data_comentado.py                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CÓMO LEER ESTE ARCHIVO:                                                     ║
║  • Bloques NARANJA (como este)  → guía pedagógica: qué hace, por qué,       ║
║    dónde se usó antes y qué decir en el VIDEO                               ║
║  • Comentarios # grises         → notas técnicas cortas sobre líneas         ║
║                                                                              ║
║  PROPÓSITO DEL CONTROLADOR:                                                  ║
║  Recolección de datos de entrenamiento para Conditional Imitation            ║
║  Learning (CIL). El conductor humano maneja el BMW en el Mundo 1 mientras   ║
║  el controlador captura automáticamente imágenes + ángulo de dirección +     ║
║  comando de navegación → dataset para entrenar la CNN en Google Colab.       ║
║                                                                              ║
║  REFERENCIA:                                                                 ║
║  Codevilla et al. 2017 — "End-to-end Driving via Conditional Imitation       ║
║  Learning" (arxiv 1710.02410)                                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from controller import Keyboard
from vehicle import Driver
import numpy as np
import cv2
import os
import csv

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 1 — PARÁMETROS DEL SISTEMA                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Define todas las constantes del controlador en un solo lugar.               ║
║  Cambiar un parámetro aquí lo afecta en todo el programa.                   ║
║                                                                              ║
║  DECISIONES DE DISEÑO:                                                       ║
║  • CRUISE_SPEED = 30 km/h: velocidad estándar urbana. La rúbrica pide       ║
║    velocidad constante durante la recolección. Más rápido → imágenes         ║
║    borrosas y menos tiempo de reacción en intersecciones.                    ║
║  • CAPTURE_EVERY = 5 frames × 16 ms = 80 ms → ~12.5 fps:                   ║
║    suficiente para capturar cambios de dirección sin llenar el disco.        ║
║    A 30 km/h el auto avanza 1 m en 120 ms → no se pierden eventos.          ║
║  • STEER_STEP = 0.015 rad: incremento fino por pulsación de tecla.          ║
║    Permite giros suaves, similares a un joystick analógico.                  ║
║                                                                              ║
║  COMANDOS DE NAVEGACIÓN (esquema Codevilla 2017):                            ║
║    0 = RECTO      → el auto debe seguir derecho en la intersección           ║
║    1 = IZQUIERDA  → el auto girará a la izquierda                           ║
║    2 = DERECHA    → el auto girará a la derecha                              ║
║  El comando se fija con tecla A/W/D y se MANTIENE hasta el próximo          ║
║  cambio. Esto es intencional: el conductor señala la intención               ║
║  ANTES de llegar a la intersección, igual que en conducción real.            ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 2.1 (H2/H3): CRUISE_SPEED=30, MAX_ANGLE=0.5, misma filosofía        ║
║  • Act 4.2 (evasión bus): SPEED_FOLLOW=30, STEER_STEP implícito en PID      ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Mostrar esta sección y mencionar: "elegimos 30 km/h porque la             ║
║    rúbrica pide velocidad constante y nos da suficiente tiempo de            ║
║    reacción para señalar los comandos antes de cada intersección"            ║
║  → "Capturamos a 12.5 fps: más de 750 imágenes por minuto de conducción"    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

CRUISE_SPEED  = 30       # km/h — constante durante toda la recolección
MAX_ANGLE     = 0.5      # rad  — límite físico del volante BmwX5
STEER_STEP    = 0.015    # rad  — incremento por pulsación de flecha
CENTER_DECAY  = 0.92     # factor de retorno al centro con tecla ↑
CAPTURE_EVERY = 5        # 1 imagen cada N frames (5 × 16 ms = 80 ms)

CMD_STRAIGHT = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_LABEL    = {0: "RECTO    ", 1: "IZQUIERDA", 2: "DERECHA  "}

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 2 — INICIALIZACIÓN DEL DRIVER Y SENSORES                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Conecta el controlador Python con los dispositivos físicos del BMW X5       ║
║  simulado. Cada getDevice() busca el dispositivo por nombre en el .wbt.     ║
║                                                                              ║
║  REGLA CRÍTICA — timestep:                                                   ║
║  timestep = int(driver.getBasicTimeStep())  → 16 ms en estos mundos         ║
║  NUNCA hacer timestep * N. En la Actividad 3.1 descubrimos que              ║
║  multiplicar el timestep causaba freeze (beachball) en macOS.               ║
║  El controlador H3 de Act 2.1 que funcionó usaba esta misma línea exacta.  ║
║                                                                              ║
║  REGLA CRÍTICA — engineSound:                                                ║
║  El PROTO BmwX5 buscaba un archivo de audio inexistente. Cada frame          ║
║  generaba un warning que bloqueaba el hilo → freeze. Solución ya aplicada   ║
║  en el .wbt: engineSound ""                                                  ║
║                                                                              ║
║  DISPOSITIVOS EN ESTE CONTROLADOR:                                           ║
║  • camera   → "camera" (default BmwX5, slot superior, 320×160 px)           ║
║  • display  → "display_image" (nombre explícito que agregamos al .wbt)      ║
║  • Keyboard → singleton global, no requiere nombre de dispositivo           ║
║                                                                              ║
║  NO se habilitan LiDAR ni radar aquí porque en World 1 no hay tráfico.      ║
║  Esos sensores son para World 2 (autonomous_cil.py).                         ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 2.1 (H3): mismo patrón Driver() + getBasicTimeStep() sin mult.       ║
║  • Act 4.2: getDevice("camera"), getDevice("display_image"), Keyboard()      ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → "Reutilizamos exactamente el mismo patrón de inicialización que           ║
║     funcionó en las Actividades 2.1 y 4.2 para evitar regresiones"          ║
║  → Mostrar las dos líneas camera.enable() y Keyboard() como evidencia        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

driver   = Driver()
timestep = int(driver.getBasicTimeStep())   # 16 ms — NUNCA multiplicar

camera = driver.getDevice("camera")
camera.enable(timestep)
CAM_W = camera.getWidth()    # 320 px
CAM_H = camera.getHeight()   # 160 px

display = driver.getDevice("display_image")   # 200×150 px

keyboard = Keyboard()
keyboard.enable(timestep)

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 3 — SISTEMA DE ARCHIVOS Y CSV                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Crea las carpetas de salida y el archivo CSV que vincula cada imagen        ║
║  con su ángulo de dirección y su comando de navegación.                      ║
║                                                                              ║
║  ESTRUCTURA DEL CSV:                                                         ║
║    image_path          , steering_angle , nav_command                        ║
║    data/images/img_000001.jpg , 0.042   , 0                                 ║
║    data/images/img_000150.jpg , -0.18   , 1                                 ║
║                                                                              ║
║  DISEÑO DE SESIONES MÚLTIPLES:                                               ║
║  El CSV se abre en modo "append" (a) para acumular datos de múltiples       ║
║  sesiones de conducción. Cuatro integrantes conducen en turnos y             ║
║  cada sesión agrega filas al mismo CSV sin borrar las anteriores.            ║
║  → Solo se escribe el header si el archivo es nuevo.                         ║
║                                                                              ║
║  RUTAS RELATIVAS:                                                            ║
║  Guardamos "data/images/img_XXXXXX.jpg" (ruta relativa) en el CSV           ║
║  para que el dataset sea portable entre las computadoras del equipo.         ║
║  El notebook de Colab hace git clone del repo → las rutas resuelven         ║
║  correctamente sin importar en qué PC se generaron.                          ║
║                                                                              ║
║  REANUDACIÓN AUTOMÁTICA:                                                     ║
║  img_count arranca contando las imágenes existentes en DATA_DIR.             ║
║  Si hay 2,500 imágenes previas, la próxima se llama img_002500.jpg.          ║
║  Evita sobreescribir datos de sesiones anteriores.                           ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 3.1 (collect_training_data): mismo patrón os.makedirs + CSV          ║
║  • Primera vez que usamos rutas relativas para portabilidad entre PCs        ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → "El dataset se construyó de forma incremental: cada integrante condujo   ║
║     ~35 minutos y sus imágenes se acumularon en el mismo CSV"               ║
║  → Mostrar el archivo CSV abierto en Excel/VSCode para visualizar           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

CTRL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "images"))
CSV_PATH = os.path.normpath(os.path.join(CTRL_DIR, "..", "..", "data", "dataset.csv"))
os.makedirs(DATA_DIR, exist_ok=True)

# Modo append: múltiples sesiones se acumulan sin borrar datos previos
csv_exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
csv_file   = open(CSV_PATH, "a", newline="")
csv_writer = csv.writer(csv_file)
if not csv_exists:
    csv_writer.writerow(["image_path", "steering_angle", "nav_command"])
    csv_file.flush()

# Contar imágenes previas para continuar la numeración sin sobreescribir
existing_imgs = [f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
img_count = len(existing_imgs)

print(f"[CIL-COLLECT] Reanudando desde imagen #{img_count}")

# =============================================================================
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🎬  BLOQUE 4 — LOOP PRINCIPAL: TECLADO Y CONTROL DE VOLANTE                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  QUÉ HACE:                                                                   ║
║  Lee las teclas presionadas y las convierte en dos tipos de acción:          ║
║    a) Ajuste del ángulo de dirección (flechas ← / →)                        ║
║    b) Cambio del comando de navegación activo (teclas A / W / D)             ║
║                                                                              ║
║  SEPARACIÓN DE CONTROLES (diseño CIL):                                       ║
║  Las flechas ← / → controlan el volante EN TIEMPO REAL (como un game).      ║
║  Las teclas A / W / D fijan el COMANDO DE NAVEGACIÓN que se grabará          ║
║  en el CSV. Son dos acciones independientes:                                  ║
║    • El conductor puede girar el volante a la derecha (←/→)                 ║
║      mientras señala que en la PRÓXIMA intersección irá a la izquierda (A)  ║
║    • El comando grabado NO afecta el volante — solo se anota en el CSV       ║
║                                                                              ║
║  PATRÓN keyboard.getKey() en BUCLE:                                           ║
║  Webots acumula varias teclas entre frames. Si hay 3 teclas presionadas      ║
║  en el mismo timestep, el bucle las procesa todas. Sin el bucle, se          ║
║  perderían pulsaciones rápidas.                                               ║
║                                                                              ║
║  CENTER_DECAY = 0.92:                                                        ║
║  Con ↑ el steering se multiplica por 0.92 cada frame (~16ms).               ║
║  A los 40 frames (~640ms) el steering está en <5% del original.              ║
║  Imita el retorno de volante de un auto real al soltar el control.           ║
║                                                                              ║
║  USADO ANTES EN:                                                              ║
║  • Act 4.2: exactamente el mismo bucle keyboard.getKey() para detectar      ║
║    teclas y controlar estados. Patrón probado y validado.                    ║
║  • Act 2.1 (H2): teclado para modo debug (tecla M)                          ║
║                                                                              ║
║  PARA EL VIDEO:                                                               ║
║  → Demostrar en vivo: presionar A mientras se gira con ←, mostrar que       ║
║    el CSV registra CMD=1 aunque el volante esté en otra posición             ║
║  → "Esta separación es clave para CIL: la CNN aprende que la MISMA imagen   ║
║    en una intersección tiene respuestas diferentes según el comando"          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

manual_steer = 0.0
nav_cmd      = CMD_STRAIGHT
frame_count  = 0

driver.setCruisingSpeed(CRUISE_SPEED)

while driver.step() != -1:
    frame_count += 1

    # Procesar TODAS las teclas acumuladas en este timestep
    key = keyboard.getKey()
    while key > 0:
        # Flechas: ajuste fino del volante en tiempo real
        if key == Keyboard.LEFT:
            manual_steer = max(-MAX_ANGLE, manual_steer - STEER_STEP)
        elif key == Keyboard.RIGHT:
            manual_steer = min(MAX_ANGLE, manual_steer + STEER_STEP)
        elif key == Keyboard.UP:
            manual_steer *= CENTER_DECAY      # retorno gradual al centro
            if abs(manual_steer) < 0.005:
                manual_steer = 0.0
        elif key == ord(' '):
            manual_steer = 0.0               # centrar volante inmediato

        # A / W / D: fijar comando de navegación (independiente del volante)
        elif key == ord('A') or key == ord('a'):
            nav_cmd = CMD_LEFT
            print("[NAV] IZQUIERDA fijado")
        elif key == ord('W') or key == ord('w'):
            nav_cmd = CMD_STRAIGHT
            print("[NAV] RECTO fijado")
        elif key == ord('D') or key == ord('d'):
            nav_cmd = CMD_RIGHT
            print("[NAV] DERECHA fijado")

        elif key == ord('Q') or key == ord('q'):
            csv_file.close()
            print(f"[CIL-COLLECT] Total: {img_count} imágenes")
            driver.setCruisingSpeed(0)

        key = keyboard.getKey()

    driver.setSteeringAngle(manual_steer)

    # =========================================================================
    """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  🎬  BLOQUE 5 — CAPTURA AUTOMÁTICA DE IMÁGENES Y ESCRITURA CSV          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  QUÉ HACE:                                                               ║
    ║  Cada 5 frames (80 ms): captura la imagen de la cámara, la guarda       ║
    ║  como JPEG y registra en el CSV el path + steering + nav_command.        ║
    ║                                                                          ║
    ║  CONVERSIÓN DE IMAGEN (BGRA → BGR):                                      ║
    ║  La cámara de Webots devuelve bytes en formato BGRA (4 canales).         ║
    ║  1. getImage() → bytes crudos                                            ║
    ║  2. np.frombuffer() → array 1D de uint8                                  ║
    ║  3. reshape(H, W, 4) → array 3D en formato BGRA                         ║
    ║  4. cv2.cvtColor(BGRA→BGR) → eliminar canal Alpha                       ║
    ║  Este patrón exacto lo usamos desde la Act 3.1 y siempre funciona.      ║
    ║                                                                          ║
    ║  CSV FLUSH INMEDIATO:                                                    ║
    ║  csv_file.flush() escribe al disco en cada imagen.                        ║
    ║  Sin flush, Python almacena en buffer y pierde las últimas imágenes      ║
    ║  si Webots se cierra inesperadamente (crash, Ctrl+C).                    ║
    ║                                                                          ║
    ║  JPEG QUALITY = 95:                                                      ║
    ║  Calidad alta para preservar detalles de las líneas del carril.          ║
    ║  Quality=95 da archivos de ~25KB vs ~5KB a quality=75.                  ║
    ║  Para 10k imágenes: ~250MB vs ~50MB — manejable para GitHub.            ║
    ║                                                                          ║
    ║  RUTA RELATIVA EN CSV:                                                   ║
    ║  Se guarda "data/images/img_XXXXXX.jpg" (sin ruta absoluta).             ║
    ║  Al hacer git clone en Colab, la ruta resuelve desde la raíz del repo.  ║
    ║                                                                          ║
    ║  REDONDEO A 5 DECIMALES:                                                 ║
    ║  round(manual_steer, 5) evita valores como 0.04200000000000001           ║
    ║  en el CSV por aritmética flotante. Los modelos son insensibles a        ║
    ║  diferencias de 0.00001 rad (~0.0006°).                                  ║
    ║                                                                          ║
    ║  USADO ANTES EN:                                                          ║
    ║  • Act 3.1 (collect_training_data): mismo patrón getImage + frombuffer  ║
    ║    + reshape + cvtColor — primer uso validado                            ║
    ║  • Act 4.2: mismo patrón de lectura de cámara en el loop principal      ║
    ║                                                                          ║
    ║  PARA EL VIDEO:                                                           ║
    ║  → Mostrar el directorio data/images/ con las imágenes en vivo           ║
    ║  → Abrir el CSV y mostrar algunas filas con sus valores                  ║
    ║  → "En total generamos X,XXX imágenes en Y minutos de conducción"       ║
    ║  → Mostrar distribución de comandos (cuántas de cada tipo)               ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    if frame_count % CAPTURE_EVERY == 0:
        # Capturar imagen (patrón validado desde Act 3.1)
        raw = camera.getImage()
        img = np.frombuffer(raw, np.uint8).reshape((CAM_H, CAM_W, 4))   # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)                       # BGR

        img_name = f"img_{img_count:06d}.jpg"
        img_path = os.path.join(DATA_DIR, img_name)
        cv2.imwrite(img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Ruta relativa para portabilidad entre PCs del equipo y Colab
        csv_writer.writerow([
            f"data/images/{img_name}",
            round(manual_steer, 5),
            nav_cmd
        ])
        csv_file.flush()   # escritura inmediata — protege contra cierres bruscos
        img_count += 1

        # HUD en display Webots (verde=recto, amarillo=giro pendiente)
        bg = 0x004400 if nav_cmd == CMD_STRAIGHT else 0x444400
        display.setColor(bg)
        display.fillRectangle(0, 0, 200, 40)
        display.setColor(0xFFFFFF)
        display.drawText(f"NAV: {CMD_LABEL[nav_cmd]}", 2, 2)
        display.drawText(f"St: {manual_steer:+.3f} rad", 2, 14)
        display.drawText(f"Img: {img_count:6d}", 2, 26)

        if img_count % 200 == 0:
            print(f"[CIL-COLLECT] {img_count:5d} | st={manual_steer:+.3f} | {CMD_LABEL[nav_cmd]}")

csv_file.close()
