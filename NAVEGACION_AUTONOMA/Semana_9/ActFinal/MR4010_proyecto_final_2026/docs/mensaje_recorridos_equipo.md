# Mensaje para compañeros — Solicitud de recorridos adicionales

---

Hola equipo,

Para completar el entrenamiento del modelo CIL del proyecto final necesitamos **más datos de conducción manual** en el simulador. Ya tenemos datos de Joel y Alberto (10,883 frames limpios), pero necesitamos reforzar las maniobras de giro para que el modelo aprenda mejor.

## ¿Qué necesitamos exactamente?

Recorridos **en el Mundo 2 de Webots** usando el controlador `collect_cil_data`. Cada sesión de ~15–20 minutos genera suficientes datos.

**Zonas prioritarias (en orden de importancia):**

| Prioridad | Tipo de maniobra | Cuántos frames aprox. |
|---|---|---|
| 🔴 1 | Giros a la **izquierda** en intersecciones | 300–400 frames por sesión |
| 🔴 2 | Giros a la **derecha** en intersecciones | 300–400 frames por sesión |
| 🟡 3 | Carretera recta con curvas suaves | 200–300 frames |

## ¿Cómo hacerlo? (paso a paso)

**1. Abrir el simulador:**
   - Webots → abrir `MR4010_proyecto_final_2026.wbt`
   - En el panel de controladores, seleccionar `collect_cil_data`

**2. Controles del teclado (en la ventana 3D de Webots):**
   ```
   ← / →       : girar el volante (izq / der)
   i / k       : acelerar / frenar
   ─────────────────────────────────────────
   s           : marcar CONTINUE (seguir carretera)
   w           : marcar RECTO (cruzar recto en intersección)
   a           : marcar IZQUIERDA (vas a girar izq)
   d           : marcar DERECHA (vas a girar der)
   ─────────────────────────────────────────
   q           : guardar y salir
   ```

**3. Técnica para giros (importante para la calidad del dato):**
   - Al **acercarte a la intersección**: presiona `a` (o `d`) **al mismo tiempo** que empiezas a girar el volante
   - No presiones la tecla antes de llegar — el modelo aprende la correlación tecla+volante
   - Mantén la tecla presionada durante todo el giro hasta salir de la intersección
   - Al terminar el giro, vuelve a presionar `s` (CONTINUE)

**4. Los datos se guardan automáticamente en:**
   ```
   data/dataset.csv   (se van agregando filas)
   data/images/       (imágenes numeradas)
   ```

## ¿Cuánto tiempo lleva?

Una sesión de 15–20 minutos con el simulador a velocidad normal genera ~1,500–2,000 frames. Con 2–3 sesiones por persona es suficiente.

## ¿Qué NO hacer?

- No correr a velocidad máxima en curvas (el modelo no puede aprender si la imagen está borrosa)
- No presionar `a`/`d` cuando vas en línea recta (solo en intersecciones)
- No cerrar Webots sin presionar `q` primero (se perderían los datos de esa sesión)

## Fecha límite para subir los datos

Por favor envíen el archivo `dataset.csv` y la carpeta `images/` (como ZIP) antes del **viernes**. Los integro al dataset principal y lanzo el entrenamiento el fin de semana.

Si tienen dudas con el simulador, me avisan por WhatsApp.

Gracias!
Joel
