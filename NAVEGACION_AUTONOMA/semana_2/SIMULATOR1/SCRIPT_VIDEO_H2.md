# Script — Video demostrativo simple_controller_H2.py
# Duración estimada: 3–4 minutos

---

## [0:00 — Introducción] pantalla: Webots en pausa, vehículo visible

"En este video voy a mostrar el controlador de seguimiento de carril
que desarrollé para la actividad 2.1 de Navegación Autónoma.

El objetivo es que un vehículo autónomo siga la línea amarilla central
de un circuito urbano en Webots, a una velocidad mínima de 50 kilómetros por hora,
incluyendo curvas, intersecciones y cruces peatonales."

---

## [0:20 — Pipeline] pantalla: código abierto en el editor, sección del pipeline

"El controlador sigue una secuencia de pasos de visión por computadora.

Primero obtenemos la imagen de la cámara a bordo del vehículo.
La convertimos a escala de grises para simplificar el procesamiento.
Luego aplicamos el algoritmo de Canny para detectar bordes en la imagen.

Después definimos una región de interés con `fillPoly` — un trapecio
que cubre solo el tramo de carretera frente al vehículo,
descartando el cielo y los costados que no aportan información útil.

Con esa imagen filtrada aplicamos la Transformada de Hough probabilística,
`HoughLinesP`, que nos entrega una lista de segmentos de línea detectados.

Finalmente, esa lista entra a un controlador PID
que calcula el ángulo de dirección del vehículo en cada frame."

---

## [1:00 — PID] pantalla: sección del PID en el código

"El setpoint del PID es el centro horizontal de la imagen —
en este caso la mitad del ancho del display de la cámara.

El error se calcula como la distancia entre el centro estimado
de las líneas detectadas y ese setpoint, normalizado entre -1 y 1.

El controlador tiene tres términos:
- Proporcional, que reacciona al error actual
- Integral, que corrige deriva acumulada a lo largo del tiempo
- Derivativo, que amortigua oscilaciones

Y la salida es el ángulo de dirección que se aplica directamente al vehículo."

---

## [1:30 — Inicio de la demo] pantalla: Webots, dar play a la simulación

"Vamos a ver el controlador en acción.
Inicio la simulación a 50 kilómetros por hora."

— [dar play] —

"En la pequeña pantalla del vehículo pueden ver lo que está viendo la cámara,
en blanco y negro.
Las líneas blancas que aparecen encima son las que el sistema detectó
como candidatas para seguir — esas son las que guían el volante."

---

## [1:50 — Recta] pantalla: vehículo en recta, display visible

"En una recta hay pocas líneas y están bien centradas,
así que el vehículo va derecho sin hacer correcciones grandes."

---

## [2:05 — Curva] pantalla: vehículo en curva

"En una curva las líneas se desplazan hacia un lado del display.
El controlador lo detecta y gira el volante para seguirlas.
Cuando vuelven al centro, el vehículo se endereza."

---

## [2:20 — Cruce peatonal] pantalla: vehículo cruzando cebra

"Aquí hay un cruce peatonal con rayas amarillas horizontales.
El sistema las ignora porque son horizontales —
solo considera líneas que vayan más o menos en la dirección del camino.

Por un momento no detecta nada útil,
pero en lugar de girar bruscamente, mantiene el ángulo que traía
hasta que vuelve a ver la línea del carril."

---

## [2:45 — Intersección] pantalla: vehículo en intersección

"En las intersecciones tampoco hay línea amarilla.
Pasa lo mismo: el vehículo sigue recto con el ángulo anterior
y retoma el seguimiento en cuanto aparece la línea de nuevo."

---

## [3:05 — Resultado] pantalla: vehículo completando una vuelta

"El controlador ha estado corriendo por más de diez minutos
sin ningún choque, completando múltiples vueltas al circuito.

Las ganancias del PID utilizadas son:
Kp = 0.28, Ki = 0.01, Kd = 0.01,
con un rate limiter de 0.03 radianes por frame como red de seguridad."

---

## [3:25 — Cierre] pantalla: código o Webots en pausa

"El código completo con comentarios está disponible en el repositorio
del equipo en GitHub.

Gracias."

---

## Notas de grabación

- Resolución recomendada: 1080p
- Captura: OBS o la herramienta de video integrada de Webots (`Tools → Movie Recorder`)
- Mostrar el display del vehículo visible en pantalla durante toda la demo
- No hace falta narrar en tiempo real — puedes grabar la simulación primero
  y agregar voz en off editando el video después
