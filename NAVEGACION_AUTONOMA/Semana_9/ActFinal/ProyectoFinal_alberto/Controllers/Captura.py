from controller import Keyboard, Display
from vehicle import Car, Driver
import os
import csv
from datetime import datetime
import numpy as np
import cv2

# =========================
# Configuración
# =========================
SAVE_DIR = r"C:\Users\betoa\Documents\TEC\NavegacionAutonoma\Modulo9\MR4010_proyecto_final_2026\MR4010_proyecto_final_2026\Capturas"
CSV_FILE = os.path.join(SAVE_DIR, "dataset_mundo1.csv")

SPEED = 25
MAX_ANGLE = 0.50
ANGLE_INCR = 0.01
STEERING_DECAY = 0.98
CAPTURE_EVERY_MS = 500

# =========================
# Utilidades
# =========================
def create_dataset_folder():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path",
                "command",
                "steering_angle",
                "speed",
                "timestamp",
                "simulation_time"
            ])

def get_camera_image(camera):
    raw_image = camera.getImage()
    image_bgra = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image_bgra

def show_camera_on_display(display, image_bgra):
    if display is None:
        return

    image_rgb = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)

    display_width = display.getWidth()
    display_height = display.getHeight()

    image_rgb = cv2.resize(image_rgb, (display_width, display_height))

    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=display_width,
        height=display_height
    )

    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)

def save_sample(camera, command, steering_angle, speed, sim_time):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_name = f"mundo1_{timestamp}.png"
    image_path = os.path.join(SAVE_DIR, image_name)

    camera.saveImage(image_path, 100)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            image_path,
            command,
            steering_angle,
            speed,
            timestamp,
            sim_time
        ])

    print(
        f"[CAPTURA] {image_name} | cmd={command} | angle={steering_angle:.3f}",
        flush=True
    )

# =========================
# Main
# =========================
def main():
    create_dataset_folder()

    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display = robot.getDevice("display")
    if display is None:
        print("ADVERTENCIA: No se encontró display.", flush=True)
    else:
        print("Display encontrado correctamente.", flush=True)

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    steering_angle = 0.0
    command = "straight"
    last_capture_time = 0.0

    print("Controlador de captura automática iniciado.", flush=True)
    print("FLECHAS IZQ/DER: controlar dirección", flush=True)
    print("SPACE: centrar volante", flush=True)
    print("A: comando LEFT", flush=True)
    print("S: comando STRAIGHT", flush=True)
    print("D: comando RIGHT", flush=True)
    print(f"Guardando imágenes en: {SAVE_DIR}", flush=True)

    while robot.step() != -1:
        image_bgra = get_camera_image(camera)
        show_camera_on_display(display, image_bgra)

        key = keyboard.getKey()
        steering_key_pressed = False

        while key != -1:

            # Control de dirección
            if key == keyboard.LEFT:
                steering_angle -= ANGLE_INCR
                steering_key_pressed = True

            elif key == keyboard.RIGHT:
                steering_angle += ANGLE_INCR
                steering_key_pressed = True

            elif key == ord(" "):
                steering_angle = 0.0
                steering_key_pressed = True

            # Comandos CIL
            elif key in [ord("A"), ord("a")]:
                command = "left"
                print("[COMANDO] left", flush=True)

            elif key in [ord("S"), ord("s")]:
                command = "straight"
                print("[COMANDO] straight", flush=True)

            elif key in [ord("D"), ord("d")]:
                command = "right"
                print("[COMANDO] right", flush=True)

            key = keyboard.getKey()

        if not steering_key_pressed:
            steering_angle *= STEERING_DECAY

        steering_angle = max(-MAX_ANGLE, min(MAX_ANGLE, steering_angle))

        driver.setCruisingSpeed(SPEED)
        driver.setSteeringAngle(steering_angle)

        sim_time_ms = robot.getTime() * 1000

        if sim_time_ms - last_capture_time >= CAPTURE_EVERY_MS:
            save_sample(
                camera=camera,
                command=command,
                steering_angle=steering_angle,
                speed=SPEED,
                sim_time=robot.getTime()
            )
            last_capture_time = sim_time_ms

if __name__ == "__main__":
    main()