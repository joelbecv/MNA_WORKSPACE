from controller import Keyboard
from vehicle import Car, Driver

robot    = Car()
driver   = Driver()
timestep = int(robot.getBasicTimeStep())

# API R2023b: obtener teclado como dispositivo del robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

driver.setCruisingSpeed(20)
steer = 0.0

print("TEST — presiona flechas en la ventana 3D de Webots")
print("Cualquier tecla debe imprimir KEY=<numero> aqui")

while robot.step() != -1:
    key = keyboard.getKey()
    if key != -1:
        print(f"KEY={key}")

    if key == Keyboard.LEFT:
        steer = max(-0.5, steer - 0.1)
        driver.setSteeringAngle(steer)
        print(f"  IZQUIERDA → {steer:.2f} rad")
    elif key == Keyboard.RIGHT:
        steer = min(0.5, steer + 0.1)
        driver.setSteeringAngle(steer)
        print(f"  DERECHA → {steer:.2f} rad")
    elif key == ord('C') or key == ord('c'):
        steer = 0.0
        driver.setSteeringAngle(0.0)
        print("  centrado")
