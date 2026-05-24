"""
Test mínimo: carro avanza recto, sin sensores, sin display, sin procesamiento.
Si Webots se congela aquí → el problema es la física del BMW.
Si NO se congela → el problema está en el procesamiento de sensores del controlador real.
"""
from vehicle import Car, Driver

robot  = Car()
driver = Driver()
ts     = int(robot.getBasicTimeStep())

driver.setCruisingSpeed(20)
driver.setSteeringAngle(0.0)

print("Test mínimo corriendo — solo movimiento, sin sensores")
while robot.step() != -1:
    pass
