from controller import Supervisor
import numpy as np
import cv2
import time

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

ped = robot.getFromDef("PED")
ped_trans = ped.getField("translation")
ped_rot = ped.getField("rotation")

# Let Webots settle
for _ in range(10):
    robot.step(timestep)

# Test 1: North (positive Y)
ped_trans.setSFVec3f([0.0, 5.0, 0.0])
ped_rot.setSFRotation([0, 0, 1, 0])
for _ in range(8):
    robot.step(timestep)
raw = camera.getImage()
img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
cv2.imwrite("test_north.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

# Test 2: South (negative Y)
ped_trans.setSFVec3f([0.0, -5.0, 0.0])
ped_rot.setSFRotation([0, 0, 1, 0])
for _ in range(8):
    robot.step(timestep)
raw = camera.getImage()
img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
cv2.imwrite("test_south.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

# Test 3: East (positive X)
ped_trans.setSFVec3f([5.0, 0.0, 0.0])
ped_rot.setSFRotation([0, 0, 1, 0])
for _ in range(8):
    robot.step(timestep)
raw = camera.getImage()
img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
cv2.imwrite("test_east.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

# Test 4: West (negative X)
ped_trans.setSFVec3f([-5.0, 0.0, 0.0])
ped_rot.setSFRotation([0, 0, 1, 0])
for _ in range(8):
    robot.step(timestep)
raw = camera.getImage()
img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
cv2.imwrite("test_west.png", cv2.cvtColor(img, cv2.COLOR_BGRA2BGR))

print("[TEST] All 4 directions saved successfully!")
