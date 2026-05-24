from controller import Supervisor
import numpy as np
import cv2
import os

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

# Settle a few frames
for _ in range(10):
    robot.step(timestep)

raw = camera.getImage()
if raw:
    img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("test_camera.png", img_bgr)
    print("[TEST] Saved test_camera.png successfully!")
else:
    print("[TEST] Error: Camera image is empty!")
