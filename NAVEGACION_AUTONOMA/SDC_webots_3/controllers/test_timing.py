"""
Mide cuánto tarda cada operación por frame.
Imprime avisos cuando un frame supera 15ms.
"""
from vehicle import Car, Driver
from controller import Keyboard
import numpy as np
import cv2
import time

robot    = Car()
driver   = Driver()
ts       = int(robot.getBasicTimeStep())

camera   = robot.getDevice("camera")
camera.enable(ts)

lidar    = robot.getDevice("Sick LMS 291")
lidar.enable(ts)

display  = robot.getDevice("display_image")
keyboard = Keyboard()
keyboard.enable(ts)

dw, dh = display.getWidth(), display.getHeight()

YELLOW_LOW  = np.array([15,  80,  80], dtype=np.uint8)
YELLOW_HIGH = np.array([35, 255, 255], dtype=np.uint8)

driver.setCruisingSpeed(20)
frame = 0

while robot.step() != -1:
    frame += 1
    t0 = time.perf_counter()

    # A) Imagen
    raw    = camera.getImage()
    img    = np.frombuffer(raw, np.uint8).reshape(camera.getHeight(), camera.getWidth(), 4)
    bgr    = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(bgr, (dw, dh))
    grey   = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    tA = (time.perf_counter() - t0) * 1000

    # B) LiDAR
    t1 = time.perf_counter()
    ranges = lidar.getRangeImage()
    tB = (time.perf_counter() - t1) * 1000

    # C) OpenCV pipeline
    t2 = time.perf_counter()
    hsv   = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    ymask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    edges = cv2.bitwise_or(cv2.Canny(grey, 50, 150), cv2.Canny(ymask, 50, 150))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=15)
    tC = (time.perf_counter() - t2) * 1000

    # D) Display
    t3 = time.perf_counter()
    if frame % 10 == 0:
        display.setColor(0x000000)
        display.fillRectangle(0, 0, dw, 30)
        display.setColor(0xFFFFFF)
        display.drawText(f"frame {frame}", 2, 2)
    tD = (time.perf_counter() - t3) * 1000

    total = (time.perf_counter() - t0) * 1000

    if total > 15 or frame % 100 == 0:
        print(f"[{frame:05d}] total={total:.1f}ms  img={tA:.1f} lidar={tB:.1f} cv={tC:.1f} disp={tD:.1f}")
