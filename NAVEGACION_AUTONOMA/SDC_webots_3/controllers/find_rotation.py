import subprocess
import time
import os
import cv2
import numpy as np

# List of candidate rotations (axis-angle)
CANDIDATES = [
    "1 0 0 0",
    "1 0 0 1.5708", "1 0 0 -1.5708", "1 0 0 3.1416",
    "0 1 0 1.5708", "0 1 0 -1.5708", "0 1 0 3.1416",
    "0 0 1 1.5708", "0 0 1 -1.5708", "0 0 1 3.1416",
    # Combined axes
    "0.707 0.707 0 1.5708", "0.707 0.707 0 -1.5708", "0.707 0.707 0 3.1416",
    "0.707 0 0.707 1.5708", "0.707 0 0.707 -1.5708", "0.707 0 0.707 3.1416",
    "0 0.707 0.707 1.5708", "0 0.707 0.707 -1.5708", "0 0.707 0.707 3.1416",
    "0.577 0.577 0.577 2.094", "0.577 0.577 0.577 -2.094",
    "-0.577 0.577 0.577 2.094", "-0.577 0.577 0.577 -2.094",
]

WORLD_PATH = "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/SDC_webots_3/worlds/collect_data.wbt"

def update_world_rotation(rot_str):
    with open(WORLD_PATH, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    in_robot = False
    for line in lines:
        if "Robot {" in line:
            in_robot = True
        elif in_robot and "rotation" in line:
            line = f"  rotation {rot_str}\n"
            in_robot = False
        new_lines.append(line)
        
    with open(WORLD_PATH, "w") as f:
        f.writelines(new_lines)

def run_test():
    # Start Webots
    webots_proc = subprocess.Popen(
        ["/Applications/Webots.app/Contents/MacOS/webots", WORLD_PATH],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(5)
    
    # Run test_camera.py
    env = os.environ.copy()
    env["WEBOTS_HOME"] = "/Applications/Webots.app/Contents"
    env["DYLD_LIBRARY_PATH"] = "/Applications/Webots.app/Contents/lib/controller"
    env["PYTHONPATH"] = "/Applications/Webots.app/Contents/lib/controller/python"
    
    try:
        res = subprocess.run(
            ["/Users/joelbecerril/miniconda3/bin/python3", "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/SDC_webots_3/controllers/test_camera.py"],
            env=env, capture_output=True, text=True, timeout=10
        )
        print(res.stdout, res.stderr)
    except Exception as e:
        print("Timeout or error running script:", e)
        
    # Terminate Webots
    webots_proc.terminate()
    webots_proc.wait()

def analyze_image():
    img_path = "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/SDC_webots_3/test_camera.png"
    if not os.path.exists(img_path):
        return False, "Image file not found"
        
    img = cv2.imread(img_path)
    if img is None:
        return False, "Failed to read image"
        
    # Check shape
    h, w, c = img.shape
    
    # Sample top center and bottom center
    top_pixel = img[10, w//2]
    bot_pixel = img[h - 10, w//2]
    
    # BGR format
    # Sky detection: Blue component is significantly larger than Red and Green
    is_sky_top = (top_pixel[0] > 100) and (top_pixel[0] > int(top_pixel[1]) + 20) and (top_pixel[0] > int(top_pixel[2]) + 20)
    
    # Ground detection: Grey color, B, G, R are close to each other
    is_ground_bot = (abs(int(bot_pixel[0]) - int(bot_pixel[1])) < 15) and \
                    (abs(int(bot_pixel[1]) - int(bot_pixel[2])) < 15) and \
                    (50 < bot_pixel[0] < 150)
                    
    print(f"Top Pixel: {top_pixel} (Sky: {is_sky_top}) | Bot Pixel: {bot_pixel} (Ground: {is_ground_bot})")
    
    if is_sky_top and is_ground_bot:
        return True, "Upright and correct!"
    return False, "Incorrect orientation"

for rot in CANDIDATES:
    print(f"\n--- Testing Rotation: {rot} ---")
    update_world_rotation(rot)
    
    # Clean previous image
    img_path = "/Users/joelbecerril/MNA_WORKSPACE/NAVEGACION_AUTONOMA/SDC_webots_3/test_camera.png"
    if os.path.exists(img_path):
        os.remove(img_path)
        
    run_test()
    success, msg = analyze_image()
    print(f"Result: {msg}")
    if success:
        print(f"\n[SUCCESS] Found correct rotation: {rot}")
        break
