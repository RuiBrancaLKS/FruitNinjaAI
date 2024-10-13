import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
import torch
import time
import os

def capture(count):
    windows = gw.getWindowsWithTitle('BlueStacks')
    if not windows:
        print('Bluestacks not found!')
        return None
    else:
        bluestacks_window = windows[0]

    left, top = bluestacks_window.topleft
    right, bottom = bluestacks_window.bottomright

    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))

    screenshot.save(f'screenshots/temp/screenshot{count}.png')

    return screenshot


def predictYOLO(model, screenshot):
    results = model(screenshot)
    print(results)
    results[0].show()

    # Extract bounding box information
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract the coordinates of the bounding box
            label = results[0].names[int(box.cls[0])]  # Get the label of the detected object
            confidence = box.conf[0]  # Get the confidence score
            print(f"Detected {label} with confidence {confidence:.2f} at coordinates: ({x1}, {y1}, {x2}, {y2})")
    

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Please check your installation.")

    model = YOLO(r'runs\detect\train\weights\best.pt')
    
    predictYOLO(model, r'screenshots\temp\screenshot4.png')