import pygetwindow as gw
from PIL import ImageGrab
from ultralytics import YOLO
import torch
import time
import os
import keyboard
import pyautogui

""""""
def capture():
    windows = gw.getWindowsWithTitle('BlueStacks')

    if not windows:
        print('Bluestacks not found!')
        return None
    else:
        bluestacks_window = windows[0]

    left, top = bluestacks_window.topleft
    right, bottom = bluestacks_window.bottomright

    screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
    screenshot.save(f'screenshots/temp/screenshot_{int(time.time())}.png')

    return screenshot, left, top

""""""
def predictYOLO(model, screenshot):
    results = model(screenshot)

    # Extract bounding box information
    boxes = results[0].boxes
    detected_objs = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Extract the coordinates of the bounding box
            label = results[0].names[int(box.cls[0])]  # Get the label of the detected object
            confidence = box.conf[0]  # Get the confidence score
            print(f"Detected {label} with confidence {confidence:.2f} at coordinates: ({x1}, {y1}, {x2}, {y2})")
            detected_objs.append((label, x1, y1, x2, y2))
    
    return detected_objs

""""""
def perform_action(objs, offset_x, offset_y):
    for obj in objs:
        label, x1, y1, x2, y2 = obj
        if label != 'bomb':
            start_x, start_y = x1 + offset_x, y1 + offset_y
            end_x, end_y = x2 + offset_x, y2 + offset_y
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=0.2)

""""""
if __name__ == '__main__':
    time.sleep(1)

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
    
    while True:
        if keyboard.is_pressed('esc'):
            print('ESC pressed, stopping the program ...')
            break
            
        screenshot, offset_x, offset_y = capture()

        if screenshot:
            detected_objs = predictYOLO(model, screenshot)
            perform_action(detected_objs, offset_x, offset_y)
        time.sleep(0.5)