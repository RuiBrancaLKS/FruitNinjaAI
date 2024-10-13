import pygetwindow as gw
from ultralytics import YOLO
import torch
import time
import win32gui
import win32ui
import win32con
import win32api
import numpy as np
import cv2
import os

class FruitNinjaAI:
    def __init__(self, model_path, screen_size, debug=False):
        self.SCREEN_SIZE = screen_size
        self.model = YOLO(model_path)
        self.cuda_available = torch.cuda.is_available()
        self.debug = debug
        self.left = 0
        self.top = 0
        self.right = 0
        self.bottom = 0

        # Create directory for saving predictions if debug mode is enabled
        if self.debug:
            os.makedirs('screenshots/preds', exist_ok=True)

        print("PyTorch version:", torch.__version__)
        print("CUDA available:", self.cuda_available)
        if self.cuda_available:
            print("CUDA version:", torch.version.cuda)
            print("Device count:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("CUDA is not available. Please check your installation.")

    def capture(self, save=False):
        windows = gw.getWindowsWithTitle('BlueStacks')

        if not windows:
            print('Bluestacks not found!')
            return None
        else:
            bluestacks_window = windows[0]

        hwnd = win32gui.FindWindow(None, bluestacks_window.title)
        self.left, self.top = bluestacks_window.topleft
        self.right, self.bottom = bluestacks_window.bottomright

        w = self.right - self.left
        h = self.bottom - self.top

        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (w, h), dcObj, (0, 0), win32con.SRCCOPY)

        # Convert dataBitMap to a numpy array using OpenCV
        bmp_info = dataBitMap.GetInfo()
        bmp_str = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmp_str, dtype='uint8').reshape((bmp_info['bmHeight'], bmp_info['bmWidth'], 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w = img.shape[:2]
        scale = min(self.SCREEN_SIZE / w, self.SCREEN_SIZE / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (resized_w, resized_h))
        img_padded = np.full((self.SCREEN_SIZE, self.SCREEN_SIZE, 3), 0, dtype=np.uint8)
        pad_top = (self.SCREEN_SIZE - resized_h) // 2
        pad_left = (self.SCREEN_SIZE - resized_w) // 2
        img_padded[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = img_resized
        img = img_padded

        if save:
            cv2.imwrite(f'screenshots/training/screenshot_{int(time.time())}.png', img)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        print("Taking screenshot!")
        return img, self.left, self.top

    def train(self):
        print(f"Starting training with dataset: 'datasets\\{self.SCREEN_SIZE}x{self.SCREEN_SIZE}\\data.yaml' and image size: {self.SCREEN_SIZE}")
        self.model.train(data=f'datasets\\{self.SCREEN_SIZE}x{self.SCREEN_SIZE}\\data.yaml', epochs=50, imgsz=self.SCREEN_SIZE, device=0 if self.cuda_available else 'cpu')

    def predict(self, screenshot):
        results = self.model(screenshot)

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

                # Draw bounding boxes if in debug mode
                if self.debug:
                    cv2.rectangle(screenshot, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(screenshot, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save image with predictions if in debug mode
            if self.debug:
                cv2.imwrite(f'screenshots/preds/pred_{int(time.time())}.png', screenshot)

        return detected_objs

    def perform_action(self, obj, offset_x, offset_y):
        label, x1, y1, x2, y2 = obj
        if label != 'Bomb':
            # Calculate the scaling factor based on the original and resized dimensions
            scale_w = (self.right - self.left) / self.SCREEN_SIZE
            scale_h = (self.bottom - self.top) / self.SCREEN_SIZE

            # Scale the coordinates back to the original dimensions
            start_x, start_y = x1 * scale_w + offset_x, y1 * scale_h + offset_y
            end_x, end_y = x2 * scale_w + offset_x, y2 * scale_h + offset_y

            # Number of steps to move the cursor smoothly
            steps = 20
            delta_x = (end_x - start_x) / steps
            delta_y = (end_y - start_y) / steps

            # Move to the starting position and simulate the drag
            win32api.SetCursorPos((int(start_x), int(start_y)))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            for i in range(steps):
                intermediate_x = start_x + delta_x * (i + 1)
                intermediate_y = start_y + delta_y * (i + 1)
                win32api.SetCursorPos((int(intermediate_x), int(intermediate_y)))
                time.sleep(0.01)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)