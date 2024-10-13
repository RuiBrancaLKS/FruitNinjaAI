from ultralytics import YOLO
import torch

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

    model = YOLO('yolo11n.pt')
    model.train(data=r'datasets\data.yaml', epochs=50, imgsz=640, device=0)
