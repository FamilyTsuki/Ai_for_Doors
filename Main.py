from ultralytics import YOLO

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml", 
        epochs=40,
        imgsz=320,
        device="cpu", 
        batch=12,
        plots=True 
    )
      