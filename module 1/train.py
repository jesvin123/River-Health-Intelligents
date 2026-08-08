from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')
    results = model.train(
        data='D:/bootcamp/module 1/data.yaml',
        epochs=60,
        imgsz=640,
        batch=8,
        patience=20,
        augment=True,
        device=0
    )

if __name__ == '__main__':
    main()