from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="/home/atharva/ML challan/model/yolo_config.yaml", epochs=10, imgsz=640)