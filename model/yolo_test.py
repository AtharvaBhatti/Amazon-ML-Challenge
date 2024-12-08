from ultralytics import YOLO

model = YOLO("/home/atharva/ML challan/model/runs/detect/train2/weights/last.pt")

results = model("/home/atharva/ML challan/dataset/train/images/41-hwoqYbTL.jpg", save=True)

print(results[0].boxes)