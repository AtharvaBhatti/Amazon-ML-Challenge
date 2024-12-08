from ultralytics import YOLO


# Load a pretrained model
model = YOLO("/home/atharva/ML challan/modelV2/best.pt")  # Load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="/home/atharva/ML challan/modelV2/custom.yaml",  # Path to your updated dataset configuration file
    epochs=100,  # Number of epochs
    imgsz=640,
    batch=4,
    patience=30,
    amp = True
)
