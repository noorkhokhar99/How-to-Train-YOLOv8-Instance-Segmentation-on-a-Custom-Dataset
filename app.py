from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load an official model

# Predict with the model
results = model(source="demo5.mp4", show=True,save=True)  # predict on an image