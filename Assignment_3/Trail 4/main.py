import torch
from ultralytics import YOLO


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
model = YOLO("yolo11n.pt")  # This will download the model weights if not present
print("YOLOv11s loaded successfully!")

# # Load the YOLOv11s model
model = YOLO("yolo11n.pt")

# Train the model with valid hyperparameters for small object detection
results = model.train(
    data="yolo_final_dataset/data.yaml",
    epochs=100,
    imgsz=1024,                      # Higher resolution for small objects
    batch=32,                       # Leverage A100's memory
    name="yolo11n_small_objects",
    device=0,
    pretrained=True,
    optimizer="SGD",
    momentum=0.937,
    lr0=0.0005,
    verbose=True,
    plots=True,
    augment=True,
    scale=0.5,
    mosaic=True,
)

# Print training results
print("Training completed!")
print(results)

# Save the trained model
model.save("/home/ubuntu/code/runs/detect/yolo11n_small_objects3/weights/best.pt")

# Optional: Validate the model with custom NMS parameters
val_results = model.val(
    data="yolo_final_dataset/data.yaml",
    imgsz=1024,
    conf=0.4,                       # Confidence threshold for validation
    iou=0.3,                        # IoU threshold for NMS during validation
)
print("Validation mAP:", val_results.box.map)