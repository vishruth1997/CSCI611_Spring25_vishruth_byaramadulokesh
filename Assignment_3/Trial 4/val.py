import torch
from ultralytics import YOLO


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
model = YOLO("yolo11n.pt")  # This will download the model weights if not present
print("YOLOv11n loaded successfully!")

# # Load the YOLOv11s model
model = YOLO("/home/ubuntu/code/runs/detect/yolo11n_small_objects3/weights/best.pt")


# Optional: Validate the model with custom NMS parameters
val_results = model.val(
    data="yolo_final_dataset/data.yaml",
    imgsz=1024,
    conf=0.4,                       # Confidence threshold for validation
    iou=0.3,                        # IoU threshold for NMS during validation
)
print("Validation mAP:", val_results.box.map)