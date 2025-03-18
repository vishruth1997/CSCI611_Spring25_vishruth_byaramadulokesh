import os
import cv2
from ultralytics import YOLO

# Ensure OpenCV runs in headless mode (avoiding Qt errors)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Load the model
model = YOLO("/home/ubuntu/code/runs/detect/yolo11n_small_objects3/weights/best.pt")

# Open video
video_path = "/home/ubuntu/for.MP4"
cap = cv2.VideoCapture(video_path)

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_path = "annotated2.mp4"
out = cv2.VideoWriter(output_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(source=frame, conf=0.25, iou=0.7, classes=None, verbose=False)

    # Process detections
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame to output video
    out.write(frame)

# Cleanup
cap.release()
out.release()

print(f"Processing complete. Output saved as {output_path}")