# YOLO Small Object Detection Project - Trail 3

## Project Overview
This project utilizes the **YOLO v8 pre-trained model** to identify and classify objects in images. The dataset used for this trial is **Traffic Sign Detection (traffic-sign-detection-yolov8-10)**. The project includes training, validation, and testing of the YOLO model on this dataset.

## Training Trial
- **Trial 3**: In Trail 3 we trained the model on a different dataset containing only small images for a 40 epochs using the **YOLO v8 pre-trained model**.

## Repository Structure
The repository consists of the following directories and files:

### **Dataset and Annotations**
- **`traffic-sign-detection-yolov8-10/`**: Contains the dataset of traffic signs along with annotations for object detection.

### **Model Runs and Outputs**
- **`runs/`**: Stores all execution details and saved model checkpoints from different runs of the YOLO model.
- **`detections/`**: Stores images that have been processed and tested using the trained YOLO model.

### **Source Code**
- **`Trail_3.ipynb`**: Jupyter Notebook containing the implementation for this training trial.

### **System Files**
- **`.DS_Store`**: A macOS system file (can be ignored).

## How to Use This Repository
1. **View the Source Code**: Open `Trail_3.ipynb` to explore the implementation.
2. **Check Model Runs**: Navigate to the `runs/` folder to view logs and saved weights from various training runs.
3. **Tested Images**: View detected objects in the `detections/` folder.
4. **Dataset & Annotations**: Explore the `traffic-sign-detection-yolov8-10/` folder for the training dataset and labeled annotations.
