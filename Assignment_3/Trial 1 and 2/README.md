# YOLO Small Object Detection Project - Trail 1 and Trail 2

## Project Overview
This project utilizes the **YOLO (You Only Look Once) object detection** model to identify and classify objects in images. We are using the **YOLO v8 pre-trained model** for this task. The dataset used consists of **U.S. road signs**, and the project includes training, validation, and testing of the YOLO model on this dataset.

## Training Trials
Two training trials were conducted:
- **Trial 1**: Trained for **30 epochs**.
- **Trial 2**: Trained for **60 epochs**.

## Repository Structure
The repository consists of the following directories and files:

### **Dataset and Annotations**
- **`US-Road-Signs-58/`**: Contains the dataset of U.S. road signs along with annotations for object detection.

### **Model Runs and Outputs**
- **`runs/`**: Stores all execution details and saved model checkpoints from different runs of the YOLO model.
- **`detections/`**: Stores images that have been processed and tested using the trained YOLO model.

### **Source Code**
- **`Trail_1.ipynb`**: Jupyter Notebook containing the implementation for the first training trial (30 epochs).
- **`Trail_2.ipynb`**: Jupyter Notebook containing the implementation for the second training trial (60 epochs).

### **System Files**
- **`.DS_Store`**: A macOS system file (can be ignored).

## How to Use This Repository
1. **View the Source Code**: Open `Trail_1.ipynb` or `Trail_2.ipynb` to explore the implementation.
2. **Check Model Runs**: Navigate to the `runs/` folder to view logs and saved weights from various training runs.
3. **Tested Images**: View detected objects in the `detections/` folder.
4. **Dataset & Annotations**: Explore the `US-Road-Signs-58/` folder for the training dataset and labeled annotations.
