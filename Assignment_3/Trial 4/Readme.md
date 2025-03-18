# YOLO Small Object Detection using v11n and Mapillary Dataset - 

## Project Overview  
This project focuses on small object detection using the YOLO (You Only Look Once) model. We leverage the YOLO v11n pre-trained model to detect and classify objects in images. The dataset used for this project consists of aerial imagery containing small objects. The project involves training, validating, and testing the YOLO model on this dataset.  

## Training Trial 
- **Trail 4**: In trial 4 we used pretrained YOLO v11n using Ultralytics, It is trained on the Mapillary dataset. The training is done for 100 epochs with a batch size of 16.

## Repository Structure  
The repository is organized as follows:  

### Dataset and Annotations  
- **Maplillary**: 100,000 high-resolution images (52,000 fully annotated, 48,000 partially annotated) from the Mapillary dataset.
- **Mapillary-DataSet-Sample/**: Contains a sample of the Mapillary dataset with annotations.
- **Mapillary-DataSet-Sample/annotations/**: Contains the annotations for the sample dataset.
- **Mapillary-DataSet-Sample/images/**: Contains the images for the sample dataset.

### Model Runs and Outputs  
- **runs/**: Stores execution details and saved model checkpoints from different YOLO training runs.  
- **detections/**: Contains images processed and tested using the trained YOLO model.  

### Files and Scripts  
- **main.py**: The main script to train the YOLO model on the mapillary dataset. 
- **detect.py**: This script allows you to run on a video or image file and save the results in the `runs/detections/` folder.
- **val.py**: This is optional, to validate the model with custom NMS parameters.
- **output.log**: Contains the output logs from the training runs.

## How to Use This Repository  
1. **View the Source Code for training**: Open `main.py`  to review the implementation details.  
2. **Check Model Runs**: Navigate to the `runs/` folder to access logs and saved weights from training.  
3. **Tested Images**: Explore the `runs/detect/` folder to view results of object detection on test images.  
4. **Dataset & Annotations**: Browse the `Mapillary-DataSet-Sample/` folder for the dataset and labeled annotations.  