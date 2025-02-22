# CNN Classifier for CIFAR-10

## Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The model is trained on the CIFAR-10 dataset, evaluated, and tested for accuracy on unseen data.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. The 10 classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Features
- Utilizes PyTorch for deep learning model implementation.
- Implements a CNN with multiple convolutional and fully connected layers.
- Uses CUDA for GPU acceleration if available.
- Performs data augmentation and normalization.
- Trains the model with Adam optimizer and cross-entropy loss.
- Saves and loads the trained model.
- Evaluates model performance on test data.

## Installation
Ensure you have Python installed along with the required dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

## Implementation Steps

### 1. Check for GPU Availability
The model checks whether CUDA (GPU) is available and switches to GPU if supported.

### 2. Load and Preprocess Data
- The dataset is loaded using `torchvision.datasets.CIFAR10`.
- The images are normalized to improve training performance.
- The dataset is split into training, validation, and test sets.
- Data loaders are created using `torch.utils.data.DataLoader`.

### 3. Visualizing the Dataset
- Random images from the dataset are displayed.
- RGB channel separation and visualization are performed.

### 4. Define the CNN Model
The CNN architecture consists of:
- Three convolutional layers with ReLU activation and max-pooling.
- Fully connected layers with dropout regularization.
- Softmax output for classification.

### 5. Specify Loss Function and Optimizer
- Cross-entropy loss is used for classification.
- The Adam optimizer is employed for weight updates.

### 6. Train the Model
- The model is trained for a specified number of epochs.
- The training and validation losses are computed at each epoch.
- The model parameters are saved when validation loss decreases.

### 7. Load Trained Model
The saved model is loaded for testing.

### 8. Test the Model
- The model is evaluated on the test dataset.
- Accuracy is calculated for each class and overall.

### 9. Visualize Predictions
- Sample test images are displayed along with predicted and true labels.
- Correct predictions are shown in green; incorrect ones in red.

## Running the Code
To train and test the model, run the Jupyter Notebook containing the provided code. The training process will take some time depending on the hardware used.

## Results
- The final model achieves a reasonable accuracy on the test dataset.
- Incorrect predictions highlight areas for potential model improvements.

## Future Improvements
- Experiment with deeper architectures or ResNet-like structures.
- Apply advanced data augmentation techniques.
- Implement learning rate scheduling for better convergence.
- Utilize pre-trained models for transfer learning.
