# Image Processing Using OpenCV

## Overview
This project demonstrates various image processing techniques using OpenCV and NumPy. The Jupyter Notebook covers basic operations like loading an image, converting it to grayscale, applying convolution filters for edge detection, blurring, corner detection, and scaling. The techniques implemented are commonly used in computer vision applications.

## Installation
To run this notebook, you need to install OpenCV and its dependencies.

### Install OpenCV
#### Example on MacOS:
```sh
pip3 --version
pip 24.0 from /opt/anaconda3/lib/python3.12/site-packages/pip (python 3.12)
python3 --version
Python 3.12.4
pip3 install opencv-python
```

### Dependencies
Ensure you have the following Python packages installed:
```sh
pip install numpy matplotlib opencv-python
```

## Features Implemented
The following image processing techniques are covered:

### 1. Importing Resources and Displaying an Image
- Uses Matplotlib and OpenCV to load and display an image.
- Reads an image (`building2.jpg`) and displays it.

### 2. Converting Image to Grayscale
- Converts the original image to grayscale using OpenCV.

### 3. Applying Convolution Filters
- **Edge Detection:**
  - Custom kernels for horizontal, vertical, and diagonal edge detection.
  - Uses `cv2.filter2D()` to apply edge detection filters.

- **Blurring:**
  - Custom kernel filters for 2×2, 3×3, and 5×5 averaging blurring.
  - Applies `cv2.filter2D()` to perform blurring operations.

### 4. Sobel Operator for Edge Detection
- Applies Sobel filters to detect horizontal and vertical edges.

### 5. Corner Detection
- Uses predefined kernels to detect different corners of an image.
- Computes a corner response using convolution.

### 6. Scaling After Blurring
- Downsamples the blurred image using a factor of 4×4.

### 7. Edge Detection After Blurring
- First applies a blurring filter, then uses the Sobel operator for edge detection.

## How to Use
1. **Clone the Repository:**
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. **Run the Jupyter Notebook:**
   ```sh
   jupyter notebook
   ```
3. Open the notebook and execute the cells to see the image processing steps in action.

## TODO
- Implement Sobel Operator-based edge detection.
- Explore different corner detection methods.
- Perform multi-step filtering (e.g., blurring followed by edge detection).
- Experiment with different images for better results.

## Dependencies
- Python 3.12
- OpenCV
- NumPy
- Matplotlib
