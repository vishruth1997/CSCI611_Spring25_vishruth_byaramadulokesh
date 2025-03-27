# Neural Style Transfer – CSCI611 Spring 2025

This project implements Neural Style Transfer using PyTorch. It combines the **content** of one image (`tiger.jpg`) with the **style** of another image (`style.jpg`) to produce a new, stylized image. The implementation is based on the VGG19 model and includes experiments with various hyperparameters to observe their effects on the final image.

## Repository Structure

```
.
├── README.md                        # This file
├── Style Transfer Report.pdf        # Final project report with analysis and visuals
├── Output Images/                   # Folder containing output result images
├── style.jpg                        # Style image
├── tiger.jpg                        # Content image
└── Style_Transfer_Exercise.ipynb    # Main source code (Jupyter Notebook)
```

## Project Description

In this notebook, I:
- Used a pretrained VGG19 model to extract features from content and style images.
- Calculated content and style losses from intermediate layers of the model.
- Iteratively updated a target image to minimize these losses.
- Experimented with various hyperparameter combinations including:
  - `style_weight`
  - `content_weight`
  - number of optimization steps
  - layer-wise style configurations (`default`, `bold`, `fine`)
  - optimizer types (Adam and L-BFGS)

## How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
   ```

2. **Install required libraries**
   ```bash
   pip install torch torchvision matplotlib numpy pillow requests
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook
   ```
   Then open `Style_Transfer_Exercise.ipynb` and run it cell by cell.

4. **Run Experiments**
   Modify parameters inside the notebook to try out different style and content weights, optimizers, and iteration counts.

5. **View Results**
   Generated images are displayed in the notebook and saved into the `Output Images/` folder for future reference.

## Style Transfer Report

The PDF report `Style Transfer Report.pdf` provides:
- A summary of the implementation
- Detailed explanation of the methods used
- Results and visualizations for various hyperparameter experiments
- Observations and conclusions drawn from the outputs

## Output Images

Inside `Output Images/`, you’ll find:
- Stylized image results for various experiment settings
- Files named based on the hyperparameters used (e.g., `result_sw1000000_cw1_s5000_lsdefault.png`)

## Notes

- The notebook will use the GPU automatically if available.
- Input images (`tiger.jpg` and `style.jpg`) must be in the same directory as the notebook.
- You can extend the notebook by adding more styles, running batch experiments, or comparing multiple optimizers.

