# Multi-Label Chest X-Ray Classifier

This project provides a robust, end-to-end pipeline for multi-label classification of chest X-ray images using deep learning and transfer learning. The model predicts the presence of multiple thoracic diseases in a single X-ray image, leveraging modern transformer-based architectures.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Testing the Model](#testing-the-model)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- Handles multi-label classification for medical images (14 thoracic disease classes).
- Utilizes Swin Transformer (or configurable backbone) with transfer learning via [timm](https://github.com/huggingface/pytorch-image-models).
- Implements advanced evaluation metrics: Precision, Recall, F1-score, and AUC (macro).
- Includes threshold optimization for best validation precision.
- Robust error handling and logging for data loading and preprocessing.
- Modular, extensible codebase for research and experimentation.

---

## Technologies Used

- **Python 3.8+**
- **PyTorch** for deep learning model implementation.
- **timm** for state-of-the-art vision backbones (Swin Transformer, etc.).
- **torchmetrics** for evaluation metrics.
- **Pandas** and **NumPy** for data manipulation.
- **scikit-learn** (optional, for further analysis).
- **Pillow** for image processing.

---

## Project Structure

```plaintext
.
|-- script.py                # Main implementation file (PyTorch pipeline)
|-- testscript.py            # Script for testing the model
|-- data.csv                 # Input dataset (CSV format)
|-- images/                  # Directory containing chest X-ray images
|-- output/                  # Directory for model checkpoints and logs
|-- README.md                # Documentation
|-- requirements.txt         # Python dependencies
```

---

## Dataset

The project expects a dataset of chest X-ray images annotated with multiple disease labels. Each image is associated with one or more binary labels indicating the presence of specific conditions.

- The CSV file (`data.csv`) should contain:
  - A column `DicomPath_y` specifying the file path to each X-ray image.
  - Columns for each disease label (e.g., `Atelectasis`, `Cardiomegaly`, etc.), with values 0 (absent), 1 (present), or -1/NaN (treated as 0).
  - A `split` column indicating the data split (`train`, `validate`, `test`).

### Example CSV Format

| DicomPath_y        | Atelectasis | Cardiomegaly | Pneumonia | ... | split    |
|--------------------|-------------|--------------|-----------|-----|----------|
| images/img1.png    | 1           | 0            | 0         | ... | train    |
| images/img2.png    | 0           | 1            | 1         | ... | validate |
| images/img3.png    | 0           | 0            | 0         | ... | test     |

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/multi-label-xray-classifier.git
   cd multi-label-xray-classifier
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place your `data.csv` and images in the appropriate directories as described above.

---

## Usage

1. **Configure settings**  
   Edit the `Config` class in `script.py` to adjust model backbone, batch size, learning rate, etc.

2. **Run the training and evaluation pipeline:**
   ```bash
   python script.py
   ```

   - The script will:
     - Load and preprocess the data.
     - Train the model with the specified configuration.
     - Evaluate on validation and test sets.
     - Optimize the classification threshold for best precision.
     - Save the best model checkpoint and log results.

3. **Check the `output/` directory** for saved models and logs.

---

## Model Training and Evaluation

- **Model:** Swin Transformer (default) or configurable backbone from `timm`.
- **Loss:** Binary Cross-Entropy with Logits (multi-label).
- **Metrics:** Precision, Recall, F1-score, AUC (macro, thresholded).
- **Threshold Optimization:** Automatically searches for the best threshold on the validation set.
- **Logging:** Detailed logs for each epoch, including metrics and checkpointing.

---

## Testing the Model

To evaluate a trained model checkpoint on the test set, use the `testscript.py`.

**Command (Bash/Linux/macOS):**

```bash
python d:\personal-projects\HealthCare\DataSets\chest-x-ray-dataset-with-lung-segmentation-1.0.0\testscript.py --checkpoint_path path/to/your/best_model.pth
```

**Command (PowerShell/Windows):**

```powershell
python d:\personal-projects\HealthCare\DataSets\chest-x-ray-dataset-with-lung-segmentation-1.0.0\testscript.py --checkpoint_path path\to\your\best_model.pth
# Or using .\ if in the script's directory:
# python .\testscript.py --checkpoint_path path\to\your\best_model.pth
```

**Explanation:**

*   `--checkpoint_path`: **(Required)** Specify the full path to the saved model checkpoint file (e.g., `output/best_model.pth` or `output\best_model.pth` generated during training). Use the appropriate path separator for your OS (`/` for Linux/macOS, `\` for Windows).

**Optional Arguments:**

You can override settings stored in the checkpoint's configuration by providing additional arguments:

*   `--data_dir`: Path to the directory containing the data and CSV file (e.g., `.`).
*   `--csv_file`: Name of the CSV file (e.g., `data.csv`).
*   `--batch_size`: Batch size for evaluation (larger sizes might speed up testing).
*   `--device`: Device to run evaluation on (e.g., `cuda`, `cpu`).
*   `--threshold`: Classification threshold to use (if you want to test a specific one instead of the one from the config).
*   `--num_workers`: Number of data loading workers.
*   `--output_dir`: Directory to save the test log file (defaults to `test_output`).

**Example with overrides (Bash/Linux/macOS):**

```bash
python d:\personal-projects\HealthCare\DataSets\chest-x-ray-dataset-with-lung-segmentation-1.0.0\testscript.py \
    --checkpoint_path output/best_model.pth \
    --data_dir . \
    --csv_file data.csv \
    --batch_size 32 \
    --device cuda \
    --output_dir test_run_1
```

**Example with overrides (PowerShell/Windows):**

```powershell
# Single line:
python d:\personal-projects\HealthCare\DataSets\chest-x-ray-dataset-with-lung-segmentation-1.0.0\testscript.py --checkpoint_path output\best_model.pth --data_dir . --csv_file data.csv --batch_size 32 --device cuda --output_dir test_run_1

# Multi-line using backticks (`):
python d:\personal-projects\HealthCare\DataSets\chest-x-ray-dataset-with-lung-segmentation-1.0.0\testscript.py `
    --checkpoint_path output\best_model.pth `
    --data_dir . `
    --csv_file data.csv `
    --batch_size 32 `
    --device cuda `
    --output_dir test_run_1
```

The script will load the specified checkpoint, prepare the test dataset based on the configuration (or overrides), run the evaluation, and print the performance metrics (Loss, Precision, Recall, F1-Score, AUC, Per-Class F1, Confusion Matrix) to the console and the log file within the specified output directory.

---

## Results

- **Best validation precision** and corresponding threshold are reported.
- **Final test set metrics** (loss, precision, recall, F1, AUC) are logged.
- Example (replace with your actual results):

  ```
  Best validation precision: 0.82 at threshold 0.45
  Test set results:
    Loss: 0.23
    Precision: 0.81
    Recall: 0.78
    F1-score: 0.79
    AUC: 0.88
  ```

---

## Future Improvements

- Add support for additional or custom backbones (e.g., ConvNeXt, EfficientNet).
- Implement data augmentation and advanced regularization.
- Add per-class threshold optimization.
- Integrate explainability (Grad-CAM, saliency maps).
- Support for DICOM image loading and preprocessing.
- Distributed/multi-GPU training.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) for backbone models.
- Dataset contributors for their invaluable work in medical imaging.
- PyTorch and torchmetrics developers for their open-source tools.

---

