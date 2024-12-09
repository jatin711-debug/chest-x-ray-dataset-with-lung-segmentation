# Multi-Label Chest X-Ray Classifier

This project implements a robust pipeline for training a multi-label image classification model on chest X-ray data, using deep learning and transfer learning techniques. The model predicts the presence of multiple conditions in a single X-ray image.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Features

- Handles multi-label classification tasks for medical images.
- Utilizes the EfficientNetB0 architecture with transfer learning.
- Provides detailed evaluation metrics including classification report, Hamming loss, and Jaccard scores.
- Implements advanced callbacks for training optimization:
  - ReduceLROnPlateau
  - EarlyStopping
- Extensive error handling and logging for image loading and preprocessing.

---

## Technologies Used

- **Python**
- **TensorFlow/Keras** for deep learning model implementation.
- **Pandas** for data manipulation.
- **NumPy** for numerical computations.
- **scikit-learn** for data preprocessing and evaluation metrics.
- **EfficientNetB0** as the base model for transfer learning.

---

## Project Structure

```plaintext
.
|-- script.py                # Main implementation file
|-- merged_CXLSeg_data.csv   # Input dataset (CSV format)
|-- images/                  # Directory containing chest X-ray images
|-- README.md                # Documentation
|-- requirements.txt         # Python dependencies
```

---

## Dataset

The project uses a dataset containing chest X-ray images annotated with multiple conditions. Each X-ray image is associated with one or more labels indicating the medical conditions present. The dataset should:

- Contain a CSV file with image paths and labels.
- Include a column named `DicomPath_y` specifying the file paths to the X-ray images.

### Example CSV Format:

| DicomPath_y       | Atelectasis | Cardiomegaly | Pneumonia | ... |
|-------------------|-------------|--------------|-----------|-----|
| path/to/image1.dcm | 1           | 0            | 0         | ... |
| path/to/image2.dcm | 0           | 1            | 1         | ... |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/multi-label-xray-classifier.git
   cd multi-label-xray-classifier
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is available in the correct structure.

---

## Usage

1. Initialize the classifier:
   ```python
   from script import ConsistentMultiLabelXRayClassifier

   classifier = ConsistentMultiLabelXRayClassifier(data_path='merged_CXLSeg_data.csv')
   ```

2. Train and evaluate the model:
   ```python
   model, history, mlb = classifier.train_and_evaluate(epochs=15)
   ```

3. View evaluation metrics in the console output.

---

## Model Training and Evaluation

### Training

The training pipeline:
- Loads and preprocesses images.
- Uses transfer learning with the EfficientNetB0 model.
- Fine-tunes the model with a fully connected head for multi-label classification.

### Evaluation

The evaluation includes:
- Classification report with precision, recall, and F1-score for each label.
- Hamming loss.
- Jaccard scores (micro and macro averages).

---

## Results

- **Accuracy**: Achieved XX% on validation data.
- **Precision/Recall/F1**: Comprehensive metrics for each condition.
- **Hamming Loss**: YY%.
- **Jaccard Score**: ZZ (micro), AA (macro).

---

## Future Improvements

- Add support for additional datasets.
- Implement data augmentation to improve model generalization.
- Experiment with other architectures like EfficientNetV2 or Vision Transformers.
- Optimize preprocessing for large datasets using parallelization.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- TensorFlow/Keras team for providing pre-trained EfficientNet models.
- Dataset contributors for their invaluable work in medical imaging.

---

