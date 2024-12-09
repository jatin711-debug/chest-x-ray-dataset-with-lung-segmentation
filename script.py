import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications import EfficientNetB0 # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, roc_auc_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MultiLabelXRayClassifier:
    def __init__(self, data_path, chunk_size=10000):
        """
        Initialize the classifier with dataset path and chunk size for processing.
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.full_data = pd.read_csv(data_path)
        self.target_columns = self.identify_target_columns()

    def identify_target_columns(self):
        """
        Dynamically identify multi-label columns in the dataset.
        """
        potential_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        existing_columns = [col for col in potential_columns if col in self.full_data.columns]
        logging.info(f"Identified target columns: {existing_columns}")
        return existing_columns

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Load and preprocess a single image.
        """
        try:
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found: {image_path}")
                return None
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            return img_array
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None

    def data_generator(self, batch_size=32):
        """
        Generate data in batches to handle large datasets.
        """
        data_size = len(self.full_data)
        for start in range(0, data_size, batch_size):
            end = min(start + batch_size, data_size)
            batch_data = self.full_data.iloc[start:end]
            images, labels = [], []
            for _, row in batch_data.iterrows():
                img = self.load_and_preprocess_image(row['DicomPath_y'])
                if img is not None:
                    images.append(img)
                    label = row[self.target_columns].fillna(0).tolist()
                    labels.append(label)
            yield np.array(images), np.array(labels)

    def prepare_multi_label_model(self, input_shape, num_classes):
        """
        Create a robust multi-label classification model.
        """
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dropout(0.6),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return model

    def fine_tune_model(self, model, fine_tune_at):
        """
        Fine-tune the model by unfreezing layers.
        """
        model.layers[0].trainable = True
        for layer in model.layers[0].layers[:fine_tune_at]:
            layer.trainable = False
        return model

    def train_and_evaluate(self, epochs=15, validation_split=0.2):
        images, labels = self.prepare_data()

        # Encode labels
        mlb = MultiLabelBinarizer(classes=self.target_columns)
        encoded_labels = mlb.fit_transform(labels)

        # Validate encoded labels
        assert encoded_labels.shape[1] == len(self.target_columns), (
            "Encoded labels do not match the number of target columns. "
            f"Expected {len(self.target_columns)} but got {encoded_labels.shape[1]}"
        )

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, encoded_labels, test_size=validation_split, random_state=42, stratify=encoded_labels
        )

        # Model preparation
        model = self.prepare_multi_label_model(input_shape=(224, 224, 3), num_classes=len(self.target_columns))

        # Callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        # Training
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, callbacks=callbacks)

        # Evaluation
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)

        print("\nClassification Report:")
        print(classification_report(y_val, y_pred_binary, target_names=self.target_columns))

        print("\nHamming Loss:", hamming_loss(y_val, y_pred_binary))
        print("Jaccard Score (Micro):", jaccard_score(y_val, y_pred_binary, average='micro'))
        print("Jaccard Score (Macro):", jaccard_score(y_val, y_pred_binary, average='macro'))

        return model, history, mlb

    def prepare_data(self):
        """
        Prepare data with comprehensive error handling and logging.
        """
        images = []
        labels = []

        for _, row in self.full_data.iterrows():
            # Load image
            img = self.load_and_preprocess_image(row['DicomPath_y'])

            if img is not None:
                images.append(img)
                # Fill NaN with 0 and convert to numeric if necessary
                label = row[self.target_columns].fillna(0).astype(float).tolist()
                labels.append(label)

        # Convert to numpy arrays
        images_array = np.array(images)
        labels_array = np.array(labels)

        # Validate shapes
        assert labels_array.shape[1] == len(self.target_columns), (
            "Mismatch between target columns and labels. "
            f"Expected {len(self.target_columns)} but got {labels_array.shape[1]}"
        )

        return images_array, labels_array
# Main execution
if __name__ == "__main__":
    classifier = MultiLabelXRayClassifier('merged_CXLSeg_data.csv')
    model, history, mlb = classifier.train_and_evaluate(epochs=15)
