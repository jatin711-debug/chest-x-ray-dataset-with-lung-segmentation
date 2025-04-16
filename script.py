import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from pydantic import BaseModel
from timm import create_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging
import time

# Configuration - Simplified for Image-only model first
class Config(BaseModel):
    img_backbone: str = "swin_base_patch4_window7_224"  # Changed from swin_large_patch4_window12_384
    num_classes: int = 14
    dropout: float = 0.2
    img_pool: str = "avg"
    use_pretrained: bool = True
    grayscale: bool = False
    img_size: int = 224  # Changed from 384
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8  # Changed from 16
    learning_rate: float = 1e-5
    epochs: int = 10
    threshold: float = 0.5
    data_dir: str = "."
    output_dir: str = "output"

config = Config()

# Ensure output directory exists
os.makedirs(config.output_dir, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Configuration loaded:")
for key, value in config.__dict__.items():
    logger.info(f"  {key}: {value}")

# --- Model Definition (Simplified) ---
class MedicalImageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Using device: {self.device}")

        logger.info(f"Creating image backbone: {config.img_backbone}")
        self.img_model = create_model(
            config.img_backbone,
            pretrained=config.use_pretrained,
            in_chans=3,
            num_classes=0,  # Output features before final classification layer
            global_pool=config.img_pool,  # Use timm's global pooling
        )
        img_feature_dim = self.img_model.num_features
        logger.info(f"Image backbone feature dimension: {img_feature_dim}")

        logger.info(f"Creating classifier with num_classes={config.num_classes}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(img_feature_dim),
            nn.Linear(img_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes),
        )

        self._init_weights()
        logger.info("Model initialized.")

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, batch: Dict):
        x_img = self.img_model(batch["image"])
        logits = self.classifier(x_img)
        return logits

# --- Dataset ---
class MedicalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config: Config, transform=None, data_dir: str = "."):
        self.data = dataframe
        self.config = config
        self.transform = transform
        self.data_dir = data_dir
        self.labels_cols = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        logger.info(f"Dataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        image_path = os.path.join(self.data_dir, row["DicomPath_y"])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            raise FileNotFoundError(f"File not found: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise IOError(f"Error loading image {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        # Apply fillna, replace, and then infer objects to avoid FutureWarning
        labels_series = row[self.labels_cols].fillna(0.0).replace(-1.0, 0.0)
        labels_raw = labels_series.infer_objects(copy=False).values
        labels = torch.tensor(labels_raw, dtype=torch.float32)

        return {"image": image, "label": labels}

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device, threshold, num_classes):
    model.eval()
    total_loss = 0.0

    precision_metric = torchmetrics.Precision(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device)
    recall_metric = torchmetrics.Recall(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device)
    f1_metric = torchmetrics.F1Score(task="multilabel", num_labels=num_classes, average='macro', threshold=threshold).to(device)
    auc_metric = torchmetrics.AUROC(task="multilabel", num_labels=num_classes, average='macro').to(device)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model({"image": images})
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)

            precision_metric.update(probs, labels.int())
            recall_metric.update(probs, labels.int())
            f1_metric.update(probs, labels.int())
            try:
                auc_metric.update(probs, labels.int())
            except ValueError as e:
                logger.warning(f"Could not update AUC: {e}. Skipping AUC update for this batch.")

    avg_loss = total_loss / len(dataloader)
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    try:
        auc = auc_metric.compute().item()
    except ValueError as e:
        logger.error(f"Final AUC calculation failed: {e}")
        auc = 0.0

    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    auc_metric.reset()

    return avg_loss, precision, recall, f1, auc

# --- Training Loop ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_num, total_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model({"image": images})
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 20 == 0 or (i + 1) == num_batches:
            elapsed_time = time.time() - start_time
            avg_loss_so_far = total_loss / (i + 1)
            logger.info(f"  Epoch {epoch_num+1}/{total_epochs} | Batch {i+1}/{num_batches} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss_so_far:.4f} | Time: {elapsed_time:.2f}s")

    avg_epoch_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch_num+1} finished. Average Training Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

# --- Main Function ---
def main():
    logger.info("Starting main function.")
    csv_file = os.path.join(config.data_dir, 'data.csv')

    try:
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)

        label_cols = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices"
        ]

        logger.info("Preprocessing labels (NaN/-1.0 -> 0.0)")
        for col in label_cols:
            if col in df.columns:
                # Ensure column is numeric first, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Apply fillna, replace, and then infer objects
                df[col] = df[col].fillna(0.0).replace(-1.0, 0.0)
                df[col] = df[col].infer_objects(copy=False)
            else:
                logger.warning(f"Label column '{col}' not found in CSV. Skipping.")

        if 'split' not in df.columns:
            logger.error("Column 'split' not found in CSV. Cannot split data.")
            return

        logger.info("Splitting data into train, validation, test sets.")
        df_train = df[df['split'] == 'train'].reset_index(drop=True)
        df_val = df[df['split'] == 'validate'].reset_index(drop=True)
        df_test = df[df['split'] == 'test'].reset_index(drop=True)
        logger.info(f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(df_test)}")

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])

        logger.info("Creating datasets and dataloaders.")
        train_dataset = MedicalDataset(df_train, config, transform=train_transform, data_dir=config.data_dir)
        val_dataset = MedicalDataset(df_val, config, transform=val_test_transform, data_dir=config.data_dir)
        test_dataset = MedicalDataset(df_test, config, transform=val_test_transform, data_dir=config.data_dir)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Initializing model, criterion, and optimizer.")
        model = MedicalImageModel(config).to(config.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

        best_val_precision = 0.0
        best_epoch = -1

        logger.info("Starting training and validation loop...")
        for epoch in range(config.epochs):
            logger.info(f"--- Epoch {epoch+1}/{config.epochs} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.device, epoch, config.epochs)
            val_loss, val_precision, val_recall, val_f1, val_auc = evaluate(model, val_loader, criterion, config.device, config.threshold, config.num_classes)

            logger.info(f"Epoch {epoch+1} Validation Results:")
            logger.info(f"  Loss: {val_loss:.4f}")
            logger.info(f"  Precision (Macro, Thr={config.threshold}): {val_precision:.4f}")
            logger.info(f"  Recall (Macro, Thr={config.threshold}): {val_recall:.4f}")
            logger.info(f"  F1-Score (Macro, Thr={config.threshold}): {val_f1:.4f}")
            logger.info(f"  AUC (Macro): {val_auc:.4f}")

            scheduler.step(val_precision)

            if val_precision > best_val_precision:
                best_val_precision = val_precision
                best_epoch = epoch + 1
                checkpoint_path = os.path.join(config.output_dir, f"best_precision_model_epoch_{best_epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Validation precision improved to {best_val_precision:.4f}. Saved model to {checkpoint_path}")

        logger.info(f"Training finished. Best validation precision ({best_val_precision:.4f}) achieved at epoch {best_epoch}.")

        logger.info("--- Starting Threshold Optimization on Validation Set ---")
        if best_epoch == -1:
            logger.warning("No best model saved during training. Skipping threshold optimization and testing.")
            return

        best_model_path = os.path.join(config.output_dir, f"best_precision_model_epoch_{best_epoch}.pth")
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))

        best_threshold = config.threshold
        best_thr_precision = 0.0

        thresholds_to_try = np.arange(0.1, 1.0, 0.05)
        for thr in thresholds_to_try:
            logger.info(f"Evaluating with threshold: {thr:.2f}")
            _, val_precision, _, _, _ = evaluate(model, val_loader, criterion, config.device, thr, config.num_classes)
            logger.info(f"  Validation Precision: {val_precision:.4f}")
            if val_precision > best_thr_precision:
                best_thr_precision = val_precision
                best_threshold = thr
                logger.info(f"  New best precision found with threshold {best_threshold:.2f}")

        logger.info(f"Optimal threshold found: {best_threshold:.2f} with Validation Precision: {best_thr_precision:.4f}")

        logger.info(f"--- Final Evaluation on Test Set using Threshold {best_threshold:.2f} ---")
        test_loss, test_precision, test_recall, test_f1, test_auc = evaluate(model, test_loader, criterion, config.device, best_threshold, config.num_classes)

        logger.info("Final Test Set Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Precision (Macro, Thr={best_threshold:.2f}): {test_precision:.4f}")
        logger.info(f"  Recall (Macro, Thr={best_threshold:.2f}): {test_recall:.4f}")
        logger.info(f"  F1-Score (Macro, Thr={best_threshold:.2f}): {test_f1:.4f}")
        logger.info(f"  AUC (Macro): {test_auc:.4f}")

        logger.info("Main function finished.")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()