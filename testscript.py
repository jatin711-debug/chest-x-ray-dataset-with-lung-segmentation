import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import logging
import time

# Import necessary components from the training script
# Assuming script.py is in the same directory or accessible via PYTHONPATH
try:
    from script import (
        Config,
        MedicalImageModel,
        EnsembleMedicalModel,
        MedicalDataset,
        FocalLoss,
        evaluate,
        seed_everything,
        setup_logging,
        is_main_process, # Use this for logging control
    )
except ImportError as e:
    print(f"Error importing from script.py: {e}")
    print("Please ensure script.py is in the same directory or accessible via PYTHONPATH.")
    exit(1)

# --- Main Testing Logic ---
def test_model(args):
    """Loads a trained model and evaluates it on the test set."""

    # Basic logging setup for testing
    logger = setup_logging(args.output_dir, -1) # Use rank -1 for single-process testing

    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return

    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    # Load checkpoint on CPU first to avoid GPU memory issues if model is large
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    # Load config from checkpoint and update with command-line args
    if "config" not in checkpoint:
        logger.error("Config not found in checkpoint. Cannot proceed.")
        return
    loaded_config_dict = checkpoint["config"]
    config = Config(**loaded_config_dict)

    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.csv_file is not None:
        config.csv_file = args.csv_file
    if args.device is not None:
        config.device = args.device
    if args.threshold is not None:
        config.threshold = args.threshold
    # Ensure num_workers is appropriate for testing (can be 0)
    config.num_workers = args.num_workers if args.num_workers is not None else config.num_workers

    seed_everything(config.seed) # Use seed from config for consistency

    logger.info("Loaded configuration from checkpoint:")
    for key, value in config.dict().items():
        logger.info(f"  {key}: {value}")

    device = torch.device(config.device)
    logger.info(f"Using device: {device}")

    # --- Initialize Model ---
    logger.info("Initializing model architecture...")
    if config.is_ensemble:
        model = EnsembleMedicalModel(config)
    else:
        model = MedicalImageModel(config)

    # Load model state dict
    try:
        # Handle potential keys mismatch (e.g., if saved with DDP)
        state_dict = checkpoint["model_state_dict"]
        # Remove `module.` prefix if saved with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        logger.info("Model weights loaded successfully.")
    except KeyError:
        logger.error("Could not find 'model_state_dict' in checkpoint.")
        return
    except Exception as e:
        logger.error(f"Error loading state dict: {e}")
        return

    model.to(device)
    model.eval()

    # --- Prepare Data ---
    csv_path = os.path.join(config.data_dir, config.csv_file)
    logger.info(f"Loading test data from: {csv_path}")
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    standardized_columns = df.columns.tolist()
    logger.info(f"Original columns: {original_columns}")
    logger.info(f"Standardized columns: {standardized_columns}")

    label_cols = MedicalDataset.LABEL_COLS
    img_path_col = "dicompath_y"
    split_col = "split"

    if img_path_col not in df.columns:
        logger.error(f"Image path column '{img_path_col}' not found.")
        return
    if split_col not in df.columns:
        logger.error(f"Split column '{split_col}' not found.")
        return
    if not all(col in df.columns for col in label_cols):
        missing = [col for col in label_cols if col not in df.columns]
        logger.error(f"Missing required label columns: {missing}")
        return

    logger.info("Preprocessing labels (NaN/-1.0 -> 0.0)")
    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0.0).replace(-1.0, 0.0)
        df[col] = df[col].astype(np.float32)

    df_test = df[df[split_col] == "test"].reset_index(drop=True)

    if len(df_test) == 0:
        logger.warning("No samples found in the 'test' split. Exiting.")
        return

    logger.info(f"Found {len(df_test)} samples in the test set.")

    # Use the same validation/test transforms as in training
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    test_transform = A.Compose(
        [
            A.Resize(height=config.img_size, width=config.img_size),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),
        ]
    )

    test_dataset = MedicalDataset(
        df_test, config, transform=test_transform, data_dir=config.data_dir
    )

    # Note: DALI is usually not necessary/overkill for testing unless the test set is massive
    # and inference speed is critical during the testing phase itself.
    # Using standard DataLoader for simplicity here.
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2, # Often use larger batch size for eval
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0, # Match training setup if workers > 0
    )

    # --- Define Criterion ---
    if config.use_focal_loss:
        logger.info(f"Using Focal Loss for evaluation metric calculation (reduction='mean')")
        criterion = FocalLoss(
            alpha=config.focal_loss_alpha,
            gamma=config.focal_loss_gamma,
            reduction="mean", # Ensure reduction is mean for evaluation loss
        ).to(device)
    else:
        logger.info("Using BCEWithLogitsLoss for evaluation metric calculation.")
        criterion = nn.BCEWithLogitsLoss(reduction="mean").to(device)

    # --- Run Evaluation ---
    logger.info(f"--- Starting Final Evaluation on Test Set using Threshold {config.threshold:.2f} ---")
    start_time = time.time()

    # Use the evaluate function from script.py
    (
        test_loss,
        test_precision,
        test_recall,
        test_f1,
        test_auc,
        test_class_f1,
        test_conf_mat,
    ) = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        threshold=config.threshold,
        num_classes=config.num_classes,
        label_cols=label_cols,
        use_amp=config.use_amp, # Use AMP settings from config
        amp_dtype=config.amp_dtype,
    )

    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds.")

    # --- Log Results ---
    logger.info("="*30)
    logger.info(" Final Test Set Results")
    logger.info("="*30)
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Precision (Macro, Thr={config.threshold:.2f}): {test_precision:.4f}")
    logger.info(f"  Recall (Macro, Thr={config.threshold:.2f}): {test_recall:.4f}")
    logger.info(f"  F1-Score (Macro, Thr={config.threshold:.2f}): {test_f1:.4f}")
    logger.info(f"  AUC (Macro): {test_auc:.4f}")
    logger.info(f"  Per-Class F1 (Thr={config.threshold:.2f}):")
    for i, label_name in enumerate(label_cols):
        logger.info(f"    {label_name}: {test_class_f1[i]:.4f}")
    # Format confusion matrix for better readability if needed
    logger.info(f"  Confusion Matrix (Flattened, Thr={config.threshold:.2f}):\n{test_conf_mat}")
    logger.info("="*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Medical Image Classification Model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None, # Default to config value from checkpoint
        help="Directory containing the data and CSV (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None, # Default to config value from checkpoint
        help="Name of the CSV file within data_dir (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None, # Default to config value from checkpoint
        help="Batch size for testing (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Default to config value from checkpoint
        help="Device for testing, e.g., 'cuda' or 'cpu' (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None, # Default to config value from checkpoint
        help="Threshold for classification (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None, # Default to config value from checkpoint
        help="Number of dataloader workers (overrides checkpoint config if set)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output",
        help="Directory to save testing logs",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    test_model(args)
