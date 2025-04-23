import os
import random
from typing import Dict, Tuple, Optional, List, Union
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pydantic import BaseModel, Field, validator, ConfigDict
from timm import create_model
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
import logging
import time
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
import math
from sklearn.metrics import confusion_matrix

# Attempt DALI import, handle if not installed
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    print(
        "Warning: NVIDIA DALI not found. Falling back to standard PyTorch DataLoader. Install with 'pip install nvidia-dali-cudaXXX' (e.g., nvidia-dali-cuda118)."
    )


# Configuration
class Config(BaseModel):
    # Model Architecture
    img_backbone: Union[
        str, List[str]
    ] = "swin_base_patch4_window7_224"  # Can be single string or list for ensemble
    ensemble_weights: Optional[
        List[float]
    ] = None  # Weights for ensemble models, must sum to 1
    num_classes: int = 14
    dropout: float = 0.2
    img_pool: str = "avg"
    use_pretrained: bool = True
    grayscale: bool = False  # Note: DALI pipeline currently assumes RGB
    img_size: int = 224
    use_gradient_checkpointing: bool = False  # Enable gradient checkpointing

    # Training Hyperparameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    learning_rate: float = 1e-5
    classifier_lr_multiplier: float = 10.0
    weight_decay: float = 0.01
    epochs: int = 2 # Total epochs for training
    freeze_backbone_epochs: int = 0
    threshold: float = 0.5  # Default threshold, optimized later
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 0  # Steps for LR warmup

    # Learning Rate Strategies
    use_layerwise_lr_decay: bool = False
    layerwise_lr_decay_rate: float = 0.9  # Multiplicative decay factor per layer/block group

    # Loss Function
    use_focal_loss: bool = False
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_ohem: bool = False  # Online Hard Example Mining
    ohem_fraction: float = 0.25  # Fraction of hardest examples to keep per batch

    # Data & Paths
    data_dir: str = "."
    output_dir: str = "output"
    num_workers: int = 4
    seed: int = 42
    csv_file: str = "data.csv"  # Added CSV file name to config

    # Performance Optimizations
    precision: str = "fp16"  # "fp32", "fp16", "bf16"
    use_dali: bool = False  # Enable NVIDIA DALI for data loading

    # Distributed Training
    local_rank: int = -1

    # Internal state, not set by user directly
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float32
    is_ensemble: bool = False

    # Pydantic v2 configuration to allow arbitrary types like torch.dtype
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator("precision")
    def validate_precision(cls, v):
        if v not in ["fp32", "fp16", "bf16"]:
            raise ValueError(
                f"Invalid precision: {v}. Choose from 'fp32', 'fp16', 'bf16'."
            )
        return v

    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v, values):
        if isinstance(values.get("img_backbone"), list):
            if v is None:
                # Default to equal weights if not provided
                num_models = len(values["img_backbone"])
                return [1.0 / num_models] * num_models
            elif len(v) != len(values["img_backbone"]):
                raise ValueError(
                    "Length of ensemble_weights must match the number of img_backbones."
                )
            elif not math.isclose(sum(v), 1.0, abs_tol=1e-6):
                raise ValueError("Ensemble weights must sum to 1.0.")
        elif v is not None:
            raise ValueError(
                "ensemble_weights should only be provided when img_backbone is a list."
            )
        return v

    def __init__(self, **data):
        super().__init__(**data)
        # Post-initialization logic
        self.is_ensemble = isinstance(self.img_backbone, list)
        if self.precision == "bf16" and not (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ):
            print(
                "Warning: BF16 precision requested but not supported. Falling back to FP32."
            )
            self.precision = "fp32"
        self.use_amp = self.precision in ["fp16", "bf16"]
        self.amp_dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16

        if self.use_dali and not DALI_AVAILABLE:
            print(
                "Warning: use_dali=True but DALI is not available. Falling back to PyTorch DataLoader."
            )
            self.use_dali = False


# Initialize config with defaults first
config = Config()

# === Logging Setup ===
def setup_logging(output_dir: str, local_rank: int):
    """Sets up logging for the training process."""
    log_level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],  # Log to console
    )
    logger = logging.getLogger(__name__)
    if local_rank in [-1, 0]:  # Only main process writes to file
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


# === Distributed Training Setup ===
def setup_ddp(local_rank: int):
    """Initializes the distributed process group."""
    if local_rank != -1:
        if not dist.is_available():
            raise RuntimeError("Distributed training requested but not available.")
        if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
            raise RuntimeError(
                f"CUDA not available or local_rank {local_rank} invalid."
            )

        # Initialize process group
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        print(
            f"DDP Setup: Rank {dist.get_rank()}/{dist.get_world_size()} on device cuda:{local_rank}"
        )
    else:
        print("DDP Setup: Not using distributed training.")


def cleanup_ddp():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("DDP Cleanup: Destroyed process group.")


def is_main_process(local_rank: int = -1):
    """Checks if the current process is the main process (rank 0 or non-distributed)."""
    if local_rank == -1:  # Not distributed
        return True
    return dist.get_rank() == 0


# Initialize logger after potential DDP setup might modify rank
logger = setup_logging(config.output_dir, config.local_rank)


# === Seeding ===
def seed_everything(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
        # The following two lines are important for reproducibility with CUDA
        # but might impact performance. Use with caution.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


seed_everything(config.seed)


logger.info("Configuration loaded:")
for key, value in config.__dict__.items():
    logger.info(f"  {key}: {value}")


# === Loss Function ===
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
        Implementation adapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label, reduction=None):
        """
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        """
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma)
        log_probs = F.logsigmoid(logits)  # More stable than log(probs)
        log_loss = label * log_probs + (1 - label) * F.logsigmoid(
            -logits
        )  # Equivalent to BCEWithLogitsLoss

        # Apply alpha weighting (optional)
        if self.alpha is not None:
            alpha_factor = torch.where(label == 1, self.alpha, 1.0 - self.alpha)
            log_loss = alpha_factor * log_loss

        loss = coeff * log_loss

        # Apply reduction
        reduction_mode = reduction if reduction is not None else self.reduction
        if reduction_mode == "mean":
            loss = loss.mean()
        elif reduction_mode == "sum":
            loss = loss.sum()
        # If 'none', return per-element loss

        return -loss  # Return negative loss as we multiplied by log_loss which is negative


# === Single Backbone Model ===
class MedicalImageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Initializing Single Backbone Model on device: {self.device}")

        if isinstance(config.img_backbone, list):
            raise ValueError(
                "MedicalImageModel expects a single backbone string, not a list."
            )

        logger.info(f"Creating backbone: {config.img_backbone}")
        self.img_model = create_model(
            config.img_backbone,
            pretrained=config.use_pretrained,
            in_chans=1 if config.grayscale else 3,
            num_classes=0,  # Remove classifier head
            global_pool=config.img_pool,
        )
        self.feature_dim = self.img_model.num_features

        if config.use_gradient_checkpointing:
            try:
                self.img_model.forward_features = torch.utils.checkpoint.checkpoint_wrapper(
                    self.img_model.forward_features, use_reentrant=False
                )
                logger.info(
                    f"Gradient checkpointing enabled for backbone: {config.img_backbone}"
                )
            except AttributeError:
                logger.warning(
                    f"Could not automatically wrap forward_features for gradient checkpointing on {config.img_backbone}."
                )

        logger.info(f"Using feature dimension: {self.feature_dim}")
        logger.info(f"Creating classifier with num_classes={config.num_classes}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes),
        )

        self._init_weights()  # Initialize classifier weights
        logger.info("Single Backbone Model initialized.")

    def _init_weights(self):
        """Initializes weights for the classifier head."""
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

    def forward(self, image: torch.Tensor):  # Accept tensor directly
        x_img = self.img_model(image)
        logits = self.classifier(x_img)
        return logits

    def freeze_backbone(self):
        """Freezes the weights of the image backbone."""
        logger.info("Freezing backbone weights.")
        for param in self.img_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the weights of the image backbone."""
        logger.info("Unfreezing backbone weights.")
        for param in self.img_model.parameters():
            param.requires_grad = True


class EnsembleMedicalModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Initializing Ensemble Model on device: {self.device}")

        if not isinstance(config.img_backbone, list):
            raise ValueError(
                "EnsembleMedicalModel expects a list of backbone strings."
            )
        if not config.ensemble_weights or len(config.ensemble_weights) != len(
            config.img_backbone
        ):
            raise ValueError(
                "Ensemble weights must be provided and match the number of backbones."
            )

        self.backbones = nn.ModuleList()
        feature_dims = []
        for i, backbone_name in enumerate(config.img_backbone):
            logger.info(
                f"Creating backbone {i+1}/{len(config.img_backbone)}: {backbone_name}"
            )
            backbone = create_model(
                backbone_name,
                pretrained=config.use_pretrained,
                in_chans=1 if config.grayscale else 3,
                num_classes=0,  # Remove classifier head
                global_pool=config.img_pool,
            )
            if config.use_gradient_checkpointing:
                try:
                    # Note: Checkpointing might behave differently with ModuleList, test carefully
                    backbone.forward_features = torch.utils.checkpoint.checkpoint_wrapper(
                        backbone.forward_features, use_reentrant=False
                    )
                    logger.info(
                        f"Gradient checkpointing enabled for backbone: {backbone_name}"
                    )
                except AttributeError:
                    logger.warning(
                        f"Could not automatically wrap forward_features for gradient checkpointing on {backbone_name}."
                    )

            self.backbones.append(backbone)
            feature_dims.append(backbone.num_features)

        # Simple approach: Assume all backbones output same feature dim or use first one's dim
        # More complex: Add adapter layers if dims differ before classifier
        self.feature_dim = feature_dims[0]
        if not all(d == self.feature_dim for d in feature_dims):
            logger.warning(
                f"Ensemble backbones have different feature dimensions: {feature_dims}. Using first dim ({self.feature_dim}) for classifier. Consider adding adapter layers."
            )
            # Example: Add linear layers here to project all features to self.feature_dim

        logger.info(f"Using feature dimension: {self.feature_dim}")
        logger.info(f"Creating shared classifier with num_classes={config.num_classes}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.num_classes),
        )

        # Ensure weights are on the correct device during initialization
        self.ensemble_weights = torch.tensor(config.ensemble_weights, dtype=torch.float32).view(
            -1, 1
        )
        logger.info(
            f"Ensemble weights (before moving to device): {self.ensemble_weights.cpu().numpy().flatten()}"
        )

        self._init_weights()  # Initialize classifier weights
        logger.info("Ensemble Model initialized.")

    def to(self, *args, **kwargs):
        """Override .to() to ensure weights tensor is moved."""
        super().to(*args, **kwargs)
        # Determine target device from arguments
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.ensemble_weights = self.ensemble_weights.to(device)
            logger.info(f"Moved ensemble_weights to device: {self.ensemble_weights.device}")
        return self

    def _init_weights(self):
        # ... (same as in MedicalImageModel) ...
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

    def forward(self, image: torch.Tensor):  # Accept tensor directly
        all_logits = []
        image_input = image
        # Ensure weights are on the same device as input
        if self.ensemble_weights.device != image_input.device:
            self.ensemble_weights = self.ensemble_weights.to(image_input.device)

        for backbone in self.backbones:
            features = backbone(image_input)
            # If feature dims differ and adapters were added, apply them here
            logits = self.classifier(features)
            all_logits.append(logits)

        # Weighted average of logits
        stacked_logits = torch.stack(all_logits, dim=0)  # Shape: [num_models, batch_size, num_classes]
        weighted_logits = (stacked_logits * self.ensemble_weights.unsqueeze(1)).sum(
            dim=0
        )  # Shape: [batch_size, num_classes]

        return weighted_logits

    def freeze_backbone(self):
        logger.info("Freezing all backbone weights.")
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        logger.info("Unfreezing all backbone weights.")
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = True


# === Standard PyTorch Dataset ===
class MedicalDataset(Dataset):
    """Standard PyTorch Dataset for loading medical images."""

    # Define standard label columns expected in the CSV
    # Updated to match the standardized columns from the log
    LABEL_COLS = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "enlarged_cardiomediastinum",
        "fracture",
        "lung_lesion",
        "lung_opacity",
        "no_finding",
        "pleural_effusion",
        "pleural_other",
        "pneumonia",
        "pneumothorax",
        "support_devices",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        transform: Optional[A.Compose] = None,
        data_dir: str = ".",
    ):
        self.df = df
        self.config = config
        self.transform = transform
        self.data_dir = data_dir
        self.image_paths = self.df["dicompath_y"].tolist()  # Assuming this column holds relative paths
        self.labels = self.df[self.LABEL_COLS].values.astype(np.float32)

        # Verify all label columns exist
        if not all(col in df.columns for col in self.LABEL_COLS):
            missing = [col for col in self.LABEL_COLS if col not in df.columns]
            raise ValueError(f"Missing required label columns in DataFrame: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path_relative = self.image_paths[idx]
        img_path_full = os.path.join(self.data_dir, img_path_relative)

        try:
            # Load image using PIL
            img = Image.open(img_path_full)
            if self.config.grayscale:
                img = img.convert("L")  # Convert to grayscale if needed
            else:
                img = img.convert("RGB")  # Ensure 3 channels

            img_np = np.array(img)

            # Apply transformations if provided
            if self.transform:
                augmented = self.transform(image=img_np)
                img_tensor = augmented["image"]
            else:
                # Basic conversion if no augmentation
                img_tensor = (
                    torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                )  # HWC -> CHW

            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

            return {"image": img_tensor, "label": label_tensor}

        except FileNotFoundError:
            logger.error(f"Image file not found: {img_path_full} at index {idx}")
            # Return a placeholder or skip? For now, re-raise or return None/handle in collate_fn
            # Returning None might require a custom collate_fn
            # Let's try returning zero tensors as placeholders
            logger.warning(f"Returning zero tensors for missing image at index {idx}")
            img_channels = 1 if self.config.grayscale else 3
            img_tensor = torch.zeros(
                (img_channels, self.config.img_size, self.config.img_size),
                dtype=torch.float32,
            )
            label_tensor = torch.zeros(self.config.num_classes, dtype=torch.float32)
            return {"image": img_tensor, "label": label_tensor}
        except UnidentifiedImageError:
            logger.error(
                f"Could not read image file (corrupted?): {img_path_full} at index {idx}"
            )
            logger.warning(f"Returning zero tensors for corrupted image at index {idx}")
            img_channels = 1 if self.config.grayscale else 3
            img_tensor = torch.zeros(
                (img_channels, self.config.img_size, self.config.img_size),
                dtype=torch.float32,
            )
            label_tensor = torch.zeros(self.config.num_classes, dtype=torch.float32)
            return {"image": img_tensor, "label": label_tensor}
        except Exception as e:
            logger.error(
                f"Error loading image {img_path_full} at index {idx}: {e}",
                exc_info=True,
            )
            logger.warning(f"Returning zero tensors due to error at index {idx}")
            img_channels = 1 if self.config.grayscale else 3
            img_tensor = torch.zeros(
                (img_channels, self.config.img_size, self.config.img_size),
                dtype=torch.float32,
            )
            label_tensor = torch.zeros(self.config.num_classes, dtype=torch.float32)
            return {"image": img_tensor, "label": label_tensor}


# === DALI Data Loading ===
def create_dali_pipeline(
    batch_size, num_threads, device_id, seed, data_paths, labels, img_size, is_training, grayscale=False
):
    """Creates a DALI pipeline for image loading and augmentation."""
    output_type = types.GRAY if grayscale else types.RGB
    num_channels = 1 if grayscale else 3
    # Adjust mean/std for grayscale if needed (using ImageNet mean for single channel)
    img_mean = [0.485 * 255] if grayscale else [0.485 * 255, 0.456 * 255, 0.406 * 255]
    img_std = [0.229 * 255] if grayscale else [0.229 * 255, 0.224 * 255, 0.225 * 255]
    output_layout = "CHW"

    pipeline = Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed + device_id,
        # prefetch_queue_depth=2 # Optional: Can sometimes improve performance
    )
    with pipeline:
        # Use external source for file paths and labels
        image_files = fn.external_source(
            source=data_paths, name="INPUT_FILES", cycle="quiet", device="cpu"
        )
        dali_labels = fn.external_source(
            source=labels,
            name="INPUT_LABELS",
            cycle="quiet",
            device="cpu",
            batch=False, # Process labels per sample
        )

        # Decode images
        # Use 'mixed' device for potential HW acceleration if available
        images = fn.decoders.image(
            image_files, device="mixed", output_type=output_type
        )

        # Resize
        # Use INTERP_LINEAR for resizing, common practice
        images = fn.resize(images, device="gpu", size=[img_size, img_size], interp_type=types.INTERP_LINEAR)

        # Augmentations (on GPU)
        if is_training:
            # Geometric augmentations
            images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
            images = fn.rotate(
                images, angle=fn.random.uniform(range=(-15, 15)), fill_value=0, interp_type=types.INTERP_LINEAR
            )
            # Brightness/Contrast adjustments (only if not grayscale)
            if not grayscale:
                images = fn.brightness_contrast(
                    images,
                    brightness=fn.random.uniform(range=(0.9, 1.1)),
                    contrast=fn.random.uniform(range=(0.9, 1.1))
                )
            # Note: DALI doesn't have direct equivalents for ElasticTransform, GridDistortion, CoarseDropout easily.
            # More complex augmentations might need custom operators or fallback to CPU Albumentations before DALI.

        # Normalize and transpose NCHW
        images = fn.crop_mirror_normalize(
            images.gpu(), # Ensure data is on GPU before this op
            dtype=types.FLOAT,
            mean=img_mean,
            std=img_std,
            output_layout=output_layout,
        )

        pipeline.set_outputs(images, dali_labels.gpu()) # Move labels to GPU as well
    return pipeline


class DALIDataLoader:
    """Wraps the DALI pipeline in a PyTorch-compatible iterator."""
    def __init__(self, df: pd.DataFrame, config: Config, is_training: bool, data_dir: str):
        self.batch_size = config.batch_size
        self.num_threads = config.num_workers
        self.device_id = config.local_rank if config.local_rank != -1 else 0 # DALI needs a specific GPU ID
        self.data_dir = data_dir
        self.is_training = is_training
        self.num_classes = config.num_classes
        self.img_size = config.img_size
        self.grayscale = config.grayscale

        # Ensure the correct column name is used for file paths
        path_col = "dicompath_y" # Standardized column name
        if path_col not in df.columns:
            raise ValueError(
                f"Column '{path_col}' not found in DataFrame for DALI file paths."
            )
        # Create full paths
        self.file_paths = [
            os.path.join(self.data_dir, row[path_col]) for _, row in df.iterrows()
        ]
        # Extract labels
        label_cols = MedicalDataset.LABEL_COLS
        if not all(col in df.columns for col in label_cols):
             missing = [col for col in label_cols if col not in df.columns]
             raise ValueError(f"Missing required label columns in DataFrame for DALI: {missing}")
        self.labels = df[label_cols].values.astype(np.float32)

        if not self.file_paths or not len(self.labels):
             raise ValueError("DALI DataLoader: file_paths or labels list is empty.")

        # Create the DALI pipeline
        self.pipeline = create_dali_pipeline(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            seed=config.seed,
            data_paths=self.file_paths,
            labels=self.labels,
            img_size=self.img_size,
            is_training=self.is_training,
            grayscale=self.grayscale
        )

        # Determine the policy for the last batch
        last_batch_policy = LastBatchPolicy.DROP if is_training else LastBatchPolicy.PARTIAL
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.epoch_size = math.ceil(len(self.file_paths) / world_size)

        # Build the iterator
        self.dali_iterator = DALIGenericIterator(
            pipelines=[self.pipeline],
            output_map=["image", "label"], # Match pipeline's set_outputs order
            size=self.epoch_size, # Number of samples in the epoch for this rank
            reader_name="Reader", # Optional name
            last_batch_policy=last_batch_policy,
            auto_reset=True # Automatically reset iterator at the end of epoch
        )

    def __iter__(self):
        return self.dali_iterator.__iter__() # Return the DALI iterator

    def __len__(self):
        # Calculate number of batches per epoch for this rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        samples_per_rank = math.ceil(len(self.file_paths) / world_size)
        num_batches = math.ceil(samples_per_rank / self.batch_size)
        # Adjust for drop_last during training
        if self.is_training and samples_per_rank % self.batch_size != 0:
             num_batches = samples_per_rank // self.batch_size # DALI iterator with DROP policy handles this

        # The DALIGenericIterator handles sharding, so __len__ should return batches per rank
        # However, DALIGenericIterator itself doesn't have a reliable __len__ before iteration.
        # We estimate it based on the total size and batch size.
        # Note: This might not be perfectly accurate if LastBatchPolicy.PARTIAL is used and the last batch is smaller.
        return num_batches


# === Evaluation Function ===
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    threshold,
    num_classes,
    label_cols,
    use_amp,
    amp_dtype,
):
    """Evaluates the model on the given dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Metrics (using torchmetrics for efficiency and correctness, esp. in DDP)
    # Ensure metrics are on the correct device
    metric_device = device  # Calculate metrics on the main device
    # Macro average is generally preferred for multi-label classification
    prec = torchmetrics.classification.MultilabelPrecision(
        num_labels=num_classes, threshold=threshold, average="macro"
    ).to(metric_device)
    rec = torchmetrics.classification.MultilabelRecall(
        num_labels=num_classes, threshold=threshold, average="macro"
    ).to(metric_device)
    f1 = torchmetrics.classification.MultilabelF1Score(
        num_labels=num_classes, threshold=threshold, average="macro"
    ).to(metric_device)
    auc = torchmetrics.classification.MultilabelAUROC(
        num_labels=num_classes, average="macro"
    ).to(metric_device)  # AUC uses probabilities
    # For per-class F1 and confusion matrix (calculated later from aggregated preds/labels)
    per_class_f1 = torchmetrics.classification.MultilabelF1Score(
        num_labels=num_classes, threshold=threshold, average=None
    ).to(metric_device)

    num_batches = len(dataloader)
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        # Handle potential DALI output format
        if isinstance(batch, list) and isinstance(batch[0], dict):  # Common DALI output format
            # Assuming the list contains one dict per GPU, get the one for the current device
            # This might need adjustment based on exact DALI iterator setup
            batch_data = batch[0]
            images = batch_data["image"]  # Already on GPU
            labels = batch_data["label"].to(dtype=torch.float32)  # Ensure type
            if labels.device != device:
                labels = labels.to(device, non_blocking=True)
        elif isinstance(batch, dict):  # Standard PyTorch DataLoader or DALIGenericIterator with direct dict output
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
        else:
            raise TypeError(f"Unexpected batch type from dataloader: {type(batch)}")

        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(images)  # Pass image tensor directly
            # Ensure criterion has reduction='mean' for eval loss calculation
            loss = criterion(logits, labels)  # Assumes criterion passed has reduction='mean'

        total_loss += loss.item()

        # Store predictions and labels for metric calculation
        # AUC needs probabilities (logits before thresholding)
        # Other metrics need predictions based on threshold
        probs = torch.sigmoid(logits)
        preds_thresholded = (probs >= threshold).int()

        # Move tensors to the metric device before updating metrics
        all_preds.append(probs.detach().to(metric_device))
        all_labels.append(labels.detach().to(metric_device))

        # Update metrics directly (torchmetrics handles accumulation)
        prec.update(preds_thresholded, labels.int())
        rec.update(preds_thresholded, labels.int())
        f1.update(preds_thresholded, labels.int())
        auc.update(probs, labels.int())  # AUC uses probabilities
        per_class_f1.update(preds_thresholded, labels.int())

        if is_main_process() and (i + 1) % (num_batches // 5) == 0:  # Log progress periodically
            logger.info(f"  Evaluation Step {i+1}/{num_batches}")

    # Aggregate all predictions and labels (if needed for manual calculation, otherwise compute metrics)
    # all_preds_cat = torch.cat(all_preds, dim=0)
    # all_labels_cat = torch.cat(all_labels, dim=0)

    # Compute final metrics
    avg_loss = total_loss / num_batches
    final_precision = prec.compute().item()
    final_recall = rec.compute().item()
    final_f1 = f1.compute().item()
    final_auc = auc.compute().item()
    final_per_class_f1 = per_class_f1.compute().cpu().numpy()  # Get per-class F1 scores

    # Reset metrics for next evaluation
    prec.reset()
    rec.reset()
    f1.reset()
    auc.reset()
    per_class_f1.reset()

    # Calculate confusion matrix (example for overall TP/TN/FP/FN, needs adjustment for multi-label)
    # This requires thresholded predictions and integer labels
    all_preds_cat_thresh = torch.cat([p.ge(threshold).int() for p in all_preds], dim=0).cpu().numpy().flatten()
    all_labels_cat_int = torch.cat(all_labels, dim=0).int().cpu().numpy().flatten()
    conf_mat = confusion_matrix(all_labels_cat_int, all_preds_cat_thresh)  # Basic binary confusion matrix over all labels

    elapsed_time = time.time() - start_time
    if is_main_process():
        logger.info(f"Evaluation finished in {elapsed_time:.2f}s.")

    # Return metrics including per-class F1 and confusion matrix
    return avg_loss, final_precision, final_recall, final_f1, final_auc, final_per_class_f1, conf_mat


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    scheduler,
    epoch_num,
    total_epochs,
    config: Config,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    num_batches = len(dataloader)
    start_time = time.time()
    processed_samples = 0

    # Get the base model if wrapped in DDP
    base_model = model.module if isinstance(model, DDP) else model

    for i, batch in enumerate(dataloader):
        # Skip batch if collate_fn returned None (due to errors)
        if batch is None:
            logger.warning(f"Skipping empty batch at step {i+1}/{num_batches}")
            continue

        # Handle potential DALI output format
        if config.use_dali:
            if isinstance(batch, list) and isinstance(batch[0], dict):
                batch_data = batch[0]
                images = batch_data["image"]
                labels = batch_data["label"].to(dtype=torch.float32)
                if labels.device != device:
                    labels = labels.to(device, non_blocking=True)
            elif isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"].to(dtype=torch.float32)
                if labels.device != device:
                    labels = labels.to(device, non_blocking=True)
            else:
                raise TypeError(f"Unexpected batch type from DALI loader: {type(batch)}")
        else:  # Standard PyTorch DataLoader
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

        batch_size = images.size(0)

        with autocast(enabled=config.use_amp, dtype=config.amp_dtype):
            logits = model(images)  # Pass image tensor directly

            # Calculate loss (handle OHEM)
            if config.use_ohem:
                per_sample_loss = criterion(logits, labels, reduction="none")
                if per_sample_loss.ndim > 1:
                    per_sample_loss = per_sample_loss.mean(dim=1)

                num_examples = labels.size(0)
                num_hard = max(1, min(num_examples, int(config.ohem_fraction * num_examples)))

                top_k_val, _ = torch.topk(per_sample_loss, k=num_hard)

                if num_hard > 0:
                    hard_loss_threshold = top_k_val[-1]
                    weight = (per_sample_loss >= hard_loss_threshold).float()
                    if weight.sum() < num_hard:
                        _, top_k_indices = torch.topk(per_sample_loss, k=num_hard)
                        weight.zero_()
                        weight[top_k_indices] = 1.0

                    weighted_sum = (per_sample_loss * weight).sum()
                    num_selected = weight.sum().clamp(min=1.0)
                    loss = weighted_sum / num_selected
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            else:
                loss = criterion(logits, labels)

            loss = loss / config.gradient_accumulation_steps

        total_loss += loss.item() * config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        processed_samples += batch_size

        if is_main_process(config.local_rank) and (i + 1) % (num_batches // 10) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (num_batches - (i + 1))
            logger.info(
                f"  Epoch {epoch_num+1}/{total_epochs} | Batch {i+1}/{num_batches} | "
                f"Loss: {loss.item() * config.gradient_accumulation_steps:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Samples/sec: {processed_samples / elapsed_time:.2f} | "
                f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


# === Model Export Functions ===
def export_to_torchscript(model, output_path, device):
    """Exports the model to TorchScript format."""
    logger.info(f"Attempting to export model to TorchScript: {output_path}")
    try:
        model.eval()
        model_to_export = model.to("cpu")
        example_input_tensor = torch.randn(
            1, 1 if config.grayscale else 3, config.img_size, config.img_size
        ).to("cpu")

        try:
            scripted_model = torch.jit.script(model_to_export)
        except Exception as e_script:
            logger.warning(f"TorchScript scripting failed ({e_script}), attempting tracing...")
            try:
                scripted_model = torch.jit.trace(
                    model_to_export, example_input_tensor  # Pass tensor directly
                )
            except Exception as e_trace:
                logger.error(f"TorchScript tracing also failed: {e_trace}", exc_info=True)
                return

        scripted_model.save(output_path)
        logger.info(f"Model successfully exported to TorchScript: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export to TorchScript: {e}", exc_info=True)


def export_to_onnx(model, output_path, device):
    """Exports the model to ONNX format."""
    logger.info(f"Attempting to export model to ONNX: {output_path}")
    try:
        model.eval()
        model_to_export = model.to("cpu")
        example_input_tensor = torch.randn(
            1, 1 if config.grayscale else 3, config.img_size, config.img_size, requires_grad=False
        ).to("cpu")

        input_name = "image"
        input_names = [input_name]
        output_names = ["logits"]

        torch.onnx.export(
            model_to_export,
            example_input_tensor,  # Pass tensor directly
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_name: {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info(f"Model successfully exported to ONNX: {output_path}")
    except ImportError:
        logger.error(
            "ONNX export failed: 'onnx' library not installed. Please install with 'pip install onnx onnxruntime'."
        )
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}", exc_info=True)


# === Optimizer Setup with Layer-wise LR Decay ===
def setup_optimizer(model, config: Config):
    """Sets up the optimizer with optional layer-wise learning rate decay."""
    parameters = []
    classifier_params = []
    backbone_params = []

    if isinstance(model, (MedicalImageModel, EnsembleMedicalModel)):
        base_model = model
    elif isinstance(model, DDP):
        base_model = model.module
    else:
        raise TypeError(f"Unexpected model type for optimizer setup: {type(model)}")

    if isinstance(base_model, EnsembleMedicalModel):
        for backbone in base_model.backbones:
            backbone_params.extend(list(backbone.parameters()))
        classifier_params.extend(list(base_model.classifier.parameters()))
    elif isinstance(base_model, MedicalImageModel):
        backbone_params.extend(list(base_model.img_model.parameters()))
        classifier_params.extend(list(base_model.classifier.parameters()))

    if config.use_layerwise_lr_decay and hasattr(base_model, 'img_model') and hasattr(base_model.img_model, 'layers'):
        logger.info("Applying layer-wise learning rate decay.")
        num_layers = len(base_model.img_model.layers)
        base_lr = config.learning_rate
        decay_rate = config.layerwise_lr_decay_rate

        parameters.append({
            'params': list(base_model.img_model.patch_embed.parameters()),
            'lr': base_lr * (decay_rate ** num_layers)
        })
        for i, layer in enumerate(base_model.img_model.layers):
            layer_lr = base_lr * (decay_rate ** (num_layers - 1 - i))
            parameters.append({'params': list(layer.parameters()), 'lr': layer_lr})
            logger.info(f"  Layer {i} LR: {layer_lr:.2e}")
        if hasattr(base_model.img_model, 'norm'):
             parameters.append({'params': list(base_model.img_model.norm.parameters()), 'lr': base_lr})
        parameters.append({
            'params': classifier_params,
            'lr': base_lr * config.classifier_lr_multiplier
        })
        logger.info(f"  Classifier LR: {base_lr * config.classifier_lr_multiplier:.2e}")

    else:
        logger.info("Using standard optimizer setup (backbone + classifier LRs).")
        parameters.append({'params': backbone_params, 'lr': config.learning_rate})
        parameters.append({
            'params': classifier_params,
            'lr': config.learning_rate * config.classifier_lr_multiplier
        })

    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    return optimizer


def collate_fn(batch):
    """Custom collate_fn to filter out None values."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    local_rank_env = int(os.environ.get("LOCAL_RANK", -1))
    config.local_rank = local_rank_env
    setup_ddp(config.local_rank)

    global logger
    logger = setup_logging(config.output_dir, config.local_rank)

    seed_everything(config.seed + config.local_rank if config.local_rank != -1 else config.seed)

    try:
        if is_main_process(config.local_rank):
            logger.info("Starting main function on main process.")
            logger.info(f"Using device: {config.device}")
            logger.info(
                f"Using Precision: {config.precision} (AMP enabled: {config.use_amp}, dtype: {config.amp_dtype})"
            )
            logger.info(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
            logger.info(f"Using DDP: {dist.is_initialized()}")
            if dist.is_initialized():
                logger.info(f"World Size: {dist.get_world_size()}")
            logger.info(f"Using DALI: {config.use_dali}")
            logger.info(f"Using OHEM: {config.use_ohem} (Fraction: {config.ohem_fraction})")
            logger.info(
                f"Using Layer-wise LR Decay: {config.use_layerwise_lr_decay} (Rate: {config.layerwise_lr_decay_rate})"
            )
            logger.info(f"Ensemble Mode: {config.is_ensemble}")
            if config.is_ensemble:
                logger.info(f"Ensemble Backbones: {config.img_backbone}")
                logger.info(f"Ensemble Weights: {config.ensemble_weights}")
            logger.info(f"Output directory: {config.output_dir}")
            logger.info(f"Data directory: {config.data_dir}")
            logger.info(f"CSV file: {config.csv_file}")

        csv_path = os.path.join(config.data_dir, config.csv_file)

        if is_main_process(config.local_rank):
            logger.info(f"Loading data from {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        df = pd.read_csv(csv_path)

        original_columns = df.columns.tolist()
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        standardized_columns = df.columns.tolist()
        if is_main_process(config.local_rank):
            logger.info(f"Original columns: {original_columns}")
            logger.info(f"Standardized columns: {standardized_columns}")

        label_cols = MedicalDataset.LABEL_COLS
        img_path_col = "dicompath_y"
        if img_path_col not in df.columns:
            logger.error(
                f"Standardized image path column '{img_path_col}' not found in DataFrame. Check CSV header."
            )
            raise ValueError(f"Missing expected standardized column: {img_path_col}")

        if is_main_process(config.local_rank):
            logger.info("Preprocessing labels (NaN/-1.0 -> 0.0)")
        for col in label_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(0.0).replace(-1.0, 0.0)
                df[col] = df[col].astype(np.float32)
            else:
                logger.error(
                    f"Standardized label column '{col}' not found in DataFrame after standardization. Check CSV and LABEL_COLS definition."
                )
                raise ValueError(f"Missing expected standardized column: {col}")

        split_col = "split"
        if split_col not in df.columns:
            logger.error(
                f"Standardized column '{split_col}' not found in DataFrame. Cannot split data."
            )
            return

        if is_main_process(config.local_rank):
            logger.info("Splitting data into train, validation, test sets.")
        df_train = df[df[split_col] == "train"].reset_index(drop=True)
        df_val = df[df[split_col] == "validate"].reset_index(drop=True)
        df_test = df[df[split_col] == "test"].reset_index(drop=True)

        if len(df_train) == 0 or len(df_val) == 0:
            logger.error(
                "Training or validation set is empty after splitting. Check 'split' column in CSV."
            )
            return

        if is_main_process(config.local_rank):
            logger.info(
                f"Train samples: {len(df_train)}, Validation samples: {len(df_val)}, Test samples: {len(df_test)}"
            )

        train_loader, val_loader, test_loader = None, None, None
        if config.use_dali:
            if is_main_process(config.local_rank):
                logger.info("Creating DALI dataloaders.")
            train_loader = DALIDataLoader(
                df_train, config, is_training=True, data_dir=config.data_dir
            )
            val_loader = DALIDataLoader(
                df_val, config, is_training=False, data_dir=config.data_dir
            )
            test_loader = (
                DALIDataLoader(
                    df_test, config, is_training=False, data_dir=config.data_dir
                )
                if len(df_test) > 0
                else None
            )

        else:
            if is_main_process(config.local_rank):
                logger.info("Creating PyTorch dataloaders.")
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]
            fill_value = img_mean if not config.grayscale else 0

            train_transform = A.Compose(
                [
                    A.Resize(height=config.img_size, width=config.img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, p=0.5, border_mode=0),
                    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3)
                    if not config.grayscale
                    else A.NoOp(),
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=config.img_size // 10,
                        max_width=config.img_size // 10,
                        min_holes=1,
                        min_height=config.img_size // 20,
                        min_width=config.img_size // 20,
                        fill_value=fill_value,
                        mask_fill_value=None,
                        p=0.3,
                    ),
                    A.Normalize(mean=img_mean, std=img_std),
                    ToTensorV2(),
                ]
            )
            val_test_transform = A.Compose(
                [
                    A.Resize(height=config.img_size, width=config.img_size),
                    A.Normalize(mean=img_mean, std=img_std),
                    ToTensorV2(),
                ]
            )

            train_dataset = MedicalDataset(
                df_train, config, transform=train_transform, data_dir=config.data_dir
            )
            val_dataset = MedicalDataset(
                df_val, config, transform=val_test_transform, data_dir=config.data_dir
            )
            test_dataset = (
                MedicalDataset(
                    df_test, config, transform=val_test_transform, data_dir=config.data_dir
                )
                if len(df_test) > 0
                else None
            )

            train_sampler = None
            shuffle_train = True
            if dist.is_initialized():
                train_sampler = DistributedSampler(
                    train_dataset, shuffle=True, seed=config.seed, drop_last=True
                )
                shuffle_train = False

            persistent_workers = config.num_workers > 0

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=shuffle_train,
                sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                drop_last=True if dist.is_initialized() else False,
                collate_fn=collate_fn,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size * 2,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
            test_loader = (
                DataLoader(
                    test_dataset,
                    batch_size=config.batch_size * 2,
                    shuffle=False,
                    num_workers=config.num_workers,
                    pin_memory=True,
                    persistent_workers=persistent_workers,
                    collate_fn=collate_fn,
                )
                if test_dataset
                else None
            )

        if is_main_process(config.local_rank):
            logger.info("Initializing model.")
        model_device = (
            torch.device(f"cuda:{config.local_rank}")
            if config.local_rank != -1
            else torch.device(config.device)
        )
        if config.is_ensemble:
            model = EnsembleMedicalModel(config).to(model_device)
        else:
            model = MedicalImageModel(config).to(model_device)

        optimizer = setup_optimizer(model, config)

        if dist.is_initialized():
            if is_main_process(config.local_rank):
                logger.info("Wrapping model with DistributedDataParallel.")
            find_unused = (
                config.use_gradient_checkpointing
                or config.is_ensemble
                or config.use_layerwise_lr_decay
            )
            model = DDP(
                model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=find_unused,
            )

        if config.use_focal_loss:
            if is_main_process(config.local_rank):
                logger.info(
                    f"Using Focal Loss with alpha={config.focal_loss_alpha}, gamma={config.focal_loss_gamma}"
                )
            criterion = FocalLoss(
                alpha=config.focal_loss_alpha,
                gamma=config.focal_loss_gamma,
                reduction="none" if config.use_ohem else "mean",
            )
        else:
            if is_main_process(config.local_rank):
                logger.info("Using BCEWithLogitsLoss.")
            criterion = nn.BCEWithLogitsLoss(
                reduction="none" if config.use_ohem else "mean"
            )
        criterion = criterion.to(model_device)

        if config.use_dali:
            try:
                dali_train_len = len(train_loader)
            except TypeError:
                 dali_train_len = math.ceil(len(train_loader.file_paths) / (train_loader.batch_size * (dist.get_world_size() if dist.is_initialized() else 1)))
                 logger.warning(f"Could not get DALI loader length directly, estimated as {dali_train_len} batches per rank.")

            effective_train_loader_len = dali_train_len // config.gradient_accumulation_steps

        else:
            num_train_samples = len(train_loader.dataset)
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            samples_per_rank = (
                math.floor(num_train_samples / world_size)
                if dist.is_initialized() and train_loader.drop_last
                else math.ceil(num_train_samples / world_size)
            )
            batches_per_rank = math.ceil(samples_per_rank / config.batch_size)
            effective_train_loader_len = batches_per_rank // config.gradient_accumulation_steps

        num_training_steps = effective_train_loader_len * config.epochs
        num_warmup_steps = config.num_warmup_steps
        num_warmup_steps = min(num_warmup_steps, num_training_steps // 10)
        if is_main_process(config.local_rank):
            logger.info(f"Effective steps per epoch (per rank): {effective_train_loader_len}")
            logger.info(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

        if num_training_steps < 1:
            logger.warning(
                f"Calculated num_training_steps ({num_training_steps}) is less than 1. Setting to 1. Check dataloader length and batch size."
            )
            num_training_steps = 1
            num_warmup_steps = min(num_warmup_steps, 1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")

        best_val_metric = 0.0
        best_epoch = -1
        best_model_path = os.path.join(config.output_dir, "best_model.pth")

        if is_main_process(config.local_rank):
            logger.info("Starting training and validation loop...")
        for epoch in range(config.epochs):
            if dist.is_initialized() and not config.use_dali and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if is_main_process(config.local_rank):
                logger.info(f"--- Epoch {epoch+1}/{config.epochs} ---")

            model_to_train = model.module if dist.is_initialized() else model
            if epoch < config.freeze_backbone_epochs:
                if epoch == 0:
                    logger.info(f"Freezing backbone for the first {config.freeze_backbone_epochs} epochs.")
                model_to_train.freeze_backbone()
            elif epoch == config.freeze_backbone_epochs:
                logger.info(f"Unfreezing backbone starting from epoch {epoch+1}.")
                model_to_train.unfreeze_backbone()
                optimizer = setup_optimizer(model, config)
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                model_device,
                scaler,
                scheduler,
                epoch,
                config.epochs,
                config,
            )

            eval_criterion = criterion
            if config.use_ohem:
                if config.use_focal_loss:
                    eval_criterion = FocalLoss(
                        alpha=config.focal_loss_alpha,
                        gamma=config.focal_loss_gamma,
                        reduction="mean",
                    ).to(model_device)
                else:
                    eval_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(model_device)

            eval_model = model.module if dist.is_initialized() else model
            val_loss, val_precision, val_recall, val_f1, val_auc, _, _ = evaluate(
                eval_model,
                val_loader,
                eval_criterion,
                model_device,
                config.threshold,
                config.num_classes,
                label_cols,
                config.use_amp,
                config.amp_dtype,
            )

            if is_main_process(config.local_rank):
                logger.info(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
                logger.info(f"Epoch {epoch+1} Validation Results (Threshold={config.threshold}):")
                logger.info(f"  Loss: {val_loss:.4f}")
                logger.info(f"  Precision (Macro): {val_precision:.4f}")
                logger.info(f"  Recall (Macro): {val_recall:.4f}")
                logger.info(f"  F1-Score (Macro): {val_f1:.4f}")
                logger.info(f"  AUC (Macro): {val_auc:.4f}")

                current_metric = val_f1

                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_epoch = epoch + 1
                    model_state_dict = (
                        model.module.state_dict() if dist.is_initialized() else model.state_dict()
                    )
                    checkpoint = {
                        "epoch": best_epoch,
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_metric": best_val_metric,
                        "config": config.dict(),
                    }
                    torch.save(checkpoint, best_model_path)
                    logger.info(
                        f"Validation metric improved to {best_val_metric:.4f}. Saved model checkpoint to {best_model_path}"
                    )

            if dist.is_initialized():
                dist.barrier()

        if is_main_process(config.local_rank):
            logger.info(
                f"Training finished. Best validation F1 ({best_val_metric:.4f}) achieved at epoch {best_epoch}."
            )

            if best_epoch == -1:
                logger.warning(
                    "No best model saved during training (validation metric did not improve). Skipping threshold optimization, testing, and export."
                )
            else:
                logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location="cpu")
                loaded_config_dict = checkpoint.get(
                    "config", config.dict()
                )
                loaded_config = Config(**loaded_config_dict)
                eval_device = torch.device(config.device)
                if loaded_config.is_ensemble:
                    base_model = EnsembleMedicalModel(loaded_config).to(eval_device)
                else:
                    base_model = MedicalImageModel(loaded_config).to(eval_device)
                base_model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Best model loaded successfully.")

                if loaded_config.use_focal_loss:
                    eval_criterion = FocalLoss(
                        alpha=loaded_config.focal_loss_alpha,
                        gamma=loaded_config.focal_loss_gamma,
                        reduction="mean",
                    ).to(eval_device)
                else:
                    eval_criterion = nn.BCEWithLogitsLoss(reduction="mean").to(eval_device)

                logger.info("--- Starting Threshold Optimization on Validation Set ---")
                best_threshold = config.threshold
                best_thr_f1 = 0.0

                _, _, _, initial_val_f1, _, _, _ = evaluate(
                    base_model,
                    val_loader,
                    eval_criterion,
                    eval_device,
                    best_threshold,
                    loaded_config.num_classes,
                    label_cols,
                    loaded_config.use_amp,
                    loaded_config.amp_dtype,
                )
                best_thr_f1 = initial_val_f1
                logger.info(f"  Initial Threshold {best_threshold:.2f}: Validation F1 = {best_thr_f1:.4f}")


                thresholds_to_try = np.arange(0.1, 0.95, 0.05)
                logger.info(
                    f"Optimizing threshold using F1-score on validation set across thresholds: {thresholds_to_try}"
                )
                for thr in thresholds_to_try:
                    thr = round(thr, 2)
                    if thr == best_threshold:
                        continue
                    _, _, _, val_f1_thr, _, _, _ = evaluate(
                        base_model,
                        val_loader,
                        eval_criterion,
                        eval_device,
                        thr,
                        loaded_config.num_classes,
                        label_cols,
                        loaded_config.use_amp,
                        loaded_config.amp_dtype,
                    )
                    logger.info(f"  Threshold {thr:.2f}: Validation F1 = {val_f1_thr:.4f}")
                    if val_f1_thr > best_thr_f1:
                        best_thr_f1 = val_f1_thr
                        best_threshold = thr
                        logger.info(
                            f"    New best F1 ({best_thr_f1:.4f}) found with threshold {best_threshold:.2f}"
                        )

                logger.info(
                    f"Optimal threshold found: {best_threshold:.2f} with Validation F1: {best_thr_f1:.4f}"
                )

                if test_loader:
                    logger.info(
                        f"--- Final Evaluation on Test Set using Optimal Threshold {best_threshold:.2f} ---"
                    )
                    (
                        test_loss,
                        test_precision,
                        test_recall,
                        test_f1,
                        test_auc,
                        test_class_f1,
                        test_conf_mat,
                    ) = evaluate(
                        base_model,
                        test_loader,
                        eval_criterion,
                        eval_device,
                        best_threshold,
                        loaded_config.num_classes,
                        label_cols,
                        loaded_config.use_amp,
                        loaded_config.amp_dtype,
                    )

                    logger.info("Final Test Set Results:")
                    logger.info(f"  Loss: {test_loss:.4f}")
                    logger.info(
                        f"  Precision (Macro, Thr={best_threshold:.2f}): {test_precision:.4f}"
                    )
                    logger.info(
                        f"  Recall (Macro, Thr={best_threshold:.2f}): {test_recall:.4f}"
                    )
                    logger.info(
                        f"  F1-Score (Macro, Thr={best_threshold:.2f}): {test_f1:.4f}"
                    )
                    logger.info(
                        f"  AUC (Macro): {test_auc:.4f}"
                    )
                    logger.info(f"  Per-Class F1 (Thr={best_threshold:.2f}):")
                    for i, label_name in enumerate(label_cols):
                        logger.info(f"    {label_name}: {test_class_f1[i]:.4f}")
                    logger.info(f"  Confusion Matrix (Flattened, Thr={best_threshold:.2f}):\n{test_conf_mat}")

                else:
                    logger.info("No test set found, skipping final test evaluation.")

                base_model.to("cpu")
                base_model.eval()

                ts_path = os.path.join(config.output_dir, "model.ts")
                export_to_torchscript(base_model, ts_path, "cpu")

                onnx_path = os.path.join(config.output_dir, "model.onnx")
                export_to_onnx(base_model, onnx_path, "cpu")

    except FileNotFoundError as e:
        logger.error(f"Data loading error: {e}")
    except ValueError as e:
        logger.error(f"Configuration or data error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        cleanup_ddp()
        if is_main_process(config.local_rank):
            logger.info("Main function finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Medical Image Classification Model")
    parser.add_argument("--img_backbone", type=str, help="Model backbone name")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Base learning rate")
    parser.add_argument("--data_dir", type=str, help="Directory containing the data and CSV")
    parser.add_argument("--csv_file", type=str, help="Name of the CSV file within data_dir")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], help="Training precision")
    parser.add_argument("--use_dali", action="store_true", help="Use NVIDIA DALI for data loading")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by torchrun)")

    args = parser.parse_args()

    config_updates = {k: v for k, v in vars(args).items() if v is not None}
    config = Config(**{**config.dict(), **config_updates})

    main()