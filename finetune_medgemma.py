# --- START OF REVISED FILE finetune_medgemma.py ---

"""
Production-grade fine-tuning script for MedGemma models on the CXLSeg dataset.

This script supports:
- LoRA and QLoRA (4-bit) fine-tuning
- Mixed-precision training (fp16/bf16)
- Gradient checkpointing and accumulation for memory efficiency
- Flash Attention 2 for optimized performance on supported GPUs
- CPU-only training with automatic feature disabling for compatibility
- Experiment tracking with Weights & Biases
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

# --- Core ML/DL Libraries ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Transformers Ecosystem ---
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import set_seed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# --- Datasets and Metrics ---
from datasets import Dataset as HFDataset
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms as transforms

# --- Experiment Tracking ---
import wandb

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CXLSegDataset(Dataset):
    """Dataset class for CXLSeg chest X-ray data with abnormality detection."""
    
    def __init__(
        self,
        jsonl_path: str,
        data_root: str,
        split: str,
        tokenizer,
        max_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_length = max_length
        self.image_size = image_size
        self.tokenizer = tokenizer
        
        self.data = self._load_data(jsonl_path)
        if not self.data:
            logger.warning(f"No data found for split '{split}' in {jsonl_path}.")
            return

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # NOTE: For robust multi-label classification, the binarizer should be fit
        # on the full vocabulary of abnormalities from the entire dataset (all splits).
        # For simplicity here, we define them statically.
        self.abnormality_classes = [
            "No Finding", "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion",
            "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        self.mlb = MultiLabelBinarizer(classes=self.abnormality_classes).fit([[]]) # Fit with empty to initialize
        
    def _load_data(self, jsonl_path: str) -> List[Dict]:
        """Load and filter data from a JSONL file for the specified split."""
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get('split') == self.split:
                        image_path = self.data_root / item['image_path']
                        if image_path.exists():
                            data.append(item)
                        else:
                            logger.warning(f"Image not found, skipping: {image_path}")
        except FileNotFoundError:
            logger.error(f"Dataset file not found at {jsonl_path}")
        return data

    def _load_image(self, image_path_str: str) -> torch.Tensor:
        """Load and preprocess an image, returning a blank tensor on error."""
        full_path = self.data_root / image_path_str
        try:
            image = Image.open(full_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            return torch.zeros(3, *self.image_size) # Return a blank image as a fallback
    
    def _create_prompt(self, report: str, abnormalities: List[str]) -> str:
        """Create the full instruction-following prompt string for the model."""
        clean_abnormalities = [ab for ab in abnormalities if ab and ab.strip()]
        if not clean_abnormalities:
            clean_abnormalities = ["No Finding"]
        
        system_prompt = "You are a medical AI assistant specialized in analyzing chest X-rays. Based on the provided chest X-ray image, identify any abnormalities present."
        user_prompt = "Analyze this chest X-ray image and identify any abnormalities. Provide your findings in a clear, medical format."
        abnormalities_text = ", ".join(clean_abnormalities)
        assistant_response = f"Based on the chest X-ray analysis:\n\nFindings: {report}\n\nAbnormalities detected: {abnormalities_text}"
        
        # Using the MedGemma chat template format
        return (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{assistant_response}<end_of_turn>"
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image = self._load_image(item['image_path'])
        formatted_text = self._create_prompt(item['report'], item['abnormalities'])
        
        # The <bos> token is often added automatically by the tokenizer, but we ensure it here.
        full_text_with_bos = self.tokenizer.bos_token + formatted_text
        
        encoding = self.tokenizer(
            full_text_with_bos,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        abnormality_labels = self.mlb.transform([item['abnormalities']])[0]
        
        # The model expects 'labels' for causal language modeling loss calculation.
        # It will automatically shift the labels internally.
        labels = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'image': image, # Note: The base model doesn't use this, but kept for potential multi-modal extension
            'abnormality_labels': torch.FloatTensor(abnormality_labels),
        }


def setup_model_and_tokenizer(
    model_name: str,
    use_4bit: bool,
    use_lora: bool,
    lora_config_dict: Optional[Dict] = None
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Sets up the model and tokenizer with appropriate configurations for quantization and PEFT.

    Args:
        model_name: The name of the model from Hugging Face Hub.
        use_4bit: Whether to use 4-bit quantization (QLoRA).
        use_lora: Whether to apply LoRA.
        lora_config_dict: A dictionary with LoRA configuration parameters.

    Returns:
        A tuple containing the configured model and tokenizer.
    """
    logger.info(f"Setting up model and tokenizer for '{model_name}'...")
    
    cuda_available = torch.cuda.is_available()
    bnb_config = None
    torch_dtype = torch.float32  # Default for CPU
    attn_implementation = "eager" # Default attention

    if cuda_available:
        torch_dtype = torch.float16 # Default for GPU
        if use_4bit:
            logger.info("CUDA available. Configuring 4-bit quantization (QLoRA).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        
        # Check for Flash Attention 2
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            logger.info("Flash Attention 2 detected. Will be used for training.")
        except ImportError:
            logger.info("Flash Attention 2 not installed. Using default attention. For faster training, run: pip install flash-attn --no-build-isolation")
    
    elif use_4bit:
        logger.warning("4-bit quantization was requested, but CUDA is not available. Disabling 4-bit.")
        use_4bit = False
        
    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # MedGemma uses right-side padding
    tokenizer.padding_side = "right"

    # --- Load Model ---
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map="auto" if cuda_available else "cpu",
        attn_implementation=attn_implementation if cuda_available else "eager",
    )

    if use_4bit and cuda_available:
        model = prepare_model_for_kbit_training(model)
        
    if use_lora:
        if lora_config_dict is None:
            raise ValueError("lora_config_dict must be provided if use_lora is True.")
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, peft_config)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        logger.info(f"Trainable PEFT parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
    
    # Disable cache for training
    model.config.use_cache = False
    
    return model, tokenizer


def compute_metrics(eval_pred):
    """Computes perplexity from model predictions."""
    logits, labels = eval_pred
    # The loss is calculated on the logits for the next token prediction.
    # The Trainer handles the shifting internally. We just need to compute the metric.
    # CrossEntropyLoss expects logits of shape (batch_size * sequence_length, num_classes)
    # and labels of shape (batch_size * sequence_length).
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    # We need to flatten the first two dimensions of logits and labels
    logits_flat = logits.view(-1, logits.shape[-1])
    labels_flat = labels.view(-1)
    
    loss = loss_fct(logits_flat, labels_flat)
    perplexity = torch.exp(loss).item()
    
    return {"perplexity": perplexity, "eval_loss": loss.item()}


def load_split_dataset(split_name: str, args: argparse.Namespace, tokenizer) -> Optional[CXLSegDataset]:
    """Helper function to load a single dataset split and handle errors."""
    logger.info(f"Attempting to load '{split_name}' dataset...")
    try:
        dataset = CXLSegDataset(
            jsonl_path=args.jsonl_path,
            data_root=args.data_root,
            split=split_name,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        if len(dataset) > 0:
            logger.info(f"Successfully loaded {len(dataset)} samples for '{split_name}' split.")
            return dataset
        else:
            logger.warning(f"No data found for '{split_name}' split. It will be skipped.")
            return None
    except Exception as e:
        logger.error(f"Failed to load '{split_name}' dataset: {e}", exc_info=True)
        return None

def main():
    # --- (Argument parsing and other setup code remains the same) ---
    # ... (parser setup, sanity checks, model/tokenizer/dataset loading)
    # ...

    parser = argparse.ArgumentParser(description="Fine-tune MedGemma on the CXLSeg dataset.")
    
    # --- Grouped Arguments for Clarity ---
    data_group = parser.add_argument_group("Data and Model Paths")
    data_group.add_argument("--model_name", type=str, default="google/medgemma-4b-it", help="Model name or path from Hugging Face Hub.")
    data_group.add_argument("--data_root", type=str, required=True, help="Root directory containing the image files.")
    data_group.add_argument("--jsonl_path", type=str, required=True, help="Path to the JSONL dataset file.")
    data_group.add_argument("--output_dir", type=str, default="./medgemma_finetuned", help="Output directory for saving model checkpoints.")
    data_group.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")

    training_group = parser.add_argument_group("Training Hyperparameters")
    training_group.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    training_group.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    training_group.add_argument("--eval_batch_size", type=int, default=4, help="Per-device evaluation batch size.")
    training_group.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate.")
    training_group.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    training_group.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    training_group.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    training_group.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping.")
    
    optim_group = parser.add_argument_group("Performance and Optimization")
    optim_group.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training type.")
    optim_group.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps to accumulate gradients before updating.")
    optim_group.add_argument("--use_4bit", action="store_true", help="Enable 4-bit quantization (QLoRA).")
    optim_group.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")
    
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning.")
    lora_group.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r).")
    lora_group.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    lora_group.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers.")

    run_group = parser.add_argument_group("Execution and Logging")
    run_group.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    run_group.add_argument("--eval_steps", type=int, default=250, help="Evaluate on the validation set every N steps.")
    run_group.add_argument("--logging_steps", type=int, default=50, help="Log training metrics every N steps.")
    run_group.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    run_group.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for experiment tracking.")
    run_group.add_argument("--wandb_project", type=str, default="medgemma-cxlseg", help="W&B project name.")
    run_group.add_argument("--run_name", type=str, default=None, help="A specific name for the W&B run.")
    args = parser.parse_args()
    set_seed(args.seed)

    # --- Hardware and Optimization Sanity Checks ---
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.warning("!! CUDA is not available. Training on CPU. !!")
        logger.warning("!! Performance will be severely degraded.   !!")
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if args.use_4bit:
            args.use_4bit = False
            logger.warning("Disabling --use_4bit as it requires CUDA.")
        if args.mixed_precision != "no":
            args.mixed_precision = "no"
            logger.warning("Disabling mixed precision as it requires CUDA.")
        if args.gradient_checkpointing:
            args.gradient_checkpointing = False
            logger.warning("Disabling --gradient_checkpointing as it's not beneficial on CPU.")
    
    if args.mixed_precision == "bf16" and (not cuda_available or not torch.cuda.is_bf16_supported()):
        logger.warning("bf16 is not supported on this device. Falling back to fp16.")
        args.mixed_precision = "fp16"
        
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=args)

    # --- Setup Model and Tokenizer ---
    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    } if args.use_lora else None

    model, tokenizer = setup_model_and_tokenizer(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        use_lora=args.use_lora,
        lora_config_dict=lora_config
    )
    
    # --- Load Datasets ---
    train_dataset = load_split_dataset("train", args, tokenizer)
    eval_dataset = load_split_dataset("validate", args, tokenizer)
    
    if not train_dataset:
        logger.error("Training dataset could not be loaded. Aborting.")
        return
        
    # =========================================================================
    # --- START OF REVISED CODE BLOCK ---
    # =========================================================================

    # --- Configure Trainer with Robust, Backward-Compatible Logic ---
    # =========================================================================
    # --- FIXED TRAINING ARGUMENTS CONFIGURATION ---
    # =========================================================================

    # --- Configure Trainer with Robust, Backward-Compatible Logic ---
    import inspect
    training_args_signature = inspect.signature(TrainingArguments.__init__)

    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "optim": "adamw_torch",
        "fp16": (args.mixed_precision == "fp16"),
        "bf16": (args.mixed_precision == "bf16"),
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "report_to": "wandb" if args.use_wandb else "none",
        "run_name": args.run_name,
        "seed": args.seed,
    }

    # Conditionally add evaluation-related arguments ONLY if an eval dataset is present.
    if eval_dataset:
        logger.info("Evaluation dataset found. Enabling evaluation during training.")
        
        # Handle different transformers versions
        if "eval_strategy" in training_args_signature.parameters:
            # Modern transformers 4.46+
            training_args_dict["eval_strategy"] = "steps"
        elif "evaluation_strategy" in training_args_signature.parameters:
            # Transformers 3.0-4.45
            training_args_dict["evaluation_strategy"] = "steps"
        else:
            # Very old transformers versions
            training_args_dict["do_eval"] = True
        
        # These are now correctly tied to the presence of an eval_dataset
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "eval_loss"
        training_args_dict["greater_is_better"] = False
    else:
        # If no eval dataset, ensure these are not set or set to appropriate defaults
        logger.info("No evaluation dataset found. Disabling evaluation-dependent features.")
        training_args_dict["load_best_model_at_end"] = False
        
        # Handle different transformers versions for disabling evaluation
        if "eval_strategy" in training_args_signature.parameters:
            # Modern transformers 4.46+
            training_args_dict["eval_strategy"] = "no"
        elif "evaluation_strategy" in training_args_signature.parameters:
            # Transformers 3.0-4.45
            training_args_dict["evaluation_strategy"] = "no"
        else:
            # Very old transformers versions
            training_args_dict["do_eval"] = False
        
        # Don't set eval_steps when evaluation is disabled
        # Don't set metric_for_best_model when load_best_model_at_end is False

    training_args = TrainingArguments(**training_args_dict)

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # =========================================================================
    # --- END OF REVISED CODE BLOCK ---
    # =========================================================================

    # --- Start Training and Final Evaluation ---
    logger.info("Starting model training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logger.info(f"Training completed successfully. Metrics: {train_result.metrics}")
        
        logger.info(f"Saving final model and tokenizer to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    except (KeyboardInterrupt, Exception) as e:
        logger.error(f"Training stopped prematurely: {e}", exc_info=True)
        # Still try to save the current state if it's not a critical failure
        if isinstance(e, KeyboardInterrupt):
            logger.info("Keyboard interrupt received. Saving current model state...")
            trainer.save_model(os.path.join(args.output_dir, "interrupted_checkpoint"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "interrupted_checkpoint"))
        return # Exit after interruption/error

    # --- Final Evaluation on Test Set ---
    test_dataset = load_split_dataset("test", args, tokenizer)
    if test_dataset:
        logger.info("Performing final evaluation on the test set...")
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        logger.info(f"Test Set Results: {test_results}")

        results_path = os.path.join(args.output_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=4)
        logger.info(f"Test results saved to {results_path}")

        if args.use_wandb:
            wandb.log(test_results)

    if args.use_wandb:
        wandb.finish()

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
