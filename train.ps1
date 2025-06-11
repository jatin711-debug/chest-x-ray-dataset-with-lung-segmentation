# =================================================================
# Production-Style Runner for finetune_medgemma.py
#
# How to use:
# 1. Ensure you have the required Python environment activated.
# 2. Modify the parameters in the "Configuration" section below.
# 3. Run the script from your PowerShell terminal: .\run_finetune.ps1
# =================================================================

# --- Configuration ---
# Set all your training parameters here.

# --- Paths ---
$model_name   = "google/medgemma-4b-it" # Changed to 2b for easier local testing
$data_root    = "."                       # Assumes 'files' folder is in the current directory
$jsonl_path   = ".\cxlseg_finetuning_dataset.jsonl"
$output_dir   = ".\medgemma_cxlseg_finetuned"

# --- Training Hyperparameters ---
$batch_size                  = 2       # Per-device train batch size. Reduce if you get Out-of-Memory (OOM) errors.
$eval_batch_size             = 4       # Per-device eval batch size.
$learning_rate               = 2e-4
$num_epochs                  = 3
$max_length                  = 512
$gradient_accumulation_steps = 8       # Effective batch size = $batch_size * $gradient_accumulation_steps (e.g., 2 * 8 = 16)

# --- LoRA Configuration (only used if $use_lora is $true) ---
$lora_r       = 16
$lora_alpha   = 32
$lora_dropout = 0.1

# --- Performance & Optimization Toggles (Set to $true or $false) ---
[bool]$use_lora                 = $true  # Enable LoRA?
[bool]$use_4bit                 = $true  # Enable 4-bit QLoRA? (Requires CUDA)
[bool]$gradient_checkpointing   = $true  # Enable gradient checkpointing to save VRAM? (Requires CUDA)
$mixed_precision                = "fp16" # "fp16" or "bf16" (if supported), or "no"

# --- Experiment Tracking (Set $use_wandb to $true to enable) ---
[bool]$use_wandb    = $false # Set to $true to log to Weights & Biases
$wandb_project      = "medgemma-cxlseg-prod"
$run_name           = "medgemma-qlora-$(Get-Date -Format 'yyyyMMdd-HHmm')"

# --- Logging and Saving Intervals ---
$warmup_steps  = 100
$save_steps    = 500
$eval_steps    = 250
$logging_steps = 50
$seed          = 42                      # Random seed for reproducibility
# --- End of Configuration ---


# =================================================================
# Script execution logic - no need to edit below this line
# =================================================================

# Arguments that take a value
$value_params = @{
    "--model_name"                  = $model_name
    "--data_root"                   = $data_root
    "--jsonl_path"                  = $jsonl_path
    "--output_dir"                  = $output_dir
    "--batch_size"                  = $batch_size.ToString()
    "--eval_batch_size"             = $eval_batch_size.ToString()
    "--learning_rate"               = $learning_rate.ToString() # PowerShell typically handles float to string well
    "--num_epochs"                  = $num_epochs.ToString()
    "--max_length"                  = $max_length.ToString()
    "--gradient_accumulation_steps" = $gradient_accumulation_steps.ToString()
    "--lora_r"                      = $lora_r.ToString()
    "--lora_alpha"                  = $lora_alpha.ToString()
    "--lora_dropout"                = $lora_dropout.ToString()
    "--warmup_steps"                = $warmup_steps.ToString()
    "--save_steps"                  = $save_steps.ToString()
    "--eval_steps"                  = $eval_steps.ToString()
    "--logging_steps"               = $logging_steps.ToString()
    "--mixed_precision"             = $mixed_precision
    "--wandb_project"               = $wandb_project
    "--run_name"                    = $run_name
    "--seed"                        = $seed.ToString()
}

# Boolean flags (action="store_true" in Python)
$boolean_flags = @()
if ($use_lora) { $boolean_flags += "--use_lora" }
if ($use_4bit) { $boolean_flags += "--use_4bit" }
if ($gradient_checkpointing) { $boolean_flags += "--gradient_checkpointing" }
if ($use_wandb) { $boolean_flags += "--use_wandb" }


# Announce the run
Write-Host "Starting Python training script with the following configuration:" -ForegroundColor Yellow
$value_params.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0}: {1}" -f $_.Name, $_.Value)
}
if ($boolean_flags.Count -gt 0) {
    Write-Host "  Boolean flags set:"
    $boolean_flags | Sort-Object | ForEach-Object { Write-Host ("    {0}" -f $_) }
}
Write-Host ""

# Construct the argument list for the external command
$final_args_list = @()
$value_params.GetEnumerator() | ForEach-Object {
    $final_args_list += $_.Name
    $final_args_list += $_.Value
}
$final_args_list += $boolean_flags

# For debugging purposes, you can uncomment the next line to see the exact command
# Write-Host "Executing: python.exe .\\finetune_medgemma.py $($final_args_list -join ' ')"