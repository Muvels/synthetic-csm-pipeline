# Sesame CSM Training

This directory contains the training script for fine-tuning the Sesame CSM (1B) model using Unsloth.

## Overview

The `train.py` script loads the pre-processed dataset shards and fine-tunes the model using 4-bit LoRA (Low-Rank Adaptation) for efficiency. It leverages `unsloth` for faster training and lower memory usage.

## Installation

This project uses `uv` for dependency management.

**Note**: `unsloth` requires a CUDA-enabled GPU (NVIDIA). It does not support macOS (Apple Silicon).

### Environment Setup

Create a `.env` file in this directory to store your secrets (e.g., WandB API key):

```bash
WANDB_API_KEY=your_api_key_here
```

```bash
uv sync
```

## Usage

### Train Model

Run the training script pointing to your prepared dataset:

```bash
uv run train.py \
    --dataset_path ../prep_for_training/prepared_dataset \
    --output_dir ./outputs \
    --config ./configs/finetune_param_defaults_bs4.yaml \
    --max_steps 60 \
    --use_wandb
```

### Arguments

*   `--config`: Path to a YAML configuration file (e.g., `configs/finetune_param_defaults_bs4.yaml`). Values in the config file override defaults, but can be overridden by explicit CLI arguments.
*   `--dataset_path`: Path to the directory containing the prepared Arrow shards (output of `prepare_dataset_for_training.py`).
*   `--output_dir`: Directory to save the trained model and checkpoints.
*   `--max_steps`: Total number of training steps (default: -1, uses epochs).
*   `--n_epochs`: Number of training epochs (default: 2).
*   `--batch_size`: Per-device training batch size (default: 2).
*   `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 4).
*   `--learning_rate`: Learning rate (default: 2e-4).
*   `--model_id`: Base model ID (default: "unsloth/csm-1b").
*   `--seed`: Random seed (default: 3407).
*   `--use_wandb`: Flag to enable Weights & Biases tracking (requires `WANDB_API_KEY` in `.env`).
*   `--full_finetune`: Flag to finetune all weights instead of using LoRA (default: False).
*   `--warmup_steps`: Warmup steps (default: 5).
*   `--weight_decay`: Weight decay (default: 0.001).
*   `--weight_decay`: Weight decay (default: 0.001).
*   `--lr_scheduler_type`: Learning rate scheduler type (default: "linear").
*   `--max_grad_norm`: Max gradient norm (default: 1.0).
*   `--gen_every`: Generate audio sample every N steps (default: 1000).
*   `--gen_from`: Path to a folder containing `segments/script.txt` and audio files for generation context (e.g., `example_gen`).

### Inference

You can also generate audio using the trained model (or base model) without training:

```bash
uv run inference.py \
    --model_path unsloth/csm-1b \
    --input_dir ./example_gen \
    --output_dir ./inference_output
```

**Arguments:**
*   `--model_path`: Path to the trained model directory or Hugging Face model ID.
*   `--input_dir`: Path to the folder containing `segments/script.txt` and audio context.
*   `--output_dir`: Directory to save the generated audio.
*   `--load_in_4bit`: Flag to load the model in 4-bit quantization (useful for low VRAM).

## Hardware Requirements

*   **GPU**: NVIDIA GPU with at least 8GB VRAM (Tesla T4 or better recommended).
*   **OS**: Linux (or WSL on Windows). macOS is **not** supported due to `unsloth` dependencies.

## Output

The script saves the fine-tuned model adapters and tokenizer to the specified `--output_dir`.
