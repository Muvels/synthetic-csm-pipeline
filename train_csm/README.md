# Sesame CSM Training

This directory contains the training script for fine-tuning the Sesame CSM (1B) model using Unsloth.

## Overview

The `train.py` script provides memory-efficient fine-tuning of Sesame CSM-1B:

- **Memory-efficient preprocessing**: Use `preprocess_dataset.py` to tokenize data once, then train without loading audio
- **Context-aware training**: Includes previous N conversation turns as context for better voice consistency
- **Flexible configuration**: Supports both YAML config files and CLI arguments
- **Periodic generation**: Generates sample audio during training to monitor progress
- **LoRA or full fine-tuning**: Choose between efficient LoRA adapters or full weight updates

## Installation

This project uses `uv` for dependency management.

> **Note**: `unsloth` requires a CUDA-enabled GPU (NVIDIA). It does not support macOS (Apple Silicon).

### Environment Setup

Create a `.env` file in this directory to store your secrets (e.g., WandB API key):

```bash
# .env
WANDB_API_KEY=your_key_here
```

Install dependencies:

```bash
uv sync
```

## Dataset Format

The training script expects a dataset with the following structure:

```
synthetic-dataset/
├── de/                              # German conversations
│   └── Conversation_Name/
│       └── segments/
│           ├── vibevoice-podcast-script.txt
│           ├── 001_speaker1.wav
│           ├── 002_speaker2.wav
│           └── ...
├── en/                              # English conversations
│   └── Conversation_Name/
│       └── segments/
│           ├── vibevoice-podcast-script.txt
│           └── *.wav
└── _meta/                           # Metadata (ignored)
```

### Script Format

The `vibevoice-podcast-script.txt` file should contain one line per segment:

```
[1]: First speaker says this.
[2]: Second speaker responds.
[1]: First speaker continues.
```

The speaker ID in brackets corresponds to `speaker1` or `speaker2` in the audio filenames.

## Usage

### Recommended: Preprocess First (Low RAM)

For large datasets, preprocess first to save RAM during training:

```bash
# Step 1: Preprocess dataset (one-time, processes one conversation at a time)
uv run preprocess_dataset.py \
    --input_path ../synthetic-dataset \
    --output_path ./preprocessed \
    --num_context_turns 3 \
    --max_audio_seconds 10.0

# Step 2: Train on preprocessed data (no audio loading needed!)
uv run train.py \
    --preprocessed_path ./preprocessed \
    --output_dir ./outputs \
    --batch_size 2
```

Benefits of preprocessing:
- **Low RAM during preprocessing**: Only one conversation loaded at a time
- **Very low RAM during training**: No audio in memory, just loads tokenized tensors
- **Faster training start**: No preprocessing delay when training
- **Reusable**: Preprocess once, train multiple times with different hyperparameters

### Alternative: Direct Training (Higher RAM)

Train directly on raw data (loads audio during preprocessing):

```bash
uv run train.py \
    --dataset_path ../synthetic-dataset \
    --output_dir ./outputs \
    --num_context_turns 3
```

### With Config File

```bash
uv run train.py \
    --dataset_path ../synthetic-dataset \
    --output_dir ./outputs \
    --config ./configs/finetune_param_defaults_bs4.yaml \
    --gen_from ./example_gen \
    --use_wandb
```

### Low VRAM Mode

For GPUs with limited memory (8-12GB):

```bash
uv run train.py \
    --dataset_path ../synthetic-dataset \
    --output_dir ./outputs \
    --config ./configs/finetune_param_defaults_lower_vram.yaml \
    --load_in_4bit
```

## Arguments

### Dataset Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | None | Path to raw dataset directory (containing `de/` and `en/` folders) |
| `--preprocessed_path` | None | Path to preprocessed dataset (created by `preprocess_dataset.py`) |
| `--num_context_turns` | 3 | Number of previous conversation turns to include as context |
| `--max_audio_seconds` | 15.0 | Maximum audio length in seconds (longer samples skipped) |
| `--context_length` | 2048 | Maximum context length in tokens |

**Note**: Either `--dataset_path` or `--preprocessed_path` is required. Using `--preprocessed_path` is recommended for lower RAM usage.

### Preprocessing Arguments (preprocess_dataset.py)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_path` | Required | Path to raw dataset |
| `--output_path` | `./preprocessed` | Output directory for tokenized data |
| `--model_id` | `unsloth/csm-1b` | Model ID for processor |
| `--num_context_turns` | 3 | Context turns to include |
| `--max_audio_seconds` | 10.0 | Max audio length per segment |
| `--min_audio_seconds` | 0.5 | Min audio length per segment |
| `--text_max_length` | 256 | Max text tokens |

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `unsloth/csm-1b` | Base model ID from Hugging Face |
| `--load_in_4bit` | False | Load model in 4-bit quantization |
| `--full_finetune` | False | Finetune all weights (disable LoRA) |
| `--lora_r` | 32 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to YAML config file |
| `--output_dir` | `./outputs` | Path to save model/checkpoints |
| `--batch_size` | 2 | Per-device training batch size |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--learning_rate` | 2e-4 | Learning rate |
| `--n_epochs` | 2 | Number of training epochs |
| `--max_steps` | -1 | Max training steps (-1 uses epochs) |
| `--warmup_steps` | 5 | Number of warmup steps |
| `--weight_decay` | 0.001 | Weight decay |
| `--lr_scheduler_type` | `linear` | Learning rate scheduler type |
| `--max_grad_norm` | 1.0 | Maximum gradient norm |
| `--seed` | 3407 | Random seed |

### Logging & Saving Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--save_every` | 500 | Save checkpoint every N steps |
| `--gen_every` | 1000 | Generate audio sample every N steps |
| `--gen_from` | None | Path to example folder for generation context |
| `--use_wandb` | False | Enable Weights & Biases tracking |

## Config Files

Three pre-configured YAML files are provided:

- `finetune_param_defaults.yaml` - Standard settings (batch_size=8)
- `finetune_param_defaults_bs4.yaml` - Moderate VRAM (batch_size=4, grad_acc=2)
- `finetune_param_defaults_lower_vram.yaml` - Low VRAM (batch_size=2, grad_acc=4)

## Context Turns

The `--num_context_turns` argument controls how many previous conversation turns are included when training on each sample. This helps the model learn voice consistency and conversational patterns.

Example with `num_context_turns=2`:

```
Training on segment 5:
  Context: [segment 3 audio + text], [segment 4 audio + text]
  Target: [segment 5 audio + text]
```

More context turns provide better voice consistency but require more memory.

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 8GB VRAM (Tesla T4 or better recommended)
- **OS**: Linux (or WSL on Windows). macOS is **not** supported
- **RAM**: 16GB+ recommended for larger datasets

## Output

The script saves:
- Fine-tuned LoRA adapters (or full model if `--full_finetune`)
- Tokenizer/processor
- Checkpoints every `--save_every` steps
- Generated audio samples every `--gen_every` steps

## Example Generation

The `example_gen/` folder contains sample data for periodic generation during training. Structure:

```
example_gen/
└── segments/
    ├── script.txt           # Same format as vibevoice-podcast-script.txt
    ├── segment_0001_speaker1.wav
    ├── segment_0002_speaker1.wav
    └── ...
```

Generated audio is saved to `output_dir/gen_step_XXXXXX.wav`.
