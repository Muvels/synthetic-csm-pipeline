"""
Sesame CSM-1B Fine-tuning Script

This script fine-tunes the Sesame CSM model using Unsloth for efficiency.
Uses preprocessed data from preprocess_dataset.py for memory efficiency.

Key Features:
- Memory-efficient: loads pre-tokenized data from disk
- Context-aware training: uses data preprocessed with conversation context
- Full CLI control with YAML config support
- Periodic audio generation for monitoring
- LoRA or full fine-tuning options
"""
import unsloth  # Must be first to enable optimizations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from dotenv import load_dotenv
from safetensors.torch import load_file
from unsloth import FastModel, is_bfloat16_supported
from transformers import AutoProcessor, Trainer, TrainingArguments

# Load environment variables
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""
    # Dataset
    preprocessed_path: str = None  # Path to preprocessed data (required)
    
    # Model
    model_id: str = "unsloth/csm-1b"
    load_in_4bit: bool = False
    full_finetune: bool = False
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    
    # Training
    output_dir: str = "./outputs"
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    max_steps: int = -1
    warmup_steps: int = 5
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = 3407
    
    # Logging & Saving
    logging_steps: int = 1
    save_every: int = 500
    gen_every: int = 1000
    gen_from: Optional[str] = None
    use_wandb: bool = False
    
    # Audio settings (for generation callback)
    audio_sampling_rate: int = 24000


# =============================================================================
# Preprocessed Dataset Loading
# =============================================================================

class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-tokenized samples from disk.
    Much more memory efficient than loading raw audio.
    
    Expected structure (created by preprocess_dataset.py):
        preprocessed/
            manifest.json
            conv_id_1/
                00000.safetensors
                00001.safetensors
                ...
    """
    
    def __init__(self, preprocessed_path: Path):
        self.preprocessed_path = Path(preprocessed_path)
        
        # Load manifest
        manifest_path = self.preprocessed_path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Manifest not found at {manifest_path}")
        
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        self.samples = self.manifest["samples"]
        self.config = self.manifest.get("config", {})
        
        print(f"Loaded preprocessed dataset: {len(self.samples)} samples")
        print(f"  Preprocessing config: {self.config}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_path = self.preprocessed_path / sample_info["path"]
        
        # Load tensors from safetensors file
        tensors = load_file(sample_path)
        
        return tensors


def load_preprocessed_dataset(preprocessed_path: Path):
    """
    Load preprocessed dataset.
    Returns (HuggingFace Dataset for Trainer, PyTorch Dataset for data access).
    """
    from datasets import Dataset as HFDataset
    
    # Create PyTorch dataset
    pt_dataset = PreprocessedDataset(preprocessed_path)
    
    # Load bucket info if available
    bucket_path = preprocessed_path / "bucket_info.json"
    if bucket_path.exists():
        with open(bucket_path, "r") as f:
            bucket_info = json.load(f)
        print(f"\nBucket distribution:")
        for num_seg in sorted(bucket_info.keys(), key=int):
            print(f"  {num_seg} audio segment(s): {bucket_info[num_seg]} samples")
    
    # Create metadata list for HF Dataset
    samples_metadata = []
    for i, sample_info in enumerate(pt_dataset.samples):
        samples_metadata.append({
            "idx": i,
            "path": sample_info["path"],
            "num_audio_segments": sample_info["num_audio_segments"],
        })
    
    # Create HF Dataset with just metadata
    hf_dataset = HFDataset.from_list(samples_metadata)
    
    return hf_dataset, pt_dataset


# =============================================================================
# Data Collator
# =============================================================================

def csm_data_collator(features: list) -> dict:
    """
    Collate function for CSM that handles variable-length sequences.
    Pads input_ids, attention_mask, labels, and audio tensors.
    """
    batch = {}
    
    # Get all keys from first feature
    keys = list(features[0].keys())
    
    # Skip metadata keys
    skip_keys = {"num_audio_segments", "idx", "path"}
    
    for key in keys:
        if key in skip_keys:
            continue
        
        values = [f[key] for f in features]
        
        # Handle different tensor types
        if key == "input_ids" or key == "attention_mask":
            # Pad to max length in batch
            max_len = max(v.shape[-1] for v in values)
            padded = []
            for v in values:
                if v.shape[-1] < max_len:
                    pad_len = max_len - v.shape[-1]
                    v = torch.nn.functional.pad(v, (0, pad_len), value=0)
                padded.append(v)
            batch[key] = torch.stack(padded)
            
        elif key == "labels":
            # Pad with -100 (ignore index)
            max_len = max(v.shape[-1] for v in values)
            padded = []
            for v in values:
                if v.shape[-1] < max_len:
                    pad_len = max_len - v.shape[-1]
                    v = torch.nn.functional.pad(v, (0, pad_len), value=-100)
                padded.append(v)
            batch[key] = torch.stack(padded)
            
        elif key == "input_values":
            # Audio tensor - pad with replicate mode to avoid conv issues
            max_h = max(v.shape[-2] for v in values)
            max_w = max(v.shape[-1] for v in values)
            padded = []
            for v in values:
                pad_h = max_h - v.shape[-2]
                pad_w = max_w - v.shape[-1]
                if pad_h > 0 or pad_w > 0:
                    v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), mode='replicate')
                padded.append(v)
            batch[key] = torch.stack(padded)
            
        elif key == "input_values_cutoffs":
            # Cutoffs - pad with last valid value
            max_len = max(v.shape[-1] for v in values)
            padded = []
            for v in values:
                if v.shape[-1] < max_len:
                    pad_len = max_len - v.shape[-1]
                    pad_value = int(v[-1].item()) if len(v) > 0 else 0
                    v = torch.nn.functional.pad(v, (0, pad_len), value=pad_value)
                padded.append(v)
            batch[key] = torch.stack(padded)
            
        else:
            # Stack other tensors directly
            try:
                batch[key] = torch.stack(values)
            except Exception:
                batch[key] = values
    
    return batch


def preprocessed_data_collator(features: list, pt_dataset: PreprocessedDataset) -> dict:
    """
    Data collator for preprocessed dataset.
    Loads actual tensors from disk and pads them.
    """
    # Load actual data from disk
    loaded_features = []
    for f in features:
        idx = f["idx"]
        tensors = pt_dataset[idx]
        loaded_features.append(tensors)
    
    # Use the padding logic from csm_data_collator
    return csm_data_collator(loaded_features)


# =============================================================================
# Generation Callback
# =============================================================================

class GenerationCallback:
    """Callback to generate audio samples during training."""
    
    def __init__(
        self,
        model,
        processor,
        gen_from: str,
        output_dir: str,
        gen_every: int = 1000,
        sampling_rate: int = 24000,
    ):
        self.model = model
        self.processor = processor
        self.gen_from = Path(gen_from)
        self.output_dir = Path(output_dir)
        self.gen_every = gen_every
        self.sampling_rate = sampling_rate
        
        # Load reference data
        self.reference_data = self._load_reference_data()
        if self.reference_data:
            print(f"Generation callback initialized with {len(self.reference_data['lines'])} reference samples")
    
    def _load_reference_data(self):
        """Load reference audio and script for generation."""
        segments_dir = self.gen_from / "segments"
        if not segments_dir.exists():
            segments_dir = self.gen_from
        
        # Find script file
        script_path = segments_dir / "script.txt"
        if not script_path.exists():
            script_path = segments_dir / "vibevoice-podcast-script.txt"
        if not script_path.exists():
            print(f"Warning: No script file found in {segments_dir}")
            return None
        
        # Parse script
        lines = []
        with script_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"\[(\d+)\]:\s*(.*)", line)
                if match:
                    # CSM expects 0-indexed speaker IDs
                    lines.append({
                        "speaker": str(int(match.group(1)) - 1),
                        "text": match.group(2),
                    })
        
        if not lines:
            return None
        
        # Get audio files
        wav_files = sorted(segments_dir.glob("*.wav"))
        
        return {
            "lines": lines,
            "wav_files": wav_files,
            "segments_dir": segments_dir,
        }
    
    def generate_sample(self, step: int):
        """Generate a sample audio and save it."""
        if self.reference_data is None:
            return
        
        try:
            lines = self.reference_data["lines"]
            wav_files = self.reference_data["wav_files"]
            
            # Build conversation with available context
            conversation = []
            num_context = min(3, len(wav_files) - 1, len(lines) - 1)
            
            for i in range(num_context):
                if i < len(wav_files) and i < len(lines):
                    audio, sr = sf.read(wav_files[i])
                    if sr != self.sampling_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
                    
                    conversation.append({
                        "role": lines[i]["speaker"],
                        "content": [
                            {"type": "text", "text": lines[i]["text"]},
                            {"type": "audio", "path": audio.astype(np.float32)},
                        ],
                    })
            
            # Add target text (no audio - we'll generate it)
            target_idx = num_context
            if target_idx < len(lines):
                conversation.append({
                    "role": lines[target_idx]["speaker"],
                    "content": [
                        {"type": "text", "text": lines[target_idx]["text"]},
                    ],
                })
            
            # Generate
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                common_kwargs={"return_tensors": "pt"},
            )
            
            # Convert lists to tensors if needed
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, list):
                    processed_inputs[k] = torch.tensor(v).unsqueeze(0)
                elif isinstance(v, torch.Tensor):
                    if v.dim() == 1:
                        processed_inputs[k] = v.unsqueeze(0)
                    else:
                        processed_inputs[k] = v
                elif isinstance(v, np.ndarray):
                    processed_inputs[k] = torch.from_numpy(v).unsqueeze(0)
                else:
                    processed_inputs[k] = v
            
            # Move to device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in processed_inputs.items()}
            
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=125,  # ~10 seconds
                    output_audio=True,
                )
            
            # Save audio
            if audio_values is not None:
                audio_np = audio_values.cpu().numpy().squeeze()
                output_path = self.output_dir / f"gen_step_{step:06d}.wav"
                sf.write(output_path, audio_np, self.sampling_rate)
                print(f"\nGenerated audio saved to: {output_path}")
                
        except Exception as e:
            print(f"Warning: Generation failed at step {step}: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Model Setup
# =============================================================================

def setup_model(config: TrainConfig):
    """Setup model with LoRA or full finetuning."""
    print(f"\nLoading model: {config.model_id}")
    
    model, processor = FastModel.from_pretrained(
        config.model_id,
        load_in_4bit=config.load_in_4bit,
    )
    
    if not config.full_finetune:
        print("Setting up LoRA adapters...")
        model = FastModel.get_peft_model(
            model,
            finetune_audio_encoder=True,
            finetune_decoder=True,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )
    
    return model, processor


# =============================================================================
# Training
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""
    print("=" * 60)
    print("Sesame CSM Fine-tuning")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model, processor = setup_model(config)
    
    # Load preprocessed dataset
    print(f"\nLoading preprocessed dataset from: {config.preprocessed_path}")
    train_dataset, pt_dataset = load_preprocessed_dataset(Path(config.preprocessed_path))
    print(f"\nDataset ready: {len(train_dataset)} samples")
    
    # Setup generation callback
    gen_callback = None
    if config.gen_from:
        gen_callback = GenerationCallback(
            model=model,
            processor=processor,
            gen_from=config.gen_from,
            output_dir=str(output_dir),
            gen_every=config.gen_every,
            sampling_rate=config.audio_sampling_rate,
        )
    
    # Setup WandB if enabled
    report_to = "none"
    if config.use_wandb:
        wandb_key = os.getenv("WANDB_API_KEY")
        if wandb_key:
            import wandb
            wandb.login(key=wandb_key)
            report_to = "wandb"
        else:
            print("Warning: --use_wandb set but WANDB_API_KEY not found in environment")
    
    # Training arguments
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps,
        "learning_rate": config.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": config.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "save_steps": config.save_every,
        "save_total_limit": 3,
        "report_to": report_to,
        "remove_unused_columns": False,
    }
    
    # Set either max_steps or num_train_epochs
    if config.max_steps > 0:
        training_args_dict["max_steps"] = config.max_steps
        training_args_dict["num_train_epochs"] = None
    else:
        training_args_dict["num_train_epochs"] = config.num_train_epochs
        training_args_dict["max_steps"] = None
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Setup callbacks
    from transformers import TrainerCallback
    
    class AudioGenCallback(TrainerCallback):
        def __init__(self, gen_callback):
            self.gen_callback = gen_callback
        
        def on_step_end(self, args, state, control, **kwargs):
            if self.gen_callback and state.global_step % self.gen_callback.gen_every == 0:
                self.gen_callback.generate_sample(state.global_step)
    
    callbacks = []
    if gen_callback:
        callbacks.append(AudioGenCallback(gen_callback))
    
    # Create data collator
    data_collator = partial(preprocessed_data_collator, pt_dataset=pt_dataset)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Show memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved before training.")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer_stats = trainer.train()
    
    # Show final stats
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"\n{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    
    print("\nTraining complete!")
    return trainer_stats


# =============================================================================
# CLI
# =============================================================================

def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> TrainConfig:
    """Parse command line arguments and return config."""
    defaults = TrainConfig()
    
    parser = argparse.ArgumentParser(
        description="Fine-tune Sesame CSM-1B with preprocessed data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file.",
    )
    
    # Dataset (required)
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        required=True,
        help="Path to preprocessed dataset (created by preprocess_dataset.py).",
    )
    
    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default=defaults.model_id,
        help="Model ID to load.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=defaults.load_in_4bit,
        help="Load model in 4-bit quantization.",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        default=defaults.full_finetune,
        help="Finetune all weights (disable LoRA).",
    )
    
    # LoRA
    parser.add_argument(
        "--lora_r",
        type=int,
        default=defaults.lora_r,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=defaults.lora_alpha,
        help="LoRA alpha.",
    )
    
    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default=defaults.output_dir,
        help="Path to save the trained model/checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=defaults.batch_size,
        help="Per device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=defaults.learning_rate,
        help="Learning rate.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=defaults.num_train_epochs,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=defaults.max_steps,
        help="Max training steps. Set to -1 to use epochs.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=defaults.warmup_steps,
        help="Warmup steps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=defaults.weight_decay,
        help="Weight decay.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=defaults.lr_scheduler_type,
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=defaults.max_grad_norm,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed.",
    )
    
    # Logging & Saving
    parser.add_argument(
        "--save_every",
        type=int,
        default=defaults.save_every,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--gen_every",
        type=int,
        default=defaults.gen_every,
        help="Generate audio every N steps.",
    )
    parser.add_argument(
        "--gen_from",
        type=str,
        default=defaults.gen_from,
        help="Path to example folder for generation.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=defaults.use_wandb,
        help="Enable Weights & Biases tracking.",
    )
    
    args = parser.parse_args()
    
    # Build config dict from defaults
    config_dict = {
        "preprocessed_path": args.preprocessed_path,
        "model_id": defaults.model_id,
        "load_in_4bit": defaults.load_in_4bit,
        "full_finetune": defaults.full_finetune,
        "lora_r": defaults.lora_r,
        "lora_alpha": defaults.lora_alpha,
        "lora_dropout": defaults.lora_dropout,
        "output_dir": defaults.output_dir,
        "batch_size": defaults.batch_size,
        "gradient_accumulation_steps": defaults.gradient_accumulation_steps,
        "learning_rate": defaults.learning_rate,
        "num_train_epochs": defaults.num_train_epochs,
        "max_steps": defaults.max_steps,
        "warmup_steps": defaults.warmup_steps,
        "weight_decay": defaults.weight_decay,
        "lr_scheduler_type": defaults.lr_scheduler_type,
        "max_grad_norm": defaults.max_grad_norm,
        "seed": defaults.seed,
        "logging_steps": defaults.logging_steps,
        "save_every": defaults.save_every,
        "gen_every": defaults.gen_every,
        "gen_from": defaults.gen_from,
        "use_wandb": defaults.use_wandb,
        "audio_sampling_rate": defaults.audio_sampling_rate,
    }
    
    # Load from YAML config if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        yaml_mapping = {
            "batch_size": "batch_size",
            "grad_acc_steps": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "max_grad_norm": "max_grad_norm",
            "warmup_steps": "warmup_steps",
            "weight_decay": "weight_decay",
            "lr_decay": "lr_scheduler_type",
            "model_id": "model_id",
            "output_dir": "output_dir",
            "max_steps": "max_steps",
            "num_train_epochs": "num_train_epochs",
            "seed": "seed",
            "save_every": "save_every",
            "gen_every": "gen_every",
            "gen_from": "gen_from",
            "lora_r": "lora_r",
            "lora_alpha": "lora_alpha",
        }
        for yaml_key, config_key in yaml_mapping.items():
            if yaml_key in yaml_config:
                config_dict[config_key] = yaml_config[yaml_key]
    
    # Override with CLI arguments
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key == "config":
            continue
        if key == "n_epochs":
            key = "num_train_epochs"
        if value is not None and key in config_dict:
            config_dict[key] = value
    
    return TrainConfig(**config_dict)


def main():
    """Main entry point."""
    config = parse_args()
    
    # Print configuration
    print("\nConfiguration:")
    print("-" * 40)
    for key, value in sorted(vars(config).items()):
        print(f"  {key}: {value}")
    print("-" * 40)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
