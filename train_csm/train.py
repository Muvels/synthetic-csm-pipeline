"""
Sesame CSM-1B Fine-tuning Script with Memory-Efficient Dataset Loading

This script fine-tunes the Sesame CSM model using Unsloth for efficiency.
Unlike the notebook approach that loads all samples into RAM, this uses
lazy loading with PyTorch IterableDataset for memory efficiency.

Key Features:
- Lazy loading of audio samples (only loads when needed)
- Configurable context: include last N sentences before current sentence
- Full CLI control with YAML config support
- Periodic audio generation for monitoring
- LoRA or full fine-tuning options
"""
import unsloth  # Must be first to enable optimizations
import argparse
import gc
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from dotenv import load_dotenv
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
    dataset_path: Optional[str] = None
    preprocessed_path: Optional[str] = None  # Path to preprocessed data (use instead of dataset_path)
    context_length: int = 2048  # Max context length in tokens
    num_context_turns: int = 3  # Number of previous turns to include as context
    max_audio_seconds: float = 15.0  # Maximum audio length in seconds
    
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
    batch_size: int = 2  # BucketedTrainer handles variable context; increase for faster training
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
    
    # Preprocessing - CSM expects 24kHz audio
    target_sampling_rate: int = 24000
    audio_model_sampling_rate: int = 24000  # CSM expects 24kHz
    text_max_length: int = 256


# =============================================================================
# Dataset Loading - Uses HuggingFace datasets like the notebook
# =============================================================================

def scan_dataset_for_examples(dataset_path: Path) -> list:
    """
    Scan dataset directory and collect all examples.
    Returns list of dicts with 'audio', 'text', 'source' keys.
    
    Handles two directory structures:
    1. Flat: dataset/uuid_folder/segments/...
    2. Nested: dataset/lang/conversation_name/segments/...
    """
    examples = []
    
    def process_conversation_dir(conv_dir: Path):
        """Process a single conversation directory."""
        segments_dir = conv_dir / "segments"
        if not segments_dir.is_dir():
            return
        
        script_path = segments_dir / "vibevoice-podcast-script.txt"
        if not script_path.is_file():
            return
        
        # Parse script lines
        script_lines = []
        with script_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"\[(\d+)\]:\s*(.*)", line)
                if match:
                    # CSM expects 0-indexed speaker IDs
                    speaker_id = str(int(match.group(1)) - 1)
                    text = match.group(2)
                    script_lines.append({"speaker": speaker_id, "text": text})
        
        if not script_lines:
            return
        
        # Match with audio files
        wav_files = sorted(segments_dir.glob("*.wav"))
        
        for wav_path, line in zip(wav_files, script_lines):
            examples.append({
                "audio": str(wav_path),
                "text": line["text"],
                "source": line["speaker"],
            })
    
    # Try both flat and nested structures
    for item in sorted(dataset_path.iterdir()):
        if not item.is_dir() or item.name.startswith("_"):
            continue
        
        # Check if this is a conversation dir (has segments/)
        segments_dir = item / "segments"
        if segments_dir.is_dir():
            process_conversation_dir(item)
        else:
            # Try nested structure
            for conv_dir in sorted(item.iterdir()):
                if conv_dir.is_dir():
                    process_conversation_dir(conv_dir)
    
    return examples


def create_hf_dataset(
    examples: list,
    config: TrainConfig,
    processor: AutoProcessor,
) -> "HFDataset":
    """
    Create HuggingFace dataset from examples, following the notebook approach.
    Uses the Audio feature for proper audio loading and resampling.
    """
    from datasets import Audio, Dataset as HFDataset
    
    print(f"Total examples before filtering: {len(examples)}")
    
    # Create HF Dataset
    raw_ds = HFDataset.from_list(examples)

    # Filter out long/corrupted audio
    max_audio_samples = int(config.audio_model_sampling_rate * config.max_audio_seconds)
    
    def filter_audio(example):
        try:
            audio_path = example["audio"]["path"]
            audio_info = sf.info(audio_path)
            # Check duration at original sample rate
            duration = audio_info.frames / audio_info.samplerate
            return duration <= config.max_audio_seconds
        except Exception as e:
            print(f"Warning: Skipping {example['audio'].get('path', 'unknown')}: {e}")
            return False
    
    raw_ds = raw_ds.filter(
        filter_audio,
        desc=f"Filtering audio > {config.max_audio_seconds}s",
    )
    
    print(f"After filtering: {len(raw_ds)} examples")
    
    # Preprocessing function - matches notebook exactly
    def preprocess_example(example):
        audio_array = example["audio"]["array"]
        
        conversation = [
            {
                "role": str(example["source"]),
                "content": [
                    {"type": "text", "text": example["text"]},
                    {"type": "audio", "path": audio_array},
                ],
            }
        ]
        
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length",
                "max_length": config.text_max_length,
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": config.audio_model_sampling_rate,
                "max_length": max_audio_samples,
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )
        
        required_keys = [
            "input_ids",
            "attention_mask",
            "labels",
            "input_values",
            "input_values_cutoffs",
        ]
        
        processed = {}
        for key in required_keys:
            if key not in model_inputs:
                raise ValueError(
                    f"Required key '{key}' not in processor output. "
                    f"Got keys: {list(model_inputs.keys())}"
                )
            processed[key] = model_inputs[key][0]  # Remove batch dim
        
        return processed
    
    # Map preprocessing
    processed_ds = raw_ds.map(
        preprocess_example,
        remove_columns=raw_ds.column_names,
        desc="Preprocessing dataset",
    )
    
    print(f"Preprocessed dataset: {len(processed_ds)} examples")
    
    return processed_ds


def create_hf_dataset_with_context(
    examples: list,
    config: TrainConfig,
    processor: AutoProcessor,
) -> "HFDataset":
    """
    Create HuggingFace dataset with context from previous turns.
    Groups examples by conversation and includes previous turns as context.
    """
    from datasets import Audio, Dataset as HFDataset
    
    print(f"Total examples before filtering: {len(examples)}")
    
    # Group examples by conversation (based on audio path directory)
    conversations = {}
    for ex in examples:
        audio_path = Path(ex["audio"])
        conv_id = str(audio_path.parent.parent)  # segments -> conv_dir
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(ex)
    
    # Sort each conversation by audio filename (which includes order number)
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x["audio"])
    
    print(f"Found {len(conversations)} conversations")
    
    # Create a mapping from audio path to index for context lookup
    audio_to_idx = {ex["audio"]: i for i, ex in enumerate(examples)}
    
    # Minimum audio length to ensure conv layers have enough input
    min_audio_seconds = 0.5
    
    # Filter examples by audio length BEFORE creating HF Dataset
    # This is more reliable than filtering after cast_column
    print(f"Filtering audio ({min_audio_seconds}s - {config.max_audio_seconds}s)...")
    filtered_examples = []
    for ex in examples:
        try:
            audio_info = sf.info(ex["audio"])
            duration = audio_info.frames / audio_info.samplerate
            if min_audio_seconds <= duration <= config.max_audio_seconds:
                filtered_examples.append(ex)
        except Exception:
            pass  # Skip files that can't be read
    
    print(f"After filtering: {len(filtered_examples)} / {len(examples)} examples")
    
    if len(filtered_examples) == 0:
        print("ERROR: No examples passed the audio length filter!")
        return HFDataset.from_list([])
    
    # Update examples and mappings with filtered data
    examples = filtered_examples
    audio_to_idx = {ex["audio"]: i for i, ex in enumerate(examples)}
    
    # Update conversations with filtered examples
    conversations = {}
    for ex in examples:
        audio_path = Path(ex["audio"])
        conv_id = str(audio_path.parent.parent)
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(ex)
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x["audio"])
    
    # Create HF Dataset and cast audio column to Audio feature
    # This will load audio and resample to target sample rate
    raw_ds = HFDataset.from_list(examples)
    raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=config.audio_model_sampling_rate))
    
    # Verify Audio feature is working
    first_ex = raw_ds[0]
    print(f"Audio feature check: type={type(first_ex['audio'])}")
    if isinstance(first_ex['audio'], dict):
        print(f"  Keys: {list(first_ex['audio'].keys())}")
        print(f"  Array shape: {first_ex['audio']['array'].shape}")
    
    # Calculate max audio samples for context
    # With N context turns + 1 current, we need (N+1) * max_audio_seconds of audio space
    max_total_audio_seconds = (config.num_context_turns + 1) * config.max_audio_seconds
    max_audio_samples = int(config.audio_model_sampling_rate * max_total_audio_seconds)
    print(f"Max audio samples for {config.num_context_turns + 1} turns: {max_audio_samples} ({max_total_audio_seconds}s)")
    
    # Preprocess with context (variable amount, 0 to N turns)
    def preprocess_with_context(example, idx):
        # Debug first example
        if idx == 0:
            print(f"\nDebug first example:")
            print(f"  Audio keys: {example['audio'].keys() if isinstance(example['audio'], dict) else type(example['audio'])}")
            if isinstance(example['audio'], dict) and 'array' in example['audio']:
                arr = example['audio']['array']
                print(f"  Audio array shape: {arr.shape if hasattr(arr, 'shape') else len(arr)}")
                print(f"  Audio array dtype: {arr.dtype if hasattr(arr, 'dtype') else type(arr)}")
        
        audio_path = example["audio"]["path"]
        conv_id = str(Path(audio_path).parent.parent)
        conv_examples = conversations.get(conv_id, [])
        
        # Find current position in conversation
        current_pos = next(
            (i for i, ex in enumerate(conv_examples) if ex["audio"] == audio_path),
            0
        )
        
        # Get context examples (up to N previous turns, may be fewer)
        context_start = max(0, current_pos - config.num_context_turns)
        context_examples = conv_examples[context_start:current_pos]
        
        # Build conversation with context
        conversation = []
        
        # Add context turns (need to load their audio from the dataset)
        for ctx_ex in context_examples:
            ctx_idx = audio_to_idx.get(ctx_ex["audio"])
            if ctx_idx is not None and ctx_idx < len(raw_ds):
                try:
                    ctx_data = raw_ds[ctx_idx]
                    ctx_audio = ctx_data["audio"]["array"]
                    conversation.append({
                        "role": str(ctx_ex["source"]),
                        "content": [
                            {"type": "text", "text": ctx_ex["text"]},
                            {"type": "audio", "path": ctx_audio},
                        ],
                    })
                except Exception as e:
                    if idx == 0:
                        print(f"  Warning: Failed to load context: {e}")
        
        # Get audio array
        audio_array = example["audio"]["array"]
        
        # Add current turn
        conversation.append({
            "role": str(example["source"]),
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": audio_array},
            ],
        })
        
        if idx == 0:
            print(f"  Conversation has {len(conversation)} turns (0-{config.num_context_turns} context + 1 current)")
            print(f"  Last turn audio type: {type(audio_array)}")
        
        # Process
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length",
                "max_length": config.text_max_length,
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": config.audio_model_sampling_rate,
                "max_length": max_audio_samples,
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )
        
        if idx == 0:
            print(f"  Processor returned keys: {list(model_inputs.keys())}")
            for k, v in model_inputs.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}")
        
        required_keys = [
            "input_ids",
            "attention_mask",
            "labels",
            "input_values",
            "input_values_cutoffs",
        ]
        
        processed = {}
        for key in required_keys:
            if key not in model_inputs:
                raise ValueError(f"Missing key: {key}. Got: {list(model_inputs.keys())}")
            processed[key] = model_inputs[key][0]
        
        # Store number of turns for proper padding later
        processed["num_audio_segments"] = len(conversation)
        
        return processed
    
    # Map with index for context lookup
    processed_ds = raw_ds.map(
        preprocess_with_context,
        with_indices=True,
        remove_columns=raw_ds.column_names,
        desc="Preprocessing with context",
    )
    
    print(f"Preprocessed dataset: {len(processed_ds)} examples")
    return processed_ds


# =============================================================================
# Preprocessed Dataset Loading (Memory Efficient)
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
            conv_id_2/
                ...
    """
    
    def __init__(self, preprocessed_path: Path):
        from safetensors.torch import load_file
        import json
        
        self.preprocessed_path = Path(preprocessed_path)
        self.load_file = load_file
        
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
        tensors = self.load_file(sample_path)
        
        return tensors


def load_preprocessed_dataset(preprocessed_path: Path, config: TrainConfig):
    """
    Load preprocessed dataset and wrap in HuggingFace Dataset for Trainer compatibility.
    """
    from datasets import Dataset as HFDataset
    
    # Create PyTorch dataset
    pt_dataset = PreprocessedDataset(preprocessed_path)
    
    # Load bucket info if available
    bucket_path = preprocessed_path / "bucket_info.json"
    if bucket_path.exists():
        import json
        with open(bucket_path, "r") as f:
            bucket_info = json.load(f)
        print(f"Bucket distribution:")
        for num_seg in sorted(bucket_info.keys(), key=int):
            print(f"  {num_seg} audio segment(s): {bucket_info[num_seg]} samples")
    
    # Convert to list of dicts for HF Dataset
    # This is more memory efficient than loading all at once because
    # we use the PyTorch dataset for actual data loading during training
    samples_metadata = []
    for sample_info in pt_dataset.samples:
        samples_metadata.append({
            "idx": len(samples_metadata),
            "path": sample_info["path"],
            "num_audio_segments": sample_info["num_audio_segments"],
        })
    
    # Create a simple HF Dataset that stores just metadata
    # The actual loading happens in the collator
    hf_dataset = HFDataset.from_list(samples_metadata)
    
    # Store reference to PyTorch dataset for actual data access
    hf_dataset.pt_dataset = pt_dataset
    
    return hf_dataset, pt_dataset


def preprocessed_data_collator(features: list, pt_dataset: "PreprocessedDataset") -> dict:
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
    
    # Use the same padding logic as csm_data_collator
    return csm_data_collator(loaded_features)


# =============================================================================
# Bucketed Batch Sampler for CSM
# =============================================================================

class BucketBatchSampler:
    """
    Batch sampler that groups examples by number of audio segments.
    This ensures all examples in a batch have the same input_values_cutoffs length,
    avoiding padding issues with the audio encoder.
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Group indices by number of audio segments
        self.buckets = {}
        for idx in range(len(dataset)):
            example = dataset[idx]
            # Get the number of audio segments from input_values_cutoffs
            if "input_values_cutoffs" in example:
                cutoffs = example["input_values_cutoffs"]
                if hasattr(cutoffs, '__len__'):
                    num_segments = len(cutoffs)
                else:
                    num_segments = 1
            elif "num_audio_segments" in example:
                num_segments = int(example["num_audio_segments"])
            else:
                num_segments = 1
            
            if num_segments not in self.buckets:
                self.buckets[num_segments] = []
            self.buckets[num_segments].append(idx)
        
        # Print bucket statistics
        print(f"\nBucketed batching enabled:")
        for num_seg, indices in sorted(self.buckets.items()):
            print(f"  {num_seg} audio segment(s): {len(indices)} examples")
        
        # Pre-compute batches for each bucket
        self._create_batches()
    
    def _create_batches(self):
        """Create batches from buckets."""
        import random
        
        rng = random.Random(self.seed + self.epoch)
        self.batches = []
        
        for num_segments, indices in self.buckets.items():
            bucket_indices = indices.copy()
            if self.shuffle:
                rng.shuffle(bucket_indices)
            
            # Create batches from this bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                self.batches.append(batch)
        
        # Shuffle the order of batches (not the contents)
        if self.shuffle:
            rng.shuffle(self.batches)
    
    def __iter__(self):
        self._create_batches()  # Recreate batches each epoch for fresh shuffle
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling."""
        self.epoch = epoch


# =============================================================================
# Data Collator for CSM
# =============================================================================

_collator_debug_count = [0]  # Use list to allow modification in nested function

def csm_data_collator(features: list) -> dict:
    """
    Custom data collator that pads sequences to the same length within a batch.
    For audio (input_values), uses edge padding instead of zero padding to avoid
    issues with the audio encoder's conv layers.
    """
    if not features:
        return {}
    
    # Debug first few batches
    if _collator_debug_count[0] < 2:
        print(f"\nCollator debug (batch {_collator_debug_count[0]}): {len(features)} samples in batch")
        for i, f in enumerate(features):
            # Get num_audio_segments for compact display
            num_seg = f.get("num_audio_segments", "?")
            if hasattr(num_seg, 'item'):
                num_seg = num_seg.item()
            input_ids_len = f["input_ids"].shape[0] if "input_ids" in f else "?"
            print(f"  Sample {i}: {input_ids_len} tokens, {num_seg} audio segments")
        _collator_debug_count[0] += 1
    
    batch = {}
    
    # Get all keys from first feature (exclude metadata keys)
    keys = [k for k in features[0].keys() if k != "num_audio_segments"]
    
    for key in keys:
        values = [f[key] for f in features]
        
        # Skip None values
        if values[0] is None:
            continue
        
        # Check if these are tensors that need stacking
        if isinstance(values[0], torch.Tensor):
            shapes = [v.shape for v in values]
            
            if all(s == shapes[0] for s in shapes):
                # Same shape - just stack
                batch[key] = torch.stack(values)
            else:
                # Different shapes - need to pad
                max_dims = [max(s[dim] for s in shapes) for dim in range(len(shapes[0]))]
                
                padded = []
                for v in values:
                    if len(v.shape) == 1:
                        pad_size = max_dims[0] - v.shape[0]
                        if pad_size > 0:
                            if key == "labels":
                                pad_value = -100  # Ignored in loss
                            elif key == "input_values_cutoffs":
                                # Pad with the last cutoff value
                                pad_value = int(v[-1].item()) if len(v) > 0 else 0
                            elif key == "attention_mask":
                                pad_value = 0
                            else:
                                pad_value = 0
                            v = torch.nn.functional.pad(v, (0, pad_size), value=pad_value)
                    elif len(v.shape) == 2:
                        pad_h = max_dims[0] - v.shape[0]
                        pad_w = max_dims[1] - v.shape[1]
                        if pad_h > 0 or pad_w > 0:
                            if key == "input_values":
                                # For audio: use 'replicate' mode (repeat edge values)
                                # This avoids issues with zero padding in audio encoder
                                v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), mode='replicate')
                            else:
                                v = torch.nn.functional.pad(v, (0, pad_w, 0, pad_h), value=0)
                    padded.append(v)
                
                batch[key] = torch.stack(padded)
        
        elif isinstance(values[0], np.ndarray):
            shapes = [v.shape for v in values]
            
            if all(s == shapes[0] for s in shapes):
                batch[key] = torch.from_numpy(np.stack(values))
            else:
                max_dims = [max(s[dim] for s in shapes) for dim in range(len(shapes[0]))]
                
                padded = []
                for v in values:
                    if len(v.shape) == 1:
                        pad_size = max_dims[0] - v.shape[0]
                        if pad_size > 0:
                            if key == "labels":
                                pad_value = -100
                            elif key == "input_values_cutoffs":
                                pad_value = int(v[-1]) if len(v) > 0 else 0
                            else:
                                pad_value = 0
                            v = np.pad(v, (0, pad_size), constant_values=pad_value)
                    elif len(v.shape) == 2:
                        pad_h = max_dims[0] - v.shape[0]
                        pad_w = max_dims[1] - v.shape[1]
                        if pad_h > 0 or pad_w > 0:
                            if key == "input_values":
                                # For audio: use 'edge' mode (repeat edge values)
                                v = np.pad(v, ((0, pad_h), (0, pad_w)), mode='edge')
                            else:
                                v = np.pad(v, ((0, pad_h), (0, pad_w)), constant_values=0)
                    padded.append(v)
                
                batch[key] = torch.from_numpy(np.stack(padded))
        
        elif isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values)
        else:
            # Non-tensor values - skip
            pass
    
    return batch


# =============================================================================
# Custom Trainer with Bucketed Batching
# =============================================================================

class BucketedTrainer(Trainer):
    """
    Custom Trainer that uses bucketed batching to group examples
    by number of audio segments, avoiding padding issues.
    """
    
    def __init__(self, *args, bucket_batch_size: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_batch_size = bucket_batch_size
        self._bucket_sampler = None
    
    def get_train_dataloader(self):
        """Override to use BucketBatchSampler."""
        from torch.utils.data import DataLoader
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # Create bucket sampler
        self._bucket_sampler = BucketBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.bucket_batch_size,
            shuffle=True,
            seed=self.args.seed,
        )
        
        # Create DataLoader with our batch sampler
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._bucket_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _get_train_sampler(self):
        """Disable default sampler since we use batch_sampler."""
        return None


# =============================================================================
# Generation Callback
# =============================================================================

class GenerationCallback:
    """Callback for periodic audio generation during training."""
    
    def __init__(
        self,
        model,
        processor: AutoProcessor,
        gen_from: Optional[str],
        output_dir: str,
        gen_every: int,
    ):
        self.model = model
        self.processor = processor
        self.gen_from = Path(gen_from) if gen_from else None
        self.output_dir = Path(output_dir)
        self.gen_every = gen_every
        self.step = 0
        
        # Load reference data if provided
        self.reference_data = None
        if self.gen_from and self.gen_from.exists():
            self.reference_data = self._load_reference()
    
    def _load_reference(self) -> Optional[dict]:
        """Load reference script and audio for generation."""
        segments_dir = self.gen_from / "segments"
        script_path = segments_dir / "script.txt"
        
        if not script_path.exists():
            # Try vibevoice format
            script_path = segments_dir / "vibevoice-podcast-script.txt"
        
        if not script_path.exists():
            print(f"Warning: No script found in {self.gen_from}")
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
            "audio_files": wav_files[:len(lines)],
        }
    
    def generate_sample(self, step: int):
        """Generate a sample audio and save it."""
        if self.reference_data is None:
            return
        
        try:
            lines = self.reference_data["lines"]
            audio_files = self.reference_data["audio_files"]
            
            # Build conversation with context
            conversation = []
            
            # Use first few turns as context
            for i in range(min(3, len(lines) - 1)):
                if i < len(audio_files):
                    audio, _ = sf.read(str(audio_files[i]))
                    conversation.append({
                        "role": str(lines[i]["speaker"]),
                        "content": [
                            {"type": "text", "text": lines[i]["text"]},
                            {"type": "audio", "path": audio},
                        ],
                    })
            
            # Add target turn (text only, for generation)
            target_idx = min(3, len(lines) - 1)
            conversation.append({
                "role": str(lines[target_idx]["speaker"]),
                "content": [
                    {"type": "text", "text": lines[target_idx]["text"]},
                ],
            })
            
            # Generate - process inputs
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
                    # Convert list to tensor
                    processed_inputs[k] = torch.tensor(v).unsqueeze(0)  # Add batch dim
                elif isinstance(v, torch.Tensor):
                    if v.dim() == 1:
                        processed_inputs[k] = v.unsqueeze(0)  # Add batch dim
                    else:
                        processed_inputs[k] = v
                elif isinstance(v, np.ndarray):
                    processed_inputs[k] = torch.from_numpy(v).unsqueeze(0)
                else:
                    processed_inputs[k] = v
            
            # Move all tensors to device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in processed_inputs.items()}
            
            # Debug: show input structure
            print(f"\nGeneration inputs at step {step}:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  {k}: type={type(v)}")
            
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=125,  # ~10 seconds
                    output_audio=True,
                )
            
            # Save audio
            audio = audio_values[0].to(torch.float32).cpu().numpy()
            output_path = self.output_dir / f"gen_step_{step:06d}.wav"
            sf.write(str(output_path), audio, 24000)
            print(f"Generated sample saved to {output_path}")
            
        except Exception as e:
            import traceback
            print(f"Warning: Generation failed at step {step}: {e}")
            traceback.print_exc()


# =============================================================================
# Training
# =============================================================================

def setup_model(config: TrainConfig):
    """Load and configure the model with LoRA."""
    from transformers import CsmForConditionalGeneration
    
    print(f"Loading model: {config.model_id}")
    
    model, processor = FastModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.context_length,
        dtype=None,  # Auto-detect
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=config.load_in_4bit,
    )
    
    if not config.full_finetune:
        print("Setting up LoRA adapters...")
        model = FastModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
            use_rslora=False,
            loftq_config=None,
        )
    
    return model, processor


def train(config: TrainConfig):
    """Main training function."""
    print("=" * 60)
    print("Sesame CSM Fine-tuning")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup model
    model, model_processor = setup_model(config)
    
    # Track if we're using preprocessed data
    use_preprocessed = config.preprocessed_path is not None
    pt_dataset = None  # Reference to PyTorch dataset for preprocessed data
    
    if use_preprocessed:
        # Use preprocessed data (no audio loading needed!)
        print(f"\nLoading preprocessed dataset from: {config.preprocessed_path}")
        train_dataset, pt_dataset = load_preprocessed_dataset(
            Path(config.preprocessed_path), config
        )
        print(f"\nDataset ready: {len(train_dataset)} samples (preprocessed)")
    else:
        # Load processor separately for dataset prep (like notebook does in cell 12)
        # This ensures we get the full processor, not a potentially modified one from FastModel
        print(f"\nLoading processor for dataset prep...")
        dataset_processor = AutoProcessor.from_pretrained(config.model_id)
        
        # Setup dataset using HuggingFace datasets (like the notebook)
        print(f"\nLoading dataset from: {config.dataset_path}")
        
        # Scan for examples
        examples = scan_dataset_for_examples(Path(config.dataset_path))
        
        if len(examples) == 0:
            print("\nERROR: No examples found in dataset!")
            print("Expected structure: dataset_path/[lang/]conversation_id/segments/")
            print("Each segments folder should have vibevoice-podcast-script.txt and .wav files")
            sys.exit(1)
        
        # Create dataset - use context if num_context_turns > 0
        if config.num_context_turns > 0:
            print(f"\nCreating dataset with {config.num_context_turns} context turns...")
            train_dataset = create_hf_dataset_with_context(examples, config, dataset_processor)
        else:
            print("\nCreating dataset without context (single turns)...")
            train_dataset = create_hf_dataset(examples, config, dataset_processor)
        
        # Shuffle the dataset to mix examples from different conversations
        train_dataset = train_dataset.shuffle(seed=config.seed)
        print(f"Dataset shuffled with seed {config.seed}")
        
        # Set format for PyTorch
        train_dataset.set_format("torch")
        
        # Free memory
        del examples
        gc.collect()
        
        print(f"\nDataset ready: {len(train_dataset)} samples (shuffled)")
    
    # Setup generation callback
    gen_callback = None
    if config.gen_from:
        gen_callback = GenerationCallback(
            model=model,
            processor=model_processor,
            gen_from=config.gen_from,
            output_dir=str(output_dir),
            gen_every=config.gen_every,
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
    # Note: Unsloth doesn't like None values, so we use -1 for max_steps to indicate "use epochs"
    # 
    # IMPORTANT: CSM requires samples with the same number of audio segments to be batched together.
    # This is because input_values_cutoffs defines how the audio is split, and different lengths
    # cause issues with the audio encoder's conv1d layers when padded.
    # 
    # When batch_size > 1, we use BucketedTrainer which groups samples by num_audio_segments.
    # The actual per_device_train_batch_size is set to 1 since BucketedTrainer handles batching.
    use_bucketed_trainer = config.batch_size > 1
    effective_batch_size = 1 if use_bucketed_trainer else config.batch_size
    
    training_args_dict = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": effective_batch_size,
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
    
    # Set either max_steps or num_train_epochs (not both as None)
    if config.max_steps > 0:
        training_args_dict["max_steps"] = config.max_steps
    else:
        training_args_dict["num_train_epochs"] = config.num_train_epochs
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Create custom callback for generation
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
    
    # Select data collator
    if use_preprocessed and pt_dataset is not None:
        # Preprocessed data: use collator that loads from disk
        from functools import partial
        data_collator = partial(preprocessed_data_collator, pt_dataset=pt_dataset)
        print(f"\nUsing preprocessed data collator (loads from disk)")
    else:
        data_collator = csm_data_collator
    
    # Create trainer - use BucketedTrainer for batch_size > 1
    # This ensures samples with the same number of audio segments are batched together,
    # which is required by CSM's audio encoder (input_values_cutoffs must match within batch)
    if use_bucketed_trainer:
        print(f"\nUsing BucketedTrainer with batch_size={config.batch_size} (bucketed by audio segments)")
        print(f"  NOTE: Unsloth banner will show 'Batch size per device = 1' but actual batch size is {config.batch_size}")
        print(f"  Effective batch size = {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
        trainer = BucketedTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=data_collator,
            callbacks=callbacks,
            bucket_batch_size=config.batch_size,
        )
    else:
        # Standard trainer for batch_size=1
        print(f"\nUsing standard Trainer with batch_size=1")
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
    model_processor.save_pretrained(str(output_dir))
    
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
    # Default values from TrainConfig
    defaults = TrainConfig()
    
    parser = argparse.ArgumentParser(
        description="Fine-tune Sesame CSM-1B with memory-efficient dataset loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file.",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=defaults.dataset_path,
        help="Path to the raw dataset directory (containing de/ and en/ folders).",
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        default=defaults.preprocessed_path,
        help="Path to preprocessed dataset (created by preprocess_dataset.py). Use this instead of dataset_path for faster training.",
    )
    parser.add_argument(
        "--num_context_turns",
        type=int,
        default=defaults.num_context_turns,
        help="Number of previous conversation turns to include as context.",
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=defaults.max_audio_seconds,
        help="Maximum audio length in seconds (longer samples are skipped).",
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
    parser.add_argument(
        "--context_length",
        type=int,
        default=defaults.context_length,
        help="Max context length (tokens).",
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
        help="Path to example folder for generation (e.g. train_csm/example_gen).",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=defaults.use_wandb,
        help="Enable Weights & Biases tracking.",
    )
    
    args = parser.parse_args()
    
    # Start with defaults
    config_dict = {
        "dataset_path": defaults.dataset_path,
        "preprocessed_path": defaults.preprocessed_path,
        "context_length": defaults.context_length,
        "num_context_turns": defaults.num_context_turns,
        "max_audio_seconds": defaults.max_audio_seconds,
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
        "target_sampling_rate": defaults.target_sampling_rate,
        "audio_model_sampling_rate": defaults.audio_model_sampling_rate,
        "text_max_length": defaults.text_max_length,
    }
    
    # Load from YAML config if provided
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        # Map YAML keys to config keys
        yaml_mapping = {
            "batch_size": "batch_size",
            "grad_acc_steps": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "max_grad_norm": "max_grad_norm",
            "warmup_steps": "warmup_steps",
            "weight_decay": "weight_decay",
            "lr_decay": "lr_scheduler_type",
            "num_context_turns": "num_context_turns",
            "context_length": "context_length",
            "max_audio_seconds": "max_audio_seconds",
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
    
    # Override with CLI arguments (if explicitly provided)
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key == "config":
            continue
        if key == "n_epochs":
            key = "num_train_epochs"
        if value is not None:
            # Check if this was explicitly provided (not just default)
            # We need to check against argparse defaults
            if key in config_dict:
                config_dict[key] = value
    
    # Validate required arguments - need either dataset_path or preprocessed_path
    if config_dict.get("dataset_path") is None and config_dict.get("preprocessed_path") is None:
        parser.error("Either --dataset_path or --preprocessed_path is required")
    
    if config_dict.get("preprocessed_path") is not None:
        print("Using preprocessed dataset (ignoring --dataset_path if provided)")
    
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

