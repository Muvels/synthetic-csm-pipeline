#!/usr/bin/env python3
"""
Preprocess dataset for CSM training.

This script tokenizes audio/text pairs and saves them to disk,
so training doesn't need to load audio - just pre-tokenized tensors.

Processes one conversation at a time to minimize RAM usage.

Usage:
    python preprocess_dataset.py --input_path ./dataset --output_path ./preprocessed
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import yaml
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoProcessor


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing."""
    input_path: str = None  # Raw dataset path
    output_path: str = "./preprocessed"  # Where to save processed data
    model_id: str = "unsloth/csm-1b"  # Model for processor
    num_context_turns: int = 3  # Number of context turns to include
    max_audio_seconds: float = 10.0  # Max audio length per segment
    min_audio_seconds: float = 0.5  # Min audio length
    text_max_length: int = 256  # Max text tokens
    audio_sampling_rate: int = 24000  # CSM sampling rate
    num_workers: int = 1  # Number of parallel workers (1 = sequential)


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_script_file(script_path: Path) -> list[dict]:
    """Parse vibevoice-podcast-script.txt file."""
    lines = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: [N]: text  where N is speaker ID
            match = re.match(r"\[(\d+)\]:\s*(.+)", line)
            if match:
                # CSM expects speaker ID as integer string like "0", "1"
                # Convert from 1-indexed to 0-indexed
                speaker_id = str(int(match.group(1)) - 1)
                lines.append({
                    "speaker": speaker_id,
                    "text": match.group(2).strip(),
                })
    return lines


def scan_conversations(input_path: Path) -> list[dict]:
    """
    Scan dataset and return list of conversations.
    Each conversation is a dict with 'path' and 'samples' (list of audio/text pairs).
    """
    conversations = []
    
    # Handle nested structure: lang/conv_name/segments
    for lang_dir in input_path.iterdir():
        if not lang_dir.is_dir():
            continue
        
        # Check if this is a language dir (de, en) or direct conversation dir
        segments_dir = lang_dir / "segments"
        if segments_dir.exists():
            # Flat structure: input_path/conv_name/segments
            conv_dirs = [lang_dir]
        else:
            # Nested structure: input_path/lang/conv_name/segments
            conv_dirs = [d for d in lang_dir.iterdir() if d.is_dir() and (d / "segments").exists()]
        
        print(f"Found {len(conv_dirs)} conversation directories")
        print(f"Conversation directories: {conv_dirs}")
        
        for conv_dir in conv_dirs:
            seg_dir = conv_dir / "segments"
            script_path = seg_dir / "vibevoice-podcast-script.txt"
            
            if not script_path.exists():
                # Try parent directory
                script_path = conv_dir / "vibevoice-podcast-script.txt"
            
            if not script_path.exists():
                continue
            
            script_lines = parse_script_file(script_path)
            wav_files = sorted(seg_dir.glob("*.wav"))
            
            if len(wav_files) != len(script_lines):
                print(f"Warning: {conv_dir.name}: {len(wav_files)} wav files vs {len(script_lines)} script lines")
                continue
            
            samples = []
            for wav_path, line in zip(wav_files, script_lines):
                samples.append({
                    "audio_path": str(wav_path),
                    "text": line["text"],
                    "speaker": line["speaker"],
                })
            
            conversations.append({
                "id": conv_dir.name,
                "path": str(conv_dir),
                "samples": samples,
            })
    
    return conversations


def load_and_resample_audio(audio_path: str, target_sr: int) -> tuple:
    """Load audio and resample to target sample rate."""
    import librosa
    import numpy as np
    
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio.astype(np.float32), target_sr


def process_conversation(
    conversation: dict,
    config: PreprocessConfig,
    processor: AutoProcessor,
    output_dir: Path,
) -> list[dict]:
    """
    Process a single conversation and save tokenized samples.
    Returns list of manifest entries.
    """
    import numpy as np
    
    samples = conversation["samples"]
    conv_id = conversation["id"]
    manifest_entries = []
    
    # Create output directory for this conversation
    conv_output_dir = output_dir / conv_id
    conv_output_dir.mkdir(parents=True, exist_ok=True)
    
    # First pass: load and filter audio by duration
    valid_samples = []
    audio_cache = {}  # Cache loaded audio for context reuse
    
    for i, sample in enumerate(samples):
        try:
            audio_info = sf.info(sample["audio_path"])
            duration = audio_info.frames / audio_info.samplerate
            
            if config.min_audio_seconds <= duration <= config.max_audio_seconds:
                valid_samples.append((i, sample, duration))
        except Exception as e:
            print(f"  Warning: Could not read {sample['audio_path']}: {e}")
    
    if not valid_samples:
        return []
    
    # Calculate max audio samples for context
    max_total_audio_seconds = (config.num_context_turns + 1) * config.max_audio_seconds
    max_audio_samples = int(config.audio_sampling_rate * max_total_audio_seconds)
    
    # Second pass: process each sample with context
    for sample_idx, (original_idx, sample, duration) in enumerate(valid_samples):
        # Build conversation with context
        conversation_turns = []
        
        # Get context samples (previous valid samples)
        context_start = max(0, sample_idx - config.num_context_turns)
        context_samples = valid_samples[context_start:sample_idx]
        
        # Load context audio (use cache to avoid reloading)
        for ctx_orig_idx, ctx_sample, _ in context_samples:
            if ctx_sample["audio_path"] not in audio_cache:
                try:
                    audio, _ = load_and_resample_audio(
                        ctx_sample["audio_path"], 
                        config.audio_sampling_rate
                    )
                    audio_cache[ctx_sample["audio_path"]] = audio
                except Exception:
                    continue
            
            ctx_audio = audio_cache.get(ctx_sample["audio_path"])
            if ctx_audio is not None:
                conversation_turns.append({
                    "role": ctx_sample["speaker"],
                    "content": [
                        {"type": "text", "text": ctx_sample["text"]},
                        {"type": "audio", "path": ctx_audio},
                    ],
                })
        
        # Load current sample audio
        if sample["audio_path"] not in audio_cache:
            try:
                audio, _ = load_and_resample_audio(
                    sample["audio_path"],
                    config.audio_sampling_rate
                )
                audio_cache[sample["audio_path"]] = audio
            except Exception as e:
                print(f"  Warning: Failed to load {sample['audio_path']}: {e}")
                continue
        
        current_audio = audio_cache[sample["audio_path"]]
        
        # Add current turn
        conversation_turns.append({
            "role": sample["speaker"],
            "content": [
                {"type": "text", "text": sample["text"]},
                {"type": "audio", "path": current_audio},
            ],
        })
        
        # Process with processor
        try:
            model_inputs = processor.apply_chat_template(
                conversation_turns,
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
                    "sampling_rate": config.audio_sampling_rate,
                    "max_length": max_audio_samples,
                    "padding": "max_length",
                },
                common_kwargs={"return_tensors": "pt"},
            )
        except Exception as e:
            print(f"  Warning: Processor failed for {sample['audio_path']}: {e}")
            continue
        
        # Extract tensors (remove batch dimension)
        tensors = {}
        for key in ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs", "codebook_labels"]:
            if key in model_inputs:
                val = model_inputs[key]
                if isinstance(val, torch.Tensor):
                    tensors[key] = val[0] if val.dim() > 1 else val
                elif isinstance(val, list):
                    tensors[key] = torch.tensor(val[0] if isinstance(val[0], list) else val)
        
        # Add metadata
        num_audio_segments = len(conversation_turns)
        tensors["num_audio_segments"] = torch.tensor(num_audio_segments, dtype=torch.long)
        
        # Save to safetensors
        sample_filename = f"{sample_idx:05d}.safetensors"
        sample_path = conv_output_dir / sample_filename
        save_file(tensors, sample_path)
        
        # Add manifest entry
        manifest_entries.append({
            "path": str(sample_path.relative_to(output_dir)),
            "conv_id": conv_id,
            "sample_idx": sample_idx,
            "original_idx": original_idx,
            "num_audio_segments": num_audio_segments,
            "text": sample["text"][:100],  # First 100 chars for reference
            "speaker": sample["speaker"],
        })
        
        # Clear old audio from cache to save memory
        # Keep only the last N+1 samples for potential context reuse
        keep_paths = set()
        for _, s, _ in valid_samples[max(0, sample_idx - config.num_context_turns):sample_idx + 1]:
            keep_paths.add(s["audio_path"])
        
        for path in list(audio_cache.keys()):
            if path not in keep_paths:
                del audio_cache[path]
    
    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for CSM training")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Required paths
    parser.add_argument("--input_path", type=str, help="Path to raw dataset")
    parser.add_argument("--output_path", type=str, default="./preprocessed", help="Output directory")
    
    # Processing options
    parser.add_argument("--model_id", type=str, default="unsloth/csm-1b", help="Model ID for processor")
    parser.add_argument("--num_context_turns", type=int, default=3, help="Context turns to include")
    parser.add_argument("--max_audio_seconds", type=float, default=10.0, help="Max audio length per segment")
    parser.add_argument("--min_audio_seconds", type=float, default=0.5, help="Min audio length per segment")
    parser.add_argument("--text_max_length", type=int, default=256, help="Max text tokens")
    parser.add_argument("--audio_sampling_rate", type=int, default=24000, help="Audio sampling rate")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config_dict = {}
    if args.config:
        config_dict = load_config_from_yaml(args.config)
    
    # Override with CLI args
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            config_dict[key] = value
    
    # Create config
    config = PreprocessConfig(**{k: v for k, v in config_dict.items() if k in PreprocessConfig.__dataclass_fields__})
    
    if not config.input_path:
        print("Error: --input_path is required")
        sys.exit(1)
    
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Print config
    print("=" * 60)
    print("CSM Dataset Preprocessing")
    print("=" * 60)
    print(f"Input path:          {input_path}")
    print(f"Output path:         {output_path}")
    print(f"Model ID:            {config.model_id}")
    print(f"Context turns:       {config.num_context_turns}")
    print(f"Audio length:        {config.min_audio_seconds}s - {config.max_audio_seconds}s")
    print(f"Text max length:     {config.text_max_length}")
    print(f"Audio sample rate:   {config.audio_sampling_rate}")
    print("=" * 60)
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(config.model_id)
    
    # Scan conversations
    print("\nScanning dataset...")
    conversations = scan_conversations(input_path)
    print(f"Found {len(conversations)} conversations")
    
    if not conversations:
        print("Error: No conversations found!")
        sys.exit(1)
    
    # Process each conversation
    print("\nProcessing conversations...")
    all_manifest_entries = []
    total_samples = 0
    
    for conv in tqdm(conversations, desc="Conversations"):
        entries = process_conversation(conv, config, processor, output_path)
        all_manifest_entries.extend(entries)
        total_samples += len(entries)
    
    # Save manifest
    manifest_path = output_path / "manifest.json"
    manifest = {
        "config": {
            "model_id": config.model_id,
            "num_context_turns": config.num_context_turns,
            "max_audio_seconds": config.max_audio_seconds,
            "min_audio_seconds": config.min_audio_seconds,
            "text_max_length": config.text_max_length,
            "audio_sampling_rate": config.audio_sampling_rate,
        },
        "num_samples": total_samples,
        "num_conversations": len(conversations),
        "samples": all_manifest_entries,
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save bucket info for training
    bucket_info = {}
    for entry in all_manifest_entries:
        num_seg = entry["num_audio_segments"]
        bucket_info[num_seg] = bucket_info.get(num_seg, 0) + 1
    
    bucket_path = output_path / "bucket_info.json"
    with open(bucket_path, "w") as f:
        json.dump(bucket_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"Total samples:       {total_samples}")
    print(f"Conversations:       {len(conversations)}")
    print(f"Manifest saved to:   {manifest_path}")
    print(f"Bucket info:         {bucket_path}")
    print("\nBucket distribution:")
    for num_seg in sorted(bucket_info.keys()):
        print(f"  {num_seg} audio segment(s): {bucket_info[num_seg]} samples")


if __name__ == "__main__":
    main()

