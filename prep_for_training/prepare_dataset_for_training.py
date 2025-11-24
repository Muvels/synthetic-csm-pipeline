import argparse
import os
import re
import json
import math
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from datasets import Dataset, Audio, concatenate_datasets, load_from_disk
from transformers import AutoProcessor
import numpy as np
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare conversational dataset for Sesame CSM training.")
    parser.add_argument("--input_dir", type=str, default="./synthetic-dataset", help="Path to the synthetic dataset root directory.")
    parser.add_argument("--output_dir", type=str, default="./prepared_dataset", help="Path to save the processed dataset.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of conversations to process before saving a shard.")
    parser.add_argument("--max_history", type=int, default=None, help="Max number of previous turns to include as context. Default: None (all available history).")
    parser.add_argument("--resume", action="store_true", help="Resume from the last saved shard.")
    parser.add_argument("--target_sampling_rate", type=int, default=24000, help="Target sampling rate for audio.")
    parser.add_argument("--model_id", type=str, default="unsloth/csm-1b", help="Model ID for the processor.")
    return parser.parse_args()

def get_max_audio_length(data_root, target_sr):
    """Scans all wav files to find the maximum duration in seconds."""
    print("Scanning dataset for maximum audio length...")
    max_seconds = 0.0
    wav_files = list(Path(data_root).rglob("*.wav"))
    
    for wav_path in tqdm(wav_files, desc="Scanning audio files"):
        try:
            info = sf.info(str(wav_path))
            duration = info.duration
            if duration > max_seconds:
                max_seconds = duration
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            continue
            
    print(f"Maximum audio duration found: {max_seconds:.2f} seconds")
    # Add a small buffer and round up
    return math.ceil(max_seconds + 1.0)

def parse_conversation(uuid_dir):
    """Parses a single conversation directory."""
    segments_dir = uuid_dir / "segments"
    if not segments_dir.is_dir():
        return None

    script_path = segments_dir / "vibevoice-podcast-script.txt"
    if not script_path.is_file():
        return None

    # 1) Parse transcript file
    script_lines = []
    try:
        with script_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Match format: [1]: text... or [2]: text...
                m = re.match(r"\[(\d+)\]:\s*(.*)", line)
                if not m:
                    continue
                speaker_id = m.group(1)
                text = m.group(2)
                script_lines.append({"speaker": speaker_id, "text": text})
    except Exception as e:
        print(f"Error reading script {script_path}: {e}")
        return None

    # 2) Collect wav files
    # Assuming filenames are like 001_speaker1.wav, 002_speaker2.wav
    wav_paths = sorted(p for p in segments_dir.glob("*.wav"))
    
    # Simple validation: check if counts match
    if len(wav_paths) != len(script_lines):
        # Try to match by index if possible, or just truncate to min length
        min_len = min(len(wav_paths), len(script_lines))
        wav_paths = wav_paths[:min_len]
        script_lines = script_lines[:min_len]
    
    conversation_turns = []
    for i, (wav_path, line) in enumerate(zip(wav_paths, script_lines)):
        conversation_turns.append({
            "audio_path": str(wav_path),
            "text": line["text"],
            "speaker": line["speaker"],
            "turn_index": i
        })
        
    return conversation_turns

def create_contextual_samples(conversation_turns, max_history=None):
    """Creates training samples with history context."""
    samples = []
    
    for i, turn in enumerate(conversation_turns):
        # Determine history range
        start_index = 0
        if max_history is not None:
            start_index = max(0, i - max_history)
        
        history = conversation_turns[start_index : i+1]
        
        # Construct the conversation list for the processor
        # Format: [{"role": "speaker_id", "content": [{"type": "text", ...}, {"type": "audio", ...}]}]
        # The target utterance is the last one in the list.
        
        conversation_messages = []
        for h_turn in history:
            conversation_messages.append({
                "role": str(h_turn["speaker"]),
                "content": [
                    {"type": "text", "text": h_turn["text"]},
                    {"type": "audio", "path": h_turn["audio_path"]}
                ]
            })
            
        samples.append({
            "conversation": conversation_messages,
            "target_audio": turn["audio_path"], # For reference/filtering if needed
            "target_text": turn["text"],
            "speaker": turn["speaker"]
        })
        
    return samples

def process_batch_and_save(batch_conversations, processor, target_sr, max_history, output_path):
    """Processes a batch of conversations and saves directly to disk to save RAM."""
    
    def generator():
        for conv_dir in batch_conversations:
            turns = parse_conversation(conv_dir)
            if not turns:
                continue
                
            conv_samples = create_contextual_samples(turns, max_history)
            
            for sample in conv_samples:
                # Yield the raw sample directly.
                # The training script will handle audio loading and tokenization.
                # sample structure: {"conversation": [...], "target_audio": ..., "target_text": ..., "speaker": ...}
                yield sample
            
            # No need for explicit gc.collect() as much if we aren't loading audio, 
            # but keeping it doesn't hurt for the loop variables.
            # gc.collect()

    try:
        # Use from_generator to stream processing
        # This writes to a cache file on disk instead of holding everything in RAM
        # writer_batch_size=1 ensures we flush to disk after EVERY sample, keeping RAM usage low.
        ds = Dataset.from_generator(generator, writer_batch_size=1)
        
        if len(ds) == 0:
            return False
            
        ds.save_to_disk(str(output_path))
        return True
        
    except Exception as e:
        print(f"Error creating dataset from generator: {e}")
        # If it's an empty dataset error, we might want to handle it, but from_generator usually handles empty gen?
        # It might raise error if features are not inferred.
        # But we can't easily infer features without data.
        raise e

def main():
    args = parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing shards to resume
    existing_shards = sorted(output_dir.glob("shard_*.arrow"))
    start_shard_index = 0
    processed_uuids = set()
    
    if args.resume and existing_shards:
        print(f"Resuming from {len(existing_shards)} existing shards...")
        start_shard_index = len(existing_shards)
        # Ideally we should know which UUIDs are processed. 
        # For simplicity in this script, we might just skip the first N conversations 
        # corresponding to batch_size * start_shard_index.
        # This assumes deterministic ordering.
    
    # 1. Scan for Max Length
    # We can cache this value to a file to avoid rescanning on resume
    max_len_file = output_dir / "max_audio_length.txt"
    if max_len_file.exists():
        with open(max_len_file, "r") as f:
            max_audio_seconds = float(f.read().strip())
        print(f"Loaded max audio length: {max_audio_seconds}s")
    else:
        max_audio_seconds = get_max_audio_length(input_dir, args.target_sampling_rate)
        with open(max_len_file, "w") as f:
            f.write(str(max_audio_seconds))
            
    max_audio_samples = int(max_audio_seconds * args.target_sampling_rate)
    print(f"Max audio samples: {max_audio_samples}")

    # 2. Initialize Processor
    print(f"Loading processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 3. List all conversations
    # We assume each subdirectory in input_dir/en/CATEGORY/UUID is a conversation?
    # The notebook iterates: `for uuid_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):`
    # But the user's structure seems to be `synthetic-dataset/en/CATEGORY/UUID`.
    # Let's find all UUID directories (directories containing 'segments' folder).
    
    print("Locating conversations...")
    all_conv_dirs = sorted([p.parent for p in input_dir.rglob("segments") if p.is_dir()])
    print(f"Found {len(all_conv_dirs)} conversations.")
    
    # 4. Process in batches
    total_convs = len(all_conv_dirs)
    
    # Skip already processed if resuming
    start_idx = start_shard_index * args.batch_size
    if start_idx >= total_convs:
        print("All conversations already processed.")
        return

    for batch_idx, i in enumerate(range(start_idx, total_convs, args.batch_size)):
        current_shard_index = start_shard_index + batch_idx
        batch_dirs = all_conv_dirs[i : i + args.batch_size]
        
        print(f"Processing batch {current_shard_index} ({i}/{total_convs})...")
        
        try:
            shard_path = output_dir / f"shard_{current_shard_index}.arrow"
            success = process_batch_and_save(
                batch_dirs, 
                processor, 
                args.target_sampling_rate, 
                args.max_history,
                shard_path
            )
            
            if success:
                print(f"Saved shard to {shard_path}")
            else:
                print("Batch resulted in no valid samples.")
                
        except Exception as e:
            print(f"Critical error processing batch {current_shard_index}: {e}")
            # We might want to continue or exit? Exit is safer to avoid corruption.
            raise e

    print("Data preparation complete.")

if __name__ == "__main__":
    main()
