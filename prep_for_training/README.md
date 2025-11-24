# Data Preparation for Sesame CSM Training

This directory contains the pipeline for preparing synthetic conversational datasets for training the Sesame CSM (1B) model.

## Overview

The `prepare_dataset_for_training.py` script processes raw audio and transcripts into a format suitable for training. It handles:
*   **Dynamic Audio Length**: Scans the dataset to find the maximum audio duration.
*   **Context Handling**: Includes previous conversation turns (text and audio) as context for each training sample.
*   **Batch Processing**: Saves processed data in shards to manage memory usage.
*   **Resumability**: Can resume processing from the last saved shard.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Usage

### Prepare Dataset

Run the script to process your dataset:

```bash
uv run prepare_dataset_for_training.py \
    --input_dir ../synthetic-dataset \
    --output_dir ./prepared_dataset \
    --batch_size 100 \
    --max_history 5
```

### Arguments

*   `--input_dir`: Path to the root of the synthetic dataset (containing language folders like `en`).
*   `--output_dir`: Directory where the processed Arrow shards will be saved.
*   `--batch_size`: Number of conversations to process per batch (default: 100).
*   `--max_history`: Maximum number of previous turns to include as context (default: None, meaning all available history).
*   `--resume`: Flag to resume processing if it was interrupted.
*   `--target_sampling_rate`: Target sampling rate for audio (default: 24000).
*   `--model_id`: Hugging Face model ID for the processor (default: "unsloth/csm-1b").

## Output Structure

The script produces:
1.  `prepared_dataset/shard_*.arrow`: Sharded dataset files containing `input_ids`, `attention_mask`, and `labels`.
2.  `prepared_dataset/max_audio_length.txt`: Cached maximum audio length found in the dataset.

## Testing

You can generate dummy data and verify the script using:

```bash
uv run create_dummy_data.py
uv run prepare_dataset_for_training.py --input_dir test_dataset_dummy --output_dir test_output --batch_size 1
uv run verify_shard.py
```
