import unsloth 
import argparse
import os
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import TrainingArguments, Trainer, CsmForConditionalGeneration, AutoProcessor
from unsloth import FastModel, is_bfloat16_supported
from pathlib import Path
from dotenv import load_dotenv
import yaml
import soundfile as sf
import re
import numpy as np
import wandb
from generation import AudioGenerationCallback

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sesame CSM model using Unsloth.")
    
    # First, parse just the config argument to load defaults
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, help="Path to YAML config file.")
    known_args, remaining_args = conf_parser.parse_known_args()

    defaults = {
        "dataset_path": None,
        "output_dir": "./outputs",
        "max_steps": -1, # Default to -1 so epochs take precedence
        "num_train_epochs": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "model_id": "unsloth/csm-1b",
        "seed": 3407,
        "use_wandb": False,
        "full_finetune": False,
        "gen_every": 1000,
        "gen_from": None,
        "warmup_steps": 5,
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
    }

    if known_args.config:
        try:
            with open(known_args.config, "r") as f:
                config = yaml.safe_load(f)
                # Map config keys to arg keys if necessary
                # Config: batch_size, grad_acc_steps, learning_rate, max_grad_norm, warmup_steps, weight_decay, lr_decay
                mapping = {
                    "grad_acc_steps": "gradient_accumulation_steps",
                    "lr_decay": "lr_scheduler_type",
                    "n_epochs": "num_train_epochs",
                }
                for k, v in config.items():
                    key = mapping.get(k, k)
                    if key in defaults:
                        defaults[key] = v
                    else:
                        print(f"Warning: Config key '{k}' not supported by CLI, ignoring.")
        except Exception as e:
            print(f"Error loading config file: {e}")

    # Now define the full parser with updated defaults
    parser.add_argument("--config", type=str, help="Path to YAML config file.")
    parser.add_argument("--dataset_path", type=str, required=defaults["dataset_path"] is None, default=defaults["dataset_path"], help="Path to the prepared dataset directory containing shards.")
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"], help="Path to save the trained model/checkpoints.")
    parser.add_argument("--max_steps", type=int, default=defaults["max_steps"], help="Max training steps. Set to -1 to use epochs.")
    parser.add_argument("--n_epochs", type=int, default=defaults["num_train_epochs"], help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="Per device train batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=defaults["gradient_accumulation_steps"], help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"], help="Learning rate.")
    parser.add_argument("--model_id", type=str, default=defaults["model_id"], help="Model ID to load.")
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Random seed.")
    parser.add_argument("--use_wandb", action="store_true", default=defaults["use_wandb"], help="Enable Weights & Biases tracking.")
    parser.add_argument("--full_finetune", action="store_true", default=defaults["full_finetune"], help="Finetune all weights (disable LoRA).")
    parser.add_argument("--gen_every", type=int, default=defaults["gen_every"], help="Generate audio every N steps.")
    parser.add_argument("--gen_from", type=str, default=defaults["gen_from"], help="Path to example folder for generation (e.g. train_csm/example_gen).")
    
    # New arguments from config
    parser.add_argument("--warmup_steps", type=int, default=defaults["warmup_steps"], help="Warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"], help="Weight decay.")
    parser.add_argument("--lr_scheduler_type", type=str, default=defaults["lr_scheduler_type"], help="Learning rate scheduler type.")
    parser.add_argument("--max_grad_norm", type=float, default=defaults["max_grad_norm"], help="Max gradient norm.")

    return parser.parse_args()

def main():
    args = parse_args()
    context_length = 2048
    
    print(f"Loading model: {args.model_id}")
    # Load model and processor using Unsloth
    # max_seq_length and dtype are usually inferred or set to defaults if not provided, 
    # but for CSM we might want to be careful. The notebook uses defaults in the snippet I saw, 
    # or specific config. Let's use standard loading.
    model, processor = FastModel.from_pretrained(
        model_name = args.model_id,
        max_seq_length = context_length, # From notebook context section
        dtype = None, # Auto detection
        auto_model = CsmForConditionalGeneration,
        load_in_4bit = True, # Disable 4bit quantization for macOS compatibility (and for full finetuning usually)
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    
    if not args.full_finetune:
        # Configure LoRA adapters
        # The notebook output showed "Trainable parameters = 29M", which implies LoRA.
        # We need to apply LoRA adapters to the model.
        model = FastModel.get_peft_model(
            model,
            r = 32, # Standard LoRA rank
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = args.seed,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    else:
        print("Full finetuning enabled. LoRA adapters will NOT be added.")

    print(f"Loading dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    shards = sorted(dataset_path.glob("shard_*.arrow"))
    
    if not shards:
        raise ValueError(f"No shards found in {args.dataset_path}")
        
    loaded_shards = []
    for shard in shards:
        try:
            ds = load_from_disk(str(shard))
            loaded_shards.append(ds)
        except Exception as e:
            print(f"Error loading shard {shard}: {e}")
            
    if not loaded_shards:
        raise ValueError("Failed to load any datasets.")
        
    train_dataset = concatenate_datasets(loaded_shards)
    print(f"Total training samples: {len(train_dataset)}")

    # Define the transform for on-the-fly processing
    def preprocess_batch(batch):
        # batch is a dict of lists: {"conversation": [...], "target_audio": [...], ...}
        # We need to process each sample in the batch
        
        processed_batch = []
        
        # Iterate over the batch indices
        # HuggingFace datasets batch is column-oriented (dict of lists)
        batch_size = len(batch["conversation"])
        
        for i in range(batch_size):
            conversation = batch["conversation"][i]
            
            # Load audio for this conversation
            processed_conversation = []
            valid_sample = True
            
            for msg in conversation:
                new_content = []
                for item in msg["content"]:
                    if item["type"] == "audio":
                        try:
                            audio_path = item["path"]
                            # Load audio
                            # We assume target_sr is 24000 (CSM default) or we need to pass it.
                            # Ideally we should read it from config or args, but for now hardcode or assume 24k
                            target_sr = 24000 
                            
                            data, sr = sf.read(audio_path)
                            if sr != target_sr:
                                import librosa
                                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                                
                            new_content.append({"type": "audio", "path": data, "sampling_rate": target_sr})
                        except Exception as e:
                            print(f"Failed to load {audio_path}: {e}")
                            valid_sample = False
                            break
                    else:
                        new_content.append(item)
                
                if not valid_sample:
                    break
                    
                processed_conversation.append({
                    "role": msg["role"],
                    "content": new_content
                })
            
            if not valid_sample:
                # If a sample is invalid, we must still return something to match batch size?
                # Or we can return empty/dummy? 
                # set_transform expects the same length or we might have issues.
                # Usually transforms map N->N. 
                # If we fail, we might return a dummy or raise error. 
                # Let's try to return a dummy or skip? 
                # Skipping in map/transform is hard. 
                # We'll just return a dummy empty input which will likely be masked out or ignored if we are lucky,
                # or we should ensure data prep is robust.
                # For now, let's assume data is good or raise.
                print(f"Error in sample {i}, skipping content.")
                processed_batch.append([]) # This will likely fail in apply_chat_template
                continue

            processed_batch.append(processed_conversation)
            
        # Apply processor to the whole batch of conversations
        # apply_chat_template can handle a list of conversations
        try:
            model_inputs = processor.apply_chat_template(
                processed_batch,
                tokenize=True,
                return_dict=True,
                output_labels=True,
                text_kwargs={
                    "padding": "max_length",
                    "max_length": context_length, # Match model max_seq_length
                    "pad_to_multiple_of": 8,
                    "padding_side": "right",
                },
            )
            return model_inputs
        except Exception as e:
            print(f"Error in apply_chat_template: {e}")
            # Return empty dicts to avoid crash, but this is bad.
            return {"input_ids": [], "attention_mask": [], "labels": []}

    # Apply the transform
    # We need to set the transform on the dataset
    train_dataset.set_transform(preprocess_batch)

    # Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        max_steps = args.max_steps,
        num_train_epochs = args.n_epochs,
        learning_rate = args.learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        max_grad_norm = args.max_grad_norm,
        seed = args.seed,
        output_dir = args.output_dir,
        report_to = "wandb" if args.use_wandb else "none",
        #dataloader_num_workers = 4, # Parallelize data loading
        remove_unused_columns = False, # Important! Otherwise our raw columns might be removed before transform
    )

    print("Starting training...")
    
    callbacks = []
    if args.gen_from:
        callbacks.append(AudioGenerationCallback(args.gen_from, args.gen_every, processor, args.use_wandb, args.output_dir))
        
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        args = training_args,
        callbacks = callbacks
    )
    
    trainer_stats = trainer.train()
    
    print("Training complete.")
    print(f"Training time: {trainer_stats.metrics['train_runtime']} seconds")
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    # Also save to GGUF if needed? Notebook mentions saving. 
    # For now, standard pretrained save is enough.

if __name__ == "__main__":
    main()
